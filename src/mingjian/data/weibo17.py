"""Clean-room adapter for the locally archived EANN Weibo17 dataset.

The adapter consumes the ``weibo.zip`` archive published by the EANN authors
and produces a local, never-committed training view:

* official splits come only from ``train_id.pickle`` / ``validate_id.pickle`` /
  ``test_id.pickle`` -- the four text file names are *not* split boundaries;
* labels follow the EANN protocol ``0 = real / non-rumor`` and
  ``1 = fake / rumor``;
* one image per post is selected with the documented first-match rule: walk the
  post image URLs in order and take the first URL whose filename stem exists in
  the archive image index;
* selected images are extracted next to the JSONL files;
* the manifest stores aggregate statistics and file hashes only.

The parsing rules were derived from this project's own archive-format audit
(in ``work/weibo17_verification``).  No EANN loading or preprocessing source
code is copied.  Raw text, tweet IDs, image URLs and images stay local and must
never be committed to Git.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import write_jsonl
from .schema import NewsSample

ARCHIVE_FILE_NAME = "weibo.zip"
ARCHIVE_SIZE_BYTES = 1_359_648_341
ARCHIVE_SHA256 = "06b4d5edc978ab385a75bdc064cb2d2a200e09953ac38ecc1b1e80d652381f68"
SOURCE = "weibo17-eann"

# (official pickle stem, output split name)
SPLIT_SPECS: tuple[tuple[str, str], ...] = (
    ("train_id", "train"),
    ("validate_id", "val"),
    ("test_id", "test"),
)

# (text member, label); 0 = real/non-rumor, 1 = fake/rumor
TEXT_SPECS: tuple[tuple[str, int], ...] = (
    ("tweets/train_nonrumor.txt", 0),
    ("tweets/train_rumor.txt", 1),
    ("tweets/test_nonrumor.txt", 0),
    ("tweets/test_rumor.txt", 1),
)

# (archive prefix, category)
IMAGE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("nonrumor_images/", "real"),
    ("rumor_images/", "fake"),
)

CATEGORY_BY_LABEL = {0: "real", 1: "fake"}

EXPECTED_TOTAL = 7_723
EXPECTED_BY_SPLIT = {"train": 5_415, "val": 843, "test": 1_465}
EXPECTED_BY_LABEL = {"real": 3_615, "fake": 4_108}
EXPECTED_BY_SPLIT_LABEL = {
    "train": {"real": 2_517, "fake": 2_898},
    "val": {"real": 389, "fake": 454},
    "test": {"real": 709, "fake": 756},
}

_OUTPUT_SPLIT_ORDER = ("train", "val", "test")


class Weibo17ArchiveError(RuntimeError):
    """Raised when the local Weibo17 archive does not match the data contract."""


@dataclass(frozen=True)
class ParsedPost:
    """One post parsed from the four three-line text members."""

    sample_id: str
    text: str
    label: int
    image_urls: tuple[str, ...]
    source_file: str


@dataclass(frozen=True)
class ArchivedImage:
    """One image member inside the archive."""

    member: str
    category: str
    stem: str


@dataclass(frozen=True)
class ImageChoice:
    """The image selected for one post, with transparency metadata."""

    image: ArchivedImage
    url: str
    match_count: int
    first_match_position: int
    unselected_urls: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 8 MiB chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_key(name: str) -> str:
    """Return the lower-cased filename stem used by the archive image index."""

    return name.split("/")[-1].split(".")[0].strip().lower()


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


class _NumpySafeUnpickler(pickle.Unpickler):
    """Restrict the official split pickles to builtins and numpy globals."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" or module == "numpy" or module.startswith("numpy."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global: {module}.{name}")


def _load_split_map(archive: zipfile.ZipFile) -> dict[str, dict[str, int]]:
    splits: dict[str, dict[str, int]] = {}
    for stem, output_split in SPLIT_SPECS:
        member = f"{stem}.pickle"
        try:
            raw = archive.read(member)
        except KeyError as exc:  # pragma: no cover - defensive
            raise Weibo17ArchiveError(f"archive is missing {member}") from exc
        try:
            obj = _NumpySafeUnpickler(io.BytesIO(raw)).load()
        except Exception as exc:
            raise Weibo17ArchiveError(f"cannot unpickle {member}: {exc}") from exc
        if not isinstance(obj, dict):
            raise Weibo17ArchiveError(f"{member} must contain a dict, got {type(obj).__name__}")
        mapping: dict[str, int] = {}
        for key, value in obj.items():
            if not isinstance(key, str) or not key.strip():
                raise Weibo17ArchiveError(f"{member} contains a non-string or blank id")
            try:
                event = int(value)
            except (TypeError, ValueError) as exc:
                raise Weibo17ArchiveError(f"{member} contains a non-integer event id") from exc
            mapping[key] = event
        splits[output_split] = mapping
    names = sorted(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise Weibo17ArchiveError(
                    f"official split ids overlap between {left} and {right}"
                )
    return splits


def iter_posts(archive: zipfile.ZipFile) -> Iterator[ParsedPost]:
    """Yield every post from the four official text members.

    Each record is exactly three lines: metadata, image URLs, body text.  The
    metadata line carries 15 pipe-separated fields and field 0 is the tweet id.
    """

    seen: set[str] = set()
    for member, label in TEXT_SPECS:
        try:
            raw = archive.read(member)
        except KeyError as exc:  # pragma: no cover - defensive
            raise Weibo17ArchiveError(f"archive is missing {member}") from exc
        text = raw.decode("utf-8-sig")
        lines = text.splitlines(keepends=True)
        if len(lines) % 3 != 0:
            raise Weibo17ArchiveError(
                f"{member}: line count {len(lines)} is not a multiple of three"
            )
        for offset in range(0, len(lines), 3):
            meta_line, image_line, body_line = lines[offset : offset + 3]
            record_index = offset // 3
            fields = meta_line.rstrip("\r\n").split("|")
            if len(fields) != 15:
                raise Weibo17ArchiveError(
                    f"{member}: record {record_index} has {len(fields)} metadata fields, expected 15"
                )
            sample_id = fields[0].strip()
            if not sample_id:
                raise Weibo17ArchiveError(f"{member}: record {record_index} has a blank id")
            if sample_id in seen:
                raise Weibo17ArchiveError(f"{member}: duplicate sample id in the text pool")
            seen.add(sample_id)
            image_urls = tuple(
                token.strip()
                for token in image_line.rstrip("\r\n").split("|")
                if token.strip() and token.strip().lower() != "null"
            )
            yield ParsedPost(
                sample_id=sample_id,
                text=body_line.rstrip("\r\n"),
                label=label,
                image_urls=image_urls,
                source_file=member,
            )


def index_archive_images(archive: zipfile.ZipFile) -> dict[str, ArchivedImage]:
    """Index archive images by filename stem across both categories."""

    index: dict[str, ArchivedImage] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        for prefix, category in IMAGE_PREFIXES:
            if not info.filename.startswith(prefix):
                continue
            stem = image_key(info.filename)
            if not stem:
                raise Weibo17ArchiveError(f"archive image has an empty filename stem: {prefix}")
            if stem in index:
                raise Weibo17ArchiveError(f"duplicate archive image filename stem: {stem}")
            index[stem] = ArchivedImage(
                member=info.filename, category=category, stem=stem
            )
            break
    if not index:
        raise Weibo17ArchiveError("archive contains no images under nonrumor_images/ or rumor_images/")
    return index


def choose_image(post: ParsedPost, image_index: Mapping[str, ArchivedImage]) -> ImageChoice | None:
    """Select the first matching image URL, mirroring the published rule."""

    matches: list[tuple[int, ArchivedImage, str]] = []
    for position, url in enumerate(post.image_urls, start=1):
        image = image_index.get(image_key(url))
        if image is not None:
            matches.append((position, image, url))
    if not matches:
        return None
    first_position, first_image, first_url = matches[0]
    return ImageChoice(
        image=first_image,
        url=first_url,
        match_count=len(matches),
        first_match_position=first_position,
        unselected_urls=tuple(match[2] for match in matches[1:]),
    )


def _duplicate_audit(samples: list[NewsSample]) -> dict[str, int]:
    counts = Counter(_normalise_text(sample.text) for sample in samples)
    groups = sum(1 for value in counts.values() if value > 1)
    extra = sum(value - 1 for value in counts.values() if value > 1)
    per_split: dict[str, set[str]] = {split: set() for split in _OUTPUT_SPLIT_ORDER}
    for sample in samples:
        per_split[sample.split].add(_normalise_text(sample.text))
    cross_split = 0
    names = list(_OUTPUT_SPLIT_ORDER)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            cross_split += len(per_split[left] & per_split[right])
    return {"groups": groups, "extra_records": extra, "cross_split_groups": cross_split}


def build_weibo17_dataset(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    verify_integrity: bool = True,
    extract_all_images: bool = False,
) -> dict[str, Any]:
    """Build the local Weibo17 JSONL view and return the aggregate manifest."""

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    if not archive_path.is_file():
        raise Weibo17ArchiveError(f"archive not found: {archive_path}")

    size = archive_path.stat().st_size
    digest: str | None = None
    if verify_integrity:
        digest = sha256_file(archive_path)
        if size != ARCHIVE_SIZE_BYTES or digest != ARCHIVE_SHA256:
            raise Weibo17ArchiveError(
                "archive integrity mismatch: expected size "
                f"{ARCHIVE_SIZE_BYTES} and sha256 {ARCHIVE_SHA256}"
            )

    with zipfile.ZipFile(archive_path) as archive:
        splits = _load_split_map(archive)
        posts = {post.sample_id: post for post in iter_posts(archive)}
        image_index = index_archive_images(archive)

        official_ids: set[str] = set()
        for mapping in splits.values():
            official_ids.update(mapping)

        missing_ids = official_ids - set(posts)
        if missing_ids:
            raise Weibo17ArchiveError(
                f"{len(missing_ids)} official ids have no matching text record"
            )

        samples: list[NewsSample] = []
        selected_images: dict[str, str] = {}
        single_match = 0
        multi_match = 0
        position_distribution: Counter[int] = Counter()
        category_counter: Counter[str] = Counter()

        for output_split in _OUTPUT_SPLIT_ORDER:
            mapping = splits[output_split]
            for sample_id in sorted(mapping):
                post = posts[sample_id]
                if not post.text.strip():
                    raise Weibo17ArchiveError("an official record has empty body text")
                choice = choose_image(post, image_index)
                if choice is None:
                    raise Weibo17ArchiveError(
                        "an official record has no matching archive image"
                    )
                relative_path = (
                    f"images/{choice.image.category}/{Path(choice.image.member).name}"
                )
                selected_images[choice.image.member] = relative_path
                if choice.match_count == 1:
                    single_match += 1
                else:
                    multi_match += 1
                position_distribution[choice.first_match_position] += 1
                category_counter[choice.image.category] += 1
                metadata = {
                    "event_id": int(mapping[sample_id]),
                    "source_file": post.source_file,
                    "image_url": choice.url,
                    "image_archive_entry": choice.image.member,
                    "image_match_count": choice.match_count,
                    "image_first_match_position": choice.first_match_position,
                    "image_unselected_url_count": len(choice.unselected_urls),
                    "image_unselected_urls": list(choice.unselected_urls),
                }
                samples.append(
                    NewsSample(
                        sample_id=post.sample_id,
                        text=post.text,
                        image_path=relative_path,
                        label=post.label,
                        split=output_split,
                        source=SOURCE,
                        metadata=metadata,
                    )
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_outputs: dict[str, dict[str, Any]] = {}
        for output_split in _OUTPUT_SPLIT_ORDER:
            rows = [sample for sample in samples if sample.split == output_split]
            final_path = output_dir / f"{output_split}.jsonl"
            temp_path = output_dir / f"{output_split}.jsonl.tmp"
            write_jsonl(rows, temp_path)
            temp_path.replace(final_path)
            jsonl_outputs[output_split] = {
                "path": final_path.name,
                "rows": len(rows),
                "sha256": sha256_file(final_path),
            }

        if extract_all_images:
            targets = {
                image.member: output_dir / "images" / image.category / Path(image.member).name
                for image in image_index.values()
            }
        else:
            targets = {
                member: output_dir / relative for member, relative in selected_images.items()
            }
        for member, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)

    by_split = Counter(sample.split for sample in samples)
    by_label = Counter(CATEGORY_BY_LABEL[sample.label] for sample in samples)
    by_split_label: dict[str, dict[str, int]] = {}
    for split in _OUTPUT_SPLIT_ORDER:
        counter = Counter(
            CATEGORY_BY_LABEL[sample.label] for sample in samples if sample.split == split
        )
        by_split_label[split] = {
            "real": counter["real"],
            "fake": counter["fake"],
        }
    duplicates = _duplicate_audit(samples)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": SOURCE,
        "source_dataset": "Weibo17 / EANN Weibo",
        "archive": {
            "file_name": ARCHIVE_FILE_NAME,
            "size_bytes": size,
            "sha256": digest,
            "integrity_verified": bool(verify_integrity),
        },
        "split_contract": {
            "members": {output: f"{stem}.pickle" for stem, output in SPLIT_SPECS},
            "output_split_mapping": {"validate": "val"},
            "note": "official pickle ids define the split; text file names are not split boundaries",
        },
        "label_contract": {"0": "real/non-rumor", "1": "fake/rumor"},
        "counts": {
            "total": len(samples),
            "by_split": {split: by_split[split] for split in _OUTPUT_SPLIT_ORDER},
            "by_label": {"real": by_label["real"], "fake": by_label["fake"]},
            "by_split_label": by_split_label,
        },
        "text": {
            "unique_ids_in_pool": len(posts),
            "official_ids": len(official_ids),
            "empty_text_official_records": 0,
        },
        "image_matching": {
            "matched_records": len(samples),
            "unmatched_records": 0,
            "single_match_records": single_match,
            "multi_match_records": multi_match,
            "first_match_position_distribution": {
                str(position): count for position, count in sorted(position_distribution.items())
            },
            "selected_image_categories": dict(sorted(category_counter.items())),
            "unique_selected_images": len(selected_images),
            "extracted_images": len(targets),
            "extract_all_images": bool(extract_all_images),
        },
        "duplicates": duplicates,
        "outputs": {"jsonl": jsonl_outputs, "images_dir": "images"},
        "warnings": [
            (
                "official split ids are disjoint but the official pool contains 108 duplicate "
                "text groups / 156 extra records (recomputed on build)"
            ),
            "image names, not image content hashes, are used for matching and reuse accounting",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def assert_official_invariants(manifest: Mapping[str, Any]) -> None:
    """Raise if the manifest deviates from the locked Weibo17 contract."""

    problems: list[str] = []
    counts = manifest["counts"]
    if counts["total"] != EXPECTED_TOTAL:
        problems.append(f"total {counts['total']} != {EXPECTED_TOTAL}")
    if counts["by_split"] != EXPECTED_BY_SPLIT:
        problems.append(f"by_split {counts['by_split']} != {EXPECTED_BY_SPLIT}")
    if counts["by_label"] != EXPECTED_BY_LABEL:
        problems.append(f"by_label {counts['by_label']} != {EXPECTED_BY_LABEL}")
    if counts["by_split_label"] != EXPECTED_BY_SPLIT_LABEL:
        problems.append(f"by_split_label {counts['by_split_label']} != {EXPECTED_BY_SPLIT_LABEL}")
    if manifest["image_matching"]["unmatched_records"] != 0:
        problems.append("some official records have no matching image")
    if manifest["text"]["empty_text_official_records"] != 0:
        problems.append("some official records have empty text")
    if manifest["duplicates"]["cross_split_groups"] != 0:
        problems.append("duplicate text spans official splits")
    if problems:
        raise Weibo17ArchiveError("official Weibo17 invariants failed: " + "; ".join(problems))
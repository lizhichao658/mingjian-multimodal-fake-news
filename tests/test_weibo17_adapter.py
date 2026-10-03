"""Unit tests for the clean-room Weibo17 archive adapter."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pytest

from mingjian.data.weibo17 import (
    Weibo17ArchiveError,
    assert_official_invariants,
    build_weibo17_dataset,
    choose_image,
    image_key,
    index_archive_images,
    iter_posts,
)

META_FIELDS = 15


def _meta_line(sample_id: str) -> str:
    return "|".join([sample_id] + [f"field{i}" for i in range(1, META_FIELDS)])


def _record_block(sample_id: str, image_line: str, body: str) -> str:
    return f"{_meta_line(sample_id)}\n{image_line}\n{body}\n"


def _write_text_member(archive: zipfile.ZipFile, member: str, blocks: list[str]) -> None:
    archive.writestr(member, "".join(blocks).encode("utf-8"))


def _build_fixture(
    path: Path,
    *,
    records: list[dict[str, str | int]],
    splits: dict[str, dict[str, int]],
    images: dict[str, bytes],
) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        grouped: dict[int, list[str]] = {0: [], 1: []}
        for record in records:
            grouped[int(record["label"])].append(
                _record_block(str(record["id"]), str(record["image_line"]), str(record["body"]))
            )
        _write_text_member(archive, "tweets/train_nonrumor.txt", grouped[0])
        _write_text_member(archive, "tweets/train_rumor.txt", grouped[1])
        _write_text_member(archive, "tweets/test_nonrumor.txt", [])
        _write_text_member(archive, "tweets/test_rumor.txt", [])
        pickle_names = {
            "train": "train_id.pickle",
            "validate": "validate_id.pickle",
            "test": "test_id.pickle",
        }
        for split, mapping in splits.items():
            archive.writestr(pickle_names[split], pickle.dumps(mapping))
        for member, payload in images.items():
            archive.writestr(member, payload)
    return path


def _three_records() -> list[dict[str, str | int]]:
    return [
        {"id": "r1", "label": 0, "body": "真实新闻一", "image_line": "https://x/a.jpg"},
        {"id": "r2", "label": 1, "body": "谣言新闻二", "image_line": "null|https://x/b.jpg"},
        {
            "id": "r3",
            "label": 0,
            "body": "真实新闻三",
            "image_line": "https://x/c.jpg|https://x/a.jpg",
        },
    ]


def _three_images() -> dict[str, bytes]:
    return {
        "nonrumor_images/a.jpg": b"a",
        "rumor_images/b.jpg": b"b",
        "nonrumor_images/c.jpg": b"c",
        "nonrumor_images/unused.jpg": b"unused",
    }


def test_image_key_uses_first_dot_of_basename() -> None:
    assert image_key("https://wx1.sinaimg.cn/orj360/abc.jpg") == "abc"
    assert image_key("nonrumor_images/AbC.JPEG") == "abc"


def test_iter_posts_follows_label_direction(tmp_path: Path) -> None:
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=_three_records(),
        splits={"train": {"r1": 0}, "validate": {"r2": 1}, "test": {"r3": 2}},
        images=_three_images(),
    )
    with zipfile.ZipFile(archive_path) as archive:
        posts = {post.sample_id: post for post in iter_posts(archive)}
    assert posts["r1"].label == 0
    assert posts["r2"].label == 1
    assert posts["r3"].label == 0
    # "null" tokens are dropped, the remaining URL order is preserved
    assert posts["r2"].image_urls == ("https://x/b.jpg",)
    assert posts["r3"].image_urls == ("https://x/c.jpg", "https://x/a.jpg")


def test_choose_image_takes_first_match_and_records_alternatives(tmp_path: Path) -> None:
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=_three_records(),
        splits={"train": {"r1": 0}, "validate": {"r2": 1}, "test": {"r3": 2}},
        images=_three_images(),
    )
    with zipfile.ZipFile(archive_path) as archive:
        posts = {post.sample_id: post for post in iter_posts(archive)}
        index = index_archive_images(archive)
    choice = choose_image(posts["r3"], index)
    assert choice is not None
    assert choice.image.member == "nonrumor_images/c.jpg"
    assert choice.match_count == 2
    assert choice.first_match_position == 1
    assert choice.unselected_urls == ("https://x/a.jpg",)
    assert choose_image(posts["r1"], index).match_count == 1


def test_build_synthetic_archive_writes_local_view(tmp_path: Path) -> None:
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=_three_records(),
        splits={"train": {"r1": 0}, "validate": {"r2": 1}, "test": {"r3": 2}},
        images=_three_images(),
    )
    out = tmp_path / "out"
    manifest = build_weibo17_dataset(archive_path, out, verify_integrity=False)

    assert manifest["counts"]["total"] == 3
    assert manifest["counts"]["by_split"] == {"train": 1, "val": 1, "test": 1}
    assert manifest["counts"]["by_label"] == {"real": 2, "fake": 1}
    assert manifest["counts"]["by_split_label"]["val"] == {"real": 0, "fake": 1}
    assert manifest["image_matching"]["matched_records"] == 3
    assert manifest["image_matching"]["unmatched_records"] == 0
    assert manifest["image_matching"]["multi_match_records"] == 1
    assert manifest["image_matching"]["unique_selected_images"] == 3
    assert manifest["image_matching"]["extracted_images"] == 3
    assert manifest["label_contract"] == {"0": "real/non-rumor", "1": "fake/rumor"}

    for split in ("train", "val", "test"):
        assert (out / f"{split}.jsonl").exists()
    assert not (out / "images/real/unused.jpg").exists()

    train_row = json.loads((out / "train.jsonl").read_text(encoding="utf-8").strip())
    assert train_row["label"] == 0
    assert train_row["image_path"] == "images/real/a.jpg"
    assert (out / train_row["image_path"]).read_bytes() == b"a"

    val_row = json.loads((out / "val.jsonl").read_text(encoding="utf-8").strip())
    assert val_row["label"] == 1
    assert val_row["split"] == "val"
    assert val_row["image_path"] == "images/fake/b.jpg"
    assert (out / val_row["image_path"]).read_bytes() == b"b"

    test_row = json.loads((out / "test.jsonl").read_text(encoding="utf-8").strip())
    assert test_row["metadata"]["image_match_count"] == 2
    assert test_row["metadata"]["image_first_match_position"] == 1
    assert test_row["metadata"]["image_unselected_url_count"] == 1
    assert test_row["metadata"]["image_unselected_urls"] == ["https://x/a.jpg"]
    assert test_row["metadata"]["event_id"] == 2

    manifest_text = (out / "manifest.json").read_text(encoding="utf-8")
    for raw_token in ("r1", "r2", "r3", "真实新闻一", "hello"):
        assert raw_token not in manifest_text


def test_build_rejects_official_record_without_image(tmp_path: Path) -> None:
    records = [
        {"id": "r1", "label": 0, "body": "正文", "image_line": "https://x/missing.jpg"},
    ]
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=records,
        splits={"train": {"r1": 0}, "validate": {}, "test": {}},
        images=_three_images(),
    )
    with pytest.raises(Weibo17ArchiveError, match="no matching archive image"):
        build_weibo17_dataset(archive_path, tmp_path / "out", verify_integrity=False)


def test_build_rejects_duplicate_text_id(tmp_path: Path) -> None:
    records = [
        {"id": "dup", "label": 0, "body": "一", "image_line": "https://x/a.jpg"},
        {"id": "dup", "label": 1, "body": "二", "image_line": "https://x/b.jpg"},
    ]
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=records,
        splits={"train": {"dup": 0}, "validate": {}, "test": {}},
        images=_three_images(),
    )
    with pytest.raises(Weibo17ArchiveError, match="duplicate sample id"):
        build_weibo17_dataset(archive_path, tmp_path / "out", verify_integrity=False)


def test_build_rejects_split_overlap(tmp_path: Path) -> None:
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=_three_records(),
        splits={"train": {"r1": 0}, "validate": {"r1": 1}, "test": {"r3": 2}},
        images=_three_images(),
    )
    with pytest.raises(Weibo17ArchiveError, match="overlap"):
        build_weibo17_dataset(archive_path, tmp_path / "out", verify_integrity=False)


def test_build_rejects_non_triple_line_member(tmp_path: Path) -> None:
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=_three_records(),
        splits={"train": {"r1": 0}, "validate": {"r2": 1}, "test": {"r3": 2}},
        images=_three_images(),
    )
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(broken, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "tweets/train_nonrumor.txt":
                payload = payload + b"orphan line\n"
            target.writestr(info, payload)
    with pytest.raises(Weibo17ArchiveError, match="multiple of three"):
        build_weibo17_dataset(broken, tmp_path / "out", verify_integrity=False)


def test_assert_official_invariants_rejects_synthetic_manifest(tmp_path: Path) -> None:
    archive_path = _build_fixture(
        tmp_path / "weibo.zip",
        records=_three_records(),
        splits={"train": {"r1": 0}, "validate": {"r2": 1}, "test": {"r3": 2}},
        images=_three_images(),
    )
    manifest = build_weibo17_dataset(archive_path, tmp_path / "out", verify_integrity=False)
    with pytest.raises(Weibo17ArchiveError, match="invariants failed"):
        assert_official_invariants(manifest)
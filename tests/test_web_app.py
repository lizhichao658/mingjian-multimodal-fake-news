from __future__ import annotations

import base64
import http.client
import io
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mingjian.data.tokenization import CharTokenizer
from mingjian.inference import LitePredictor
from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig
from mingjian.web import MingJianWebApp, create_server
from mingjian.web.app import MAX_IMAGE_BYTES, _handler_class


def _predictor(tmp_path) -> LitePredictor:
    config = LiteModelConfig(
        vocab_size=12,
        text_embed_dim=8,
        text_num_filters=4,
        text_kernel_sizes=(2, 3),
        image_widths=(8, 16),
        image_feature_dim=16,
        hidden_dim=12,
        dropout=0.0,
    )
    model = LiteFusionClassifier(config)
    tokenizer = CharTokenizer(("<pad>", "<unk>", "a", "b", "c", "d"), max_length=8)
    tokenizer.save(tmp_path / "tokenizer.json")
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "threshold": 0.5,
            "tokenizer_file": "tokenizer.json",
        },
        checkpoint,
    )
    return LitePredictor.from_checkpoint(checkpoint, device="cpu", image_size=16)


def _image_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), color=(80, 120, 160)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _stub_result(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "decision": {"label": "fake", "probability_fake": 0.8, "threshold": 0.5},
        "risk": {"level": "high"},
        "evidence_card": {
            "state": "insufficient",
            "state_zh": "证据不足",
            "cross_modal": {"reason_code": "missing_image"},
            "recommended_action": "人工复核",
        },
    }


class _RecordingPredictor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.image_paths: list[Path | None] = []
        self.image_sizes: list[int] = []

    def analyze(self, text, *, image_path=None, sample_id=""):
        self.image_paths.append(image_path)
        if image_path is not None:
            assert image_path.is_file()
            self.image_sizes.append(image_path.stat().st_size)
        if self.fail:
            raise RuntimeError("predictor failed")
        return _stub_result(sample_id)


@contextmanager
def _running_server(predictor):
    server = create_server(predictor, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _analysis_request(base_url: str, payload) -> urllib.request.Request:
    return urllib.request.Request(
        f"{base_url}/api/analyze",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def test_web_app_handles_text_only_and_image_payloads(tmp_path) -> None:
    app = MingJianWebApp(_predictor(tmp_path))
    text_only = app.analyze_payload({"text": "abcd", "sample_id": "text-only"})
    with_image = app.analyze_payload(
        {"text": "abcd", "sample_id": "with-image", "image_base64": _image_data_url()}
    )

    assert text_only["modalities"]["has_image"] is False
    assert text_only["evidence_card"]["state"] == "insufficient"
    assert with_image["modalities"]["has_image"] is True
    assert with_image["evidence_card"]["state"] in {"support", "conflict", "insufficient"}
    assert len(app.history()) == 2
    history_record = app.history()[0]
    assert history_record["sample_id"] == "with-image"
    assert history_record["label"] == with_image["decision"]["label"]
    assert history_record["display_label"] == with_image["decision"]["display_label"]
    assert history_record["display_label_name_zh"] == with_image["decision"]["display_label_name_zh"]
    assert history_record["probability_fake"] == with_image["decision"]["probability_fake"]
    assert history_record["risk_level"] == with_image["risk"]["level"]
    assert history_record["evidence_state"] in {"support", "conflict", "insufficient"}
    assert history_record["evidence_state_zh"]
    assert history_record["evidence_reason"]
    assert history_record["recommended_action"]


def test_web_app_rejects_invalid_payload(tmp_path) -> None:
    app = MingJianWebApp(_predictor(tmp_path))
    with pytest.raises(ValueError, match="text"):
        app.analyze_payload({"text": "  "})
    with pytest.raises(ValueError, match="base64"):
        app.analyze_payload({"text": "abcd", "image_base64": "not-base64"})


@pytest.mark.parametrize("image_value", [False, 0, [], {}])
def test_web_app_rejects_non_string_images_even_when_falsy(image_value) -> None:
    predictor = _RecordingPredictor()
    app = MingJianWebApp(predictor)

    with pytest.raises(TypeError, match="image_base64 must be a string"):
        app.analyze_payload({"text": "新闻正文", "image_base64": image_value})

    assert predictor.image_paths == []
    assert app.history() == []


@pytest.mark.parametrize(
    "image_bytes",
    [
        b"not an image",
        base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"),
        b"\x89PNG\r\n\x1a\n",
    ],
)
def test_http_server_rejects_invalid_image_files_as_client_errors(image_bytes) -> None:
    predictor = _RecordingPredictor()
    payload = {
        "text": "新闻正文",
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
    }
    with _running_server(predictor) as (server, base_url):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(_analysis_request(base_url, payload), timeout=5)
        assert caught.value.code == 400
        error = json.loads(caught.value.read().decode("utf-8"))
        assert "valid PNG, JPG or WEBP" in error["error"]
        assert predictor.image_paths == []
        assert server.app.history() == []
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=5) as response:
            assert json.loads(response.read()) == {
                "status": "ok", "project": "MingJian", "busy": False,
            }


def test_http_server_accepts_eight_megabyte_image_and_rejects_one_byte_over() -> None:
    predictor = _RecordingPredictor()
    png = base64.b64decode(_image_data_url().split(",", 1)[1])
    # Valid PNG with trailing padding exercises the upload envelope without
    # allocating a large decoded image or running a model on it.
    image_bytes = png + bytes(MAX_IMAGE_BYTES - len(png))
    payload = {
        "text": "新" * 5000,
        "sample_id": "maximum-image",
        "image_base64": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
    }
    with _running_server(predictor) as (server, base_url):
        with urllib.request.urlopen(_analysis_request(base_url, payload), timeout=10) as response:
            assert json.loads(response.read())["sample_id"] == "maximum-image"
        assert predictor.image_sizes == [MAX_IMAGE_BYTES]
        assert len(server.app.history()) == 1

        payload["image_base64"] = base64.b64encode(image_bytes + b"x").decode("ascii")
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(_analysis_request(base_url, payload), timeout=10)
        assert caught.value.code == 400
        assert "8 MB" in json.loads(caught.value.read())["error"]
        assert predictor.image_sizes == [MAX_IMAGE_BYTES]
        assert len(server.app.history()) == 1
    assert all(path is not None and not path.exists() for path in predictor.image_paths)


@pytest.mark.parametrize("fail", [False, True])
def test_web_app_removes_temporary_image_on_success_and_predictor_failure(fail) -> None:
    predictor = _RecordingPredictor(fail=fail)
    app = MingJianWebApp(predictor)
    payload = {"text": "新闻正文", "image_base64": _image_data_url()}
    if fail:
        with pytest.raises(RuntimeError, match="predictor failed"):
            app.analyze_payload(payload)
    else:
        app.analyze_payload(payload)

    assert len(predictor.image_paths) == 1
    assert not predictor.image_paths[0].exists()
    assert len(app.history()) == (0 if fail else 1)
    assert app.health()["busy"] is False
    # A failed inference must release the same lock for the next analysis.
    predictor.fail = False
    assert app.analyze_payload({"text": "继续研判", "sample_id": "retry"})["sample_id"] == "retry"


def test_web_app_serializes_model_explanations_without_blocking_health_or_history() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()

    class BlockingPredictor:
        def analyze(self, text, *, image_path=None, sample_id=""):
            if sample_id == "first":
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise RuntimeError("test did not release the first inference")
            else:
                second_entered.set()
            return _stub_result(sample_id)

    with _running_server(BlockingPredictor()) as (server, base_url):
        app = server.app

        def run_second():
            second_started.set()
            return app.analyze_payload({"text": "第二条新闻", "sample_id": "second"})

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(app.analyze_payload, {"text": "第一条新闻", "sample_id": "first"})
            assert first_entered.wait(timeout=5)
            second = executor.submit(run_second)
            try:
                assert second_started.wait(timeout=5)
                assert not second_entered.wait(timeout=0.15)
                with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as response:
                    assert json.loads(response.read())["busy"] is True
                with urllib.request.urlopen(f"{base_url}/api/history", timeout=2) as response:
                    assert json.loads(response.read())["items"] == []
            finally:
                release_first.set()
            assert first.result(timeout=5)["sample_id"] == "first"
            assert second.result(timeout=5)["sample_id"] == "second"
        assert second_entered.is_set()
        assert {record["sample_id"] for record in app.history()} == {"first", "second"}
        assert app.health()["busy"] is False


@pytest.mark.parametrize("connection_error", [BrokenPipeError, ConnectionResetError])
@pytest.mark.parametrize("phase", ["headers", "body"])
def test_http_response_tolerates_a_client_cancelling_its_wait(connection_error, phase) -> None:
    handler = object.__new__(_handler_class())
    handler.headers = {}
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.close_connection = False

    def disconnected(*args):
        raise connection_error("browser closed its request")

    handler.end_headers = disconnected if phase == "headers" else lambda: None
    handler.wfile = type("DisconnectedWriter", (), {"write": disconnected})()
    handler._send(b"{}", "application/json")

    assert handler.close_connection is True


def test_http_server_exposes_health_and_analyze_api(tmp_path) -> None:
    server = create_server(_predictor(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["status"] == "ok"
        request = urllib.request.Request(
            f"{base_url}/api/analyze",
            data=json.dumps({"text": "abcd", "sample_id": "http"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["sample_id"] == "http"
        assert "text_evidence" in result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_server_serves_polished_workspace_html(tmp_path) -> None:
    server = create_server(_predictor(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base_url}/", timeout=5) as response:
            html = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    from html.parser import HTMLParser
    from importlib.resources import files

    class PageParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = {}
            self.ids = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
                self.elements[attrs["id"]] = (tag, attrs)

    assert response.headers["Content-Type"].startswith("text/html")
    assert "no-store" in response.headers.get("Cache-Control", "").lower()
    assert html == files("mingjian.web").joinpath("workbench.html").read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids)), "UI IDs must be unambiguous"
    assert parser.elements["news-text"][0] == "textarea"
    assert parser.elements["image-input"][1]["type"] == "file"
    assert parser.elements["heat-layer"][0] == "canvas"
    assert parser.elements["technical-details"][0] == "details"
    assert "open" not in parser.elements["technical-details"][1]
    assert "hidden" in parser.elements["json-wrap"][1]
    assert "hidden" in parser.elements["media-frame"][1]
    assert 'id="result-score">—</span>' in html
    assert 'id="result-status">待分析</span>' in html
    assert "非本次模型" in html or "非实时研判" in html


def test_http_server_rejects_duplicate_listener() -> None:
    first = create_server(_RecordingPredictor(), port=0)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError):
            duplicate = create_server(_RecordingPredictor(), port=port)
            duplicate.server_close()
    finally:
        first.server_close()


def test_web_launcher_explains_a_busy_port(monkeypatch, capsys) -> None:
    from scripts import run_web_demo as launcher

    monkeypatch.setattr(launcher.LitePredictor, "from_checkpoint", lambda *a, **kw: object())

    def busy_port(*args, **kwargs):
        raise OSError(98, "address already in use")

    monkeypatch.setattr(launcher, "create_server", busy_port)
    assert launcher.main(["--checkpoint", "unused.pt", "--port", "8000"]) == 2
    error = capsys.readouterr().err
    assert "8000" in error and "--port" in error


@pytest.mark.parametrize("hostname", ["127.0.0.1", "localhost"])
def test_http_server_allows_the_actual_local_workbench_origin(hostname) -> None:
    predictor = _RecordingPredictor()
    with _running_server(predictor) as (server, _):
        base_url = f"http://{hostname}:{server.server_address[1]}"
        for path in ("/api/health", "/api/history"):
            request = urllib.request.Request(base_url + path, headers={"Origin": base_url})
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
                assert response.headers["Access-Control-Allow-Origin"] == base_url

        request = _analysis_request(base_url, {"text": "同源研判", "sample_id": "same-origin"})
        request.add_header("Origin", base_url)
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["sample_id"] == "same-origin"
        assert len(predictor.image_paths) == 1

        request = urllib.request.Request(
            f"{base_url}/api/analyze",
            headers={
                "Origin": base_url,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
            method="OPTIONS",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == base_url
            assert "POST" in response.headers["Access-Control-Allow-Methods"]
            assert "Content-Type" in response.headers["Access-Control-Allow-Headers"]
            assert "Access-Control-Allow-Private-Network" not in response.headers


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "https://attacker.invalid",
        "http://127.0.0.1:1",
        "http://localhost:{port}",
        "https://127.0.0.1:{port}",
        "http://user@127.0.0.1:{port}",
        "http://127.0.0.1:{port}/not-an-origin",
        "http://[broken",
    ],
)
def test_http_server_rejects_untrusted_origins_before_reading_or_analyzing(origin) -> None:
    predictor = _RecordingPredictor()
    with _running_server(predictor) as (server, base_url):
        origin = origin.format(port=server.server_address[1])
        for method, path in (
            ("GET", "/api/health"),
            ("GET", "/api/history"),
            ("POST", "/api/analyze"),
            ("OPTIONS", "/api/analyze"),
        ):
            # text/plain is a simple browser POST: denying CORS alone would
            # still let it invoke the model, so the handler itself must reject.
            request = urllib.request.Request(
                base_url + path,
                data=b'{"text":"blocked"}' if method == "POST" else None,
                headers={"Origin": origin, "Content-Type": "text/plain"},
                method=method,
            )
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            assert caught.value.code == 403
            assert json.loads(caught.value.read()) == {"error": "origin not allowed"}
            assert "Access-Control-Allow-Origin" not in caught.value.headers
        assert predictor.image_paths == []
        assert server.app.history() == []


@pytest.mark.parametrize(
    "host",
    [
        "attacker.invalid:{port}",
        "localhost.attacker.invalid:{port}",
        "127.0.0.1:{wrong_port}",
        "localhost:invalid",
        "user@127.0.0.1:{port}",
        "127.0.0.1:{port}/path",
    ],
)
def test_http_server_rejects_hosts_not_derived_from_the_listener(host) -> None:
    predictor = _RecordingPredictor()
    with _running_server(predictor) as (server, base_url):
        port = server.server_address[1]
        host = host.format(port=port, wrong_port=port + 1)
        for method, path in (
            ("GET", "/"),
            ("GET", "/api/history"),
            ("POST", "/api/analyze"),
            ("OPTIONS", "/api/analyze"),
        ):
            request = urllib.request.Request(
                base_url + path,
                data=b'{"text":"blocked"}' if method == "POST" else None,
                headers={"Host": host, "Origin": "http://" + host},
                method=method,
            )
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            assert caught.value.code == 403
            assert json.loads(caught.value.read()) == {"error": "host not allowed"}
            assert "Access-Control-Allow-Origin" not in caught.value.headers
        assert predictor.image_paths == []
        assert server.app.history() == []


@pytest.mark.parametrize("header_case", ["duplicate_host", "duplicate_origin", "missing_host"])
def test_http_server_rejects_ambiguous_authority_headers(header_case) -> None:
    with _running_server(_RecordingPredictor()) as (server, base_url):
        authority = f"127.0.0.1:{server.server_address[1]}"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            connection.putrequest("GET", "/api/history", skip_host=True)
            if header_case != "missing_host":
                connection.putheader("Host", authority)
            if header_case == "duplicate_host":
                connection.putheader("Host", "attacker.invalid")
            if header_case == "duplicate_origin":
                connection.putheader("Origin", base_url)
                connection.putheader("Origin", "https://attacker.invalid")
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 403
            assert response.getheader("Access-Control-Allow-Origin") is None
            assert json.loads(response.read())["error"] in {"host not allowed", "origin not allowed"}
        finally:
            connection.close()

from __future__ import annotations

import base64
import io
import json
import threading
import urllib.request

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mingjian.data.tokenization import CharTokenizer
from mingjian.inference import LitePredictor
from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig
from mingjian.web import MingJianWebApp, create_server


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


def test_web_app_handles_text_only_and_image_payloads(tmp_path) -> None:
    app = MingJianWebApp(_predictor(tmp_path))
    text_only = app.analyze_payload({"text": "abcd", "sample_id": "text-only"})
    with_image = app.analyze_payload(
        {"text": "abcd", "sample_id": "with-image", "image_base64": _image_data_url()}
    )

    assert text_only["modalities"]["has_image"] is False
    assert with_image["modalities"]["has_image"] is True
    assert len(app.history()) == 2
    assert app.history()[0]["sample_id"] == "with-image"


def test_web_app_rejects_invalid_payload(tmp_path) -> None:
    app = MingJianWebApp(_predictor(tmp_path))
    with pytest.raises(ValueError, match="text"):
        app.analyze_payload({"text": "  "})
    with pytest.raises(ValueError, match="base64"):
        app.analyze_payload({"text": "abcd", "image_base64": "not-base64"})


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

    assert response.headers["Content-Type"].startswith("text/html")
    assert 'data-ui="workspace"' in html
    assert 'data-ui="verdict"' in html
    assert 'data-ui="dropzone"' in html

"""A no-API-key local demo workspace for MingJian.

The competition environment may not have FastAPI/uvicorn installed, so the MVP
uses Python's standard library HTTP server. The API shape stays simple enough
to replace the transport with FastAPI later without changing the model layer.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import socket
import tempfile
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from mingjian.inference import LitePredictor

MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# The HTTP workspace and the standalone deliverable use this single UI source.
INDEX_HTML = Path(__file__).with_name("workbench.html").read_text(encoding="utf-8")


class MingJianWebApp:
    """In-memory application state shared by the local HTTP handlers."""

    def __init__(self, predictor: LitePredictor, *, history_size: int = 50) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.predictor = predictor
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        # Gradient explanations install hooks on the shared model and run backward.
        # Keep the entire call isolated while allowing history and health reads.
        self._inference_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "project": "MingJian",
            "busy": self._inference_lock.locked(),
        }

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def analyze_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > 5000:
            raise ValueError("text is too long for the MVP limit (5000 characters)")

        image_path: Path | None = None
        image_base64 = payload.get("image_base64")
        if image_base64 is not None and not isinstance(image_base64, str):
            raise TypeError("image_base64 must be a string")
        try:
            if image_base64:
                encoded = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("image_base64 is not valid base64") from exc
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("image is too large for the MVP limit (8 MB)")
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        if image.format not in {"PNG", "JPEG", "WEBP"}:
                            raise ValueError("unsupported image format")
                        image.verify()
                    # verify() checks file structure; load() also catches truncated
                    # JPEG/WebP pixel data before the shared predictor is entered.
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
                    raise ValueError("image must be a valid PNG, JPG or WEBP file") from exc
                with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as handle:
                    image_path = Path(handle.name)
                    handle.write(image_bytes)
            sample_id = str(payload.get("sample_id") or f"web-{int(time.time())}")
            with self._inference_lock:
                result = self.predictor.analyze(text, image_path=image_path, sample_id=sample_id)
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)

        record = {
            "sample_id": sample_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
            "label": result["decision"]["label"],
            "display_label": result["decision"].get("display_label", result["decision"]["label"]),
            "display_label_name_zh": result["decision"].get(
                "display_label_name_zh",
                result["decision"].get("label_name_zh", result["decision"]["label"]),
            ),
            "probability_fake": result["decision"]["probability_fake"],
            "threshold": result["decision"]["threshold"],
            "risk_level": result["risk"]["level"],
            "evidence_state": result["evidence_card"]["state"],
            "evidence_state_zh": result["evidence_card"]["state_zh"],
            "evidence_reason": result["evidence_card"]["cross_modal"].get("reason_code", ""),
            "recommended_action": result["evidence_card"]["recommended_action"],
        }
        with self._lock:
            self._history.appendleft(record)
        result["history_record"] = record
        return result


class MingJianHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying the application state with one listener per address."""

    # Windows SO_REUSEADDR permits multiple servers to bind the same port and
    # return different UI versions. Prefer an exclusive listener there.
    allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        app: MingJianWebApp,
    ) -> None:
        self.app = app
        super().__init__(server_address, handler_class)


def _handler_class() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MingJianMVP/0.1"

        @property
        def app(self) -> MingJianWebApp:
            return self.server.app  # type: ignore[attr-defined]

        def _allowed_origin(self) -> str:
            origin = self.headers.get("Origin", "")
            if not origin or origin == "null":
                return ""
            authority = self._trusted_authority()
            try:
                parsed = urlparse(origin)
                if (
                    authority is not None
                    and parsed.scheme == "http"
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.path
                    and not parsed.query
                    and not parsed.fragment
                    and (parsed.hostname, 80 if parsed.port is None else parsed.port) == authority
                    and len(self.headers.get_all("Origin", [])) == 1
                ):
                    return origin
            except ValueError:
                pass
            return ""

        def _trusted_authority(self) -> tuple[str, int] | None:
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1:
                return None
            try:
                parsed = urlparse("http://" + hosts[0])
                if (
                    parsed.username is not None
                    or parsed.password is not None
                    or parsed.path
                    or parsed.query
                    or parsed.fragment
                ):
                    return None
                bound_host, bound_port = self.server.server_address[:2]
                expected_hosts = {bound_host, self.connection.getsockname()[0]}
                if ip_address(bound_host).is_loopback or ip_address(bound_host).is_unspecified:
                    expected_hosts.update({"localhost", "127.0.0.1", "::1"})
                port = 80 if parsed.port is None else parsed.port
                if parsed.hostname in expected_hosts and port == bound_port:
                    return parsed.hostname, bound_port
            except ValueError:
                pass
            return None

        def _allow_request(self) -> bool:
            # Host must come from the listener, not an arbitrary DNS name that
            # resolves to loopback. An opaque Origin is not evidence of file://.
            if self._trusted_authority() is None:
                self.close_connection = True
                self._json({"error": "host not allowed"}, HTTPStatus.FORBIDDEN)
                return False
            if self.headers.get_all("Origin", []) and not self._allowed_origin():
                self.close_connection = True
                self._json({"error": "origin not allowed"}, HTTPStatus.FORBIDDEN)
                return False
            return True

        def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                origin = self._allowed_origin()
                if origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Cancelling browser fetch only ends the client wait. A completed
                # inference can still attempt to reply after that socket closes.
                self.close_connection = True

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def do_OPTIONS(self) -> None:
            if not self._allow_request():
                return
            if urlparse(self.path).path not in {"/api/analyze", "/api/history", "/api/health"}:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            if not self._allowed_origin():
                self._json({"error": "origin not allowed"}, HTTPStatus.FORBIDDEN)
                return
            self._send(b"", "text/plain; charset=utf-8", HTTPStatus.NO_CONTENT)

        def do_GET(self) -> None:
            if not self._allow_request():
                return
            path = urlparse(self.path).path
            if path == "/":
                self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(self.app.health())
            elif path == "/api/history":
                self._json({"items": self.app.history()})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if not self._allow_request():
                return
            if urlparse(self.path).path != "/api/analyze":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json({"error": "invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json({"error": "invalid request size"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = self.app.analyze_payload(payload)
            except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # noqa: BLE001 - keep the local demo server alive
                self._json({"error": f"internal error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self._json(result)

    return Handler


def create_server(
    predictor: LitePredictor,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    history_size: int = 50,
) -> MingJianHTTPServer:
    """Create a local server; the caller can serve_forever or use port 0 in tests."""

    app = MingJianWebApp(predictor, history_size=history_size)
    return MingJianHTTPServer((host, port), _handler_class(), app)

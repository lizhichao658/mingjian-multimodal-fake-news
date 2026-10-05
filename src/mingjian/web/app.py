"""A no-API-key local demo workspace for MingJian.

The competition environment may not have FastAPI/uvicorn installed, so the MVP
uses Python's standard library HTTP server. The API shape stays simple enough
to replace the transport with FastAPI later without changing the model layer.
"""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mingjian.inference import LitePredictor

MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024

INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>明鉴 · 多模态虚假新闻辅助研判</title>
  <style>
    :root { color-scheme: light; --ink:#172033; --muted:#65708a; --line:#e4e8f0;
      --brand:#3157d5; --danger:#c73535; --safe:#16805d; --warn:#ad6a00; --bg:#f5f7fb; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; background:var(--bg); color:var(--ink); }
    header { background:linear-gradient(120deg,#16234f,#3157d5); color:white; padding:28px 7vw 24px; }
    header h1 { margin:0 0 6px; font-size:28px; letter-spacing:.04em; }
    header p { margin:0; color:#dce4ff; }
    main { max-width:1180px; margin:24px auto 60px; padding:0 18px; display:grid;
      grid-template-columns:minmax(320px,1fr) minmax(360px,1.25fr); gap:20px; }
    .card { background:white; border:1px solid var(--line); border-radius:16px; padding:20px;
      box-shadow:0 8px 30px rgba(24,39,75,.06); }
    .full { grid-column:1/-1; }
    label { display:block; font-weight:700; margin:14px 0 7px; }
    textarea, input[type=file] { width:100%; border:1px solid #ccd4e3; border-radius:10px; padding:11px;
      font:inherit; background:#fbfcff; }
    textarea { min-height:170px; resize:vertical; }
    button { border:0; border-radius:10px; padding:11px 18px; font:inherit; font-weight:700;
      cursor:pointer; background:var(--brand); color:white; margin-top:14px; }
    button.secondary { background:#e9edfa; color:#253d9a; margin-left:8px; }
    button:disabled { opacity:.55; cursor:wait; }
    .hint { color:var(--muted); font-size:13px; line-height:1.6; }
    .result-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
    .badge { border-radius:999px; padding:6px 11px; font-size:13px; font-weight:700; }
    .badge.fake,.badge.high { background:#fde8e8; color:var(--danger); }
    .badge.real,.badge.low { background:#e4f6ef; color:var(--safe); }
    .badge.medium { background:#fff1d8; color:var(--warn); }
    .badge.uncertain { background:#e9edfa; color:#253d9a; }
    .prob { font-size:42px; font-weight:800; margin:8px 0 2px; }
    .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:14px; }
    .metric { border:1px solid var(--line); border-radius:11px; padding:11px; background:#fbfcff; }
    .metric b { display:block; font-size:18px; margin-top:3px; }
    .evidence { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .evidence span { border:1px solid #d9e0ef; background:#f4f7ff; border-radius:8px; padding:6px 8px; }
    .cam { width:min(100%,360px); margin-top:10px; display:grid; gap:2px;
      border:1px solid var(--line); padding:5px; background:#f7f9fd; border-radius:8px; }
    .cam div { aspect-ratio:1; border-radius:2px; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f7f9fd; border-radius:10px;
      padding:12px; font:12px/1.55 Consolas,monospace; max-height:220px; overflow:auto; }
    ul { padding-left:20px; line-height:1.75; }
    .history-item { border-top:1px solid var(--line); padding:11px 0; }
    .history-item:first-child { border-top:0; }
    .muted { color:var(--muted); font-size:13px; }
    @media (max-width:850px) { main { grid-template-columns:1fr; } .full { grid-column:auto; } }
  </style>
</head>
<body>
<header>
  <h1>明鉴 · 图文虚假新闻辅助研判</h1>
  <p>文本 + 单图 MVP · 解释证据 · 缺图降级 · 无 API Key 本地演示</p>
</header>
<main>
  <section class="card">
    <h2>输入</h2>
    <label for="text">新闻文本</label>
    <textarea id="text">某地发生突发事件，官方尚未发布通报，网传图片与现场情况存在明显矛盾。</textarea>
    <label for="image">配图（可选）</label>
    <input id="image" type="file" accept="image/*">
    <p class="hint">没有图片也可以运行，系统会执行缺图降级并把图像贡献置零。图片仅在本机内存/临时文件中处理。</p>
    <button id="run">开始研判</button>
    <button id="clear" class="secondary">清空结果</button>
    <p id="status" class="hint"></p>
  </section>
  <section class="card">
    <h2>研判结果</h2>
    <div id="empty" class="hint">提交一条文本后，这里会显示概率、风险、文本证据和图像热图。</div>
    <div id="result" hidden>
      <div class="result-head">
        <div><div id="label" class="badge"></div><div id="prob" class="prob"></div><div class="muted">P(fake) / 模型原始概率</div></div>
        <div id="risk" class="badge"></div>
      </div>
      <div class="grid">
        <div class="metric">文本贡献<b id="textContrib"></b></div>
        <div class="metric">图像贡献<b id="imageContrib"></b></div>
        <div class="metric">是否缺图<b id="hasImage"></b></div>
      </div>
      <h3>文本证据</h3><div id="textEvidence" class="evidence"></div>
      <h3>图像区域热图（Grad-CAM）</h3><div id="cam" class="cam"></div>
      <h3>限制说明</h3><ul id="limitations"></ul>
      <details><summary>查看结构化 JSON</summary><pre id="json"></pre></details>
    </div>
  </section>
  <section class="card full">
    <h2>本次会话历史</h2>
    <div id="history" class="hint">暂无记录。</div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const color = (value) => `rgba(215, 48, 48, ${Math.max(0.04, Number(value)).toFixed(3)})`;
const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

function render(data) {
  $('empty').hidden = true; $('result').hidden = false;
  $('label').textContent = `${data.decision.label_name_zh} · ${data.decision.label}`;
  $('label').className = `badge ${data.decision.label}`;
  $('prob').textContent = pct(data.decision.probability_fake);
  $('risk').textContent = `风险：${data.risk.level}`;
  $('risk').className = `badge ${data.risk.level}`;
  $('textContrib').textContent = pct(data.modalities.text_contribution);
  $('imageContrib').textContent = pct(data.modalities.image_contribution);
  $('hasImage').textContent = data.modalities.has_image ? '否' : '是（已降级）';
  $('textEvidence').innerHTML = data.text_evidence.map((item) =>
    `<span title="位置 ${item.position}，方向 ${item.direction}">${item.token} · ${pct(item.importance)}</span>`
  ).join('') || '<span>无有效文本证据</span>';
  const cam = data.image_evidence.cam || [];
  $('cam').style.gridTemplateColumns = `repeat(${cam[0]?.length || 1}, 1fr)`;
  $('cam').innerHTML = cam.flat().map((v) => `<div style="background:${color(v)}"></div>`).join('');
  $('limitations').innerHTML = data.limitations.map((x) => `<li>${x}</li>`).join('');
  $('json').textContent = JSON.stringify(data, null, 2);
}

function renderHistory(items) {
  if (!items.length) { $('history').textContent = '暂无记录。'; return; }
  $('history').innerHTML = items.map((item) => `<div class="history-item">
    <b>${item.label}</b> · ${pct(item.probability_fake)} · ${item.risk_level}
    <div class="muted">${item.sample_id} · ${item.created_at} · ${item.text_preview}</div>
  </div>`).join('');
}

async function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); reader.onload = () => resolve(reader.result);
    reader.onerror = reject; reader.readAsDataURL(file);
  });
}

$('run').onclick = async () => {
  const button = $('run'); button.disabled = true; $('status').textContent = '推理中...';
  try {
    const text = $('text').value.trim(); if (!text) throw new Error('请先输入新闻文本。');
    const file = $('image').files[0];
    const image_base64 = file ? await fileToDataURL(file) : null;
    const response = await fetch('/api/analyze', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, image_base64, sample_id:'web-demo'}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '请求失败');
    render(data); renderHistory((await (await fetch('/api/history')).json()).items);
    $('status').textContent = '完成。高风险结论请务必人工复核。';
  } catch (error) { $('status').textContent = `错误：${error.message}`; }
  finally { button.disabled = false; }
};
$('clear').onclick = () => { $('result').hidden = true; $('empty').hidden = false; $('status').textContent = ''; };
fetch('/api/history').then((r) => r.json()).then((x) => renderHistory(x.items));
</script>
</body>
</html>'''


class MingJianWebApp:
    """In-memory application state shared by the local HTTP handlers."""

    def __init__(self, predictor: LitePredictor, *, history_size: int = 50) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self.predictor = predictor
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._lock = threading.Lock()

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
        try:
            if image_base64:
                if not isinstance(image_base64, str):
                    raise TypeError("image_base64 must be a string")
                encoded = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("image_base64 is not valid base64") from exc
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("image is too large for the MVP limit (8 MB)")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as handle:
                    handle.write(image_bytes)
                    image_path = Path(handle.name)
            sample_id = str(payload.get("sample_id") or f"web-{int(time.time())}")
            result = self.predictor.analyze(text, image_path=image_path, sample_id=sample_id)
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)

        record = {
            "sample_id": sample_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
            "label": result["decision"]["label"],
            "probability_fake": result["decision"]["probability_fake"],
            "risk_level": result["risk"]["level"],
        }
        with self._lock:
            self._history.appendleft(record)
        result["history_record"] = record
        return result


class MingJianHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying the application state."""

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

        def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json({"status": "ok", "project": "MingJian"})
            elif path == "/api/history":
                self._json({"items": self.app.history()})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
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


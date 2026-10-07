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
  <meta name="theme-color" content="#10264f">
  <title>明鉴 · 多模态虚假新闻辅助研判</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #12213f;
      --ink-soft: #344568;
      --muted: #6f7d98;
      --line: #dfe5f0;
      --line-strong: #ccd6e6;
      --surface: #ffffff;
      --surface-soft: #f7f9fd;
      --canvas: #eef3fa;
      --brand: #2454d8;
      --brand-deep: #12306f;
      --cyan: #10a89b;
      --danger: #cb3c49;
      --danger-soft: #fff0f1;
      --safe: #12835f;
      --safe-soft: #eaf8f2;
      --warn: #ac6b09;
      --warn-soft: #fff7e6;
      --shadow-sm: 0 8px 24px rgba(22, 42, 84, .06);
      --shadow-lg: 0 18px 50px rgba(22, 42, 84, .1);
      --radius-lg: 22px;
      --radius-md: 14px;
      --radius-sm: 10px;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-width: 320px;
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 0%, rgba(36, 84, 216, .13), transparent 28rem),
        radial-gradient(circle at 92% 8%, rgba(16, 168, 155, .1), transparent 24rem),
        var(--canvas);
    }
    button, textarea, input { font: inherit; }
    button { border: 0; }
    .topbar {
      position: relative;
      z-index: 2;
      border-bottom: 1px solid rgba(255, 255, 255, .14);
      background: linear-gradient(118deg, #10264f 0%, #1c3d82 52%, #2454d8 100%);
      color: #fff;
      box-shadow: 0 12px 32px rgba(15, 39, 92, .16);
    }
    .topbar-inner {
      width: min(1280px, calc(100% - 40px));
      min-height: 92px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }
    .brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
    .brand-mark {
      width: 48px;
      height: 48px;
      flex: 0 0 48px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(255, 255, 255, .3);
      border-radius: 15px;
      background: linear-gradient(145deg, rgba(255, 255, 255, .24), rgba(255, 255, 255, .08));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, .28);
      font-size: 23px;
      font-weight: 900;
      letter-spacing: .08em;
    }
    .brand h1 { margin: 0; font-size: clamp(22px, 2.2vw, 29px); letter-spacing: .08em; }
    .brand p { margin: 5px 0 0; color: #d7e2ff; font-size: 13px; }
    .top-badges { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .top-badge {
      border: 1px solid rgba(255, 255, 255, .2);
      border-radius: 999px;
      padding: 7px 11px;
      background: rgba(255, 255, 255, .1);
      color: #eef4ff;
      font-size: 12px;
      white-space: nowrap;
    }
    .top-badge.live::before {
      content: "";
      display: inline-block;
      width: 7px;
      height: 7px;
      margin-right: 7px;
      border-radius: 50%;
      background: #61e5c9;
      box-shadow: 0 0 0 4px rgba(97, 229, 201, .12);
      vertical-align: 1px;
    }
    .workspace {
      width: min(1280px, calc(100% - 40px));
      margin: 28px auto 44px;
      display: grid;
      grid-template-columns: minmax(360px, .92fr) minmax(460px, 1.18fr);
      gap: 22px;
      align-items: start;
    }
    .panel {
      border: 1px solid rgba(204, 214, 230, .9);
      border-radius: var(--radius-lg);
      background: rgba(255, 255, 255, .94);
      box-shadow: var(--shadow-sm);
      backdrop-filter: blur(12px);
    }
    .input-panel, .result-panel { padding: 25px; }
    .result-panel { min-height: 620px; }
    .panel-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 21px;
    }
    .eyebrow {
      margin-bottom: 7px;
      color: var(--brand);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .14em;
    }
    .panel h2 { margin: 0; font-size: 21px; letter-spacing: .02em; }
    .panel-heading p { margin: 7px 0 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
    .step {
      min-width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border-radius: 11px;
      background: #edf2ff;
      color: var(--brand);
      font-size: 13px;
      font-weight: 900;
    }
    .field-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 8px;
      color: var(--ink-soft);
      font-size: 14px;
      font-weight: 800;
    }
    .optional { color: var(--muted); font-size: 12px; font-weight: 500; }
    .textarea-wrap { position: relative; }
    textarea {
      width: 100%;
      min-height: 190px;
      resize: vertical;
      border: 1px solid var(--line-strong);
      border-radius: var(--radius-md);
      outline: none;
      padding: 14px 15px 34px;
      background: linear-gradient(180deg, #fff, #fbfcff);
      color: var(--ink);
      line-height: 1.75;
      transition: border-color .18s ease, box-shadow .18s ease;
    }
    textarea:focus {
      border-color: rgba(36, 84, 216, .72);
      box-shadow: 0 0 0 4px rgba(36, 84, 216, .1);
    }
    textarea::placeholder { color: #9aa7bd; }
    .char-count {
      position: absolute;
      right: 12px;
      bottom: 9px;
      color: var(--muted);
      font-size: 12px;
      pointer-events: none;
    }
    .dropzone {
      position: relative;
      display: grid;
      justify-items: center;
      gap: 7px;
      min-height: 112px;
      padding: 16px;
      border: 1px dashed #b9c8e3;
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #f9fbff, #f4f7fe);
      color: var(--muted);
      text-align: center;
      cursor: pointer;
      transition: border-color .18s ease, background .18s ease, transform .18s ease;
    }
    .dropzone:hover, .dropzone.dragging {
      border-color: var(--brand);
      background: #f3f6ff;
      transform: translateY(-1px);
    }
    .drop-icon {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      border-radius: 10px;
      background: #e9efff;
      color: var(--brand);
      font-size: 19px;
      font-weight: 800;
    }
    .dropzone strong { color: var(--ink-soft); font-size: 14px; }
    .dropzone small { font-size: 12px; }
    .drop-action { color: var(--brand); font-size: 13px; font-weight: 800; }
    .image-preview {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbfcff;
    }
    .image-preview img {
      width: 54px;
      height: 54px;
      flex: 0 0 54px;
      border-radius: 9px;
      object-fit: cover;
      background: #e9eef8;
    }
    .image-meta { min-width: 0; flex: 1; }
    .image-meta b { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
    .image-meta span { display: block; margin-top: 4px; color: var(--muted); font-size: 12px; }
    .icon-button {
      width: 30px;
      height: 30px;
      flex: 0 0 30px;
      display: grid;
      place-items: center;
      border-radius: 9px;
      background: #f0f3f9;
      color: var(--ink-soft);
      cursor: pointer;
    }
    .action-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .primary, .secondary, .ghost {
      border-radius: 12px;
      padding: 12px 18px;
      font-weight: 800;
      cursor: pointer;
      transition: transform .16s ease, box-shadow .16s ease, background .16s ease;
    }
    .primary {
      min-width: 142px;
      background: linear-gradient(135deg, #2454d8, #1641ae);
      color: #fff;
      box-shadow: 0 10px 22px rgba(36, 84, 216, .22);
    }
    .primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 13px 26px rgba(36, 84, 216, .28); }
    .secondary { background: #eef2fa; color: var(--ink-soft); }
    .secondary:hover { background: #e6ecf8; }
    .primary:disabled { cursor: wait; opacity: .68; }
    .status {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      min-height: 20px;
      margin: 13px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .status:empty { display: none; }
    .status::before { content: ""; width: 7px; height: 7px; flex: 0 0 7px; margin-top: 6px; border-radius: 50%; background: #a8b4c8; }
    .status.ok { color: var(--safe); }
    .status.ok::before { background: var(--safe); }
    .status.error { color: var(--danger); }
    .status.error::before { background: var(--danger); }
    .status.loading::before { background: var(--brand); animation: pulse 1.1s infinite; }
    @keyframes pulse { 50% { opacity: .25; transform: scale(.75); } }
    .empty-state {
      min-height: 470px;
      display: grid;
      place-items: center;
      align-content: center;
      gap: 13px;
      padding: 36px 20px;
      border: 1px dashed var(--line-strong);
      border-radius: 17px;
      background: linear-gradient(180deg, #fbfcff, #f6f8fd);
      text-align: center;
    }
    .empty-orb {
      width: 72px;
      height: 72px;
      display: grid;
      place-items: center;
      border-radius: 24px;
      background: linear-gradient(145deg, #e8efff, #dbe8ff);
      color: var(--brand);
      font-size: 30px;
      box-shadow: inset 0 1px 0 #fff;
    }
    .empty-state h3 { margin: 0; font-size: 18px; }
    .empty-state p { max-width: 360px; margin: 0; color: var(--muted); font-size: 13px; line-height: 1.7; }
    .verdict-card {
      position: relative;
      overflow: hidden;
      padding: 21px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #f9fbff;
    }
    .verdict-card::after {
      content: "";
      position: absolute;
      width: 180px;
      height: 180px;
      right: -78px;
      top: -86px;
      border-radius: 50%;
      background: rgba(36, 84, 216, .08);
    }
    .verdict-card.fake { border-color: #f2c8cd; background: linear-gradient(135deg, #fff7f8, #fff); }
    .verdict-card.fake::after { background: rgba(203, 60, 73, .1); }
    .verdict-card.real { border-color: #bfe9d8; background: linear-gradient(135deg, #f2fcf8, #fff); }
    .verdict-card.real::after { background: rgba(18, 131, 95, .1); }
    .verdict-card.uncertain { border-color: #f0d9a7; background: linear-gradient(135deg, #fffaf0, #fff); }
    .hero-top { position: relative; z-index: 1; display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
    .verdict-pill, .risk-badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .verdict-pill.fake, .risk-badge.high { background: var(--danger-soft); color: var(--danger); }
    .verdict-pill.real, .risk-badge.low { background: var(--safe-soft); color: var(--safe); }
    .risk-badge.medium { background: var(--warn-soft); color: var(--warn); }
    .risk-badge.uncertain { background: #eef2ff; color: var(--brand); }
    .verdict-copy { position: relative; z-index: 1; max-width: 560px; margin: 7px 0 0; color: var(--ink-soft); font-size: 13px; line-height: 1.65; }
    .prob-layout { position: relative; z-index: 1; display: grid; grid-template-columns: auto 1fr; gap: 20px; align-items: end; margin-top: 17px; }
    .prob { font-size: clamp(38px, 5vw, 54px); line-height: .95; font-weight: 900; letter-spacing: -.04em; }
    .prob-label { margin-top: 6px; color: var(--muted); font-size: 12px; }
    .prob-track { align-self: center; }
    .prob-track-head { display: flex; justify-content: space-between; margin-bottom: 7px; color: var(--muted); font-size: 12px; }
    .meter { height: 8px; overflow: hidden; border-radius: 999px; background: #e7ecf5; }
    .meter > span { display: block; width: 0; height: 100%; border-radius: inherit; transition: width .45s ease; }
    .prob-track .meter > span { background: linear-gradient(90deg, #2454d8, #10a89b); }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 17px; }
    .metric {
      min-width: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(255, 255, 255, .78);
    }
    .metric-label { display: block; color: var(--muted); font-size: 11px; }
    .metric-value { display: block; margin-top: 5px; color: var(--ink-soft); font-size: 16px; font-weight: 900; }
    .metric-meter { margin-top: 9px; }
    .metric-meter > span { background: linear-gradient(90deg, #5277e8, #7d9cf0); }
    .metric-meter.image > span { background: linear-gradient(90deg, #10a89b, #5cc8b8); }
    .section-block { margin-top: 22px; padding-top: 20px; border-top: 1px solid var(--line); }
    .section-title { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin: 0 0 10px; font-size: 16px; }
    .section-title span { color: var(--muted); font-size: 12px; font-weight: 500; }
    .evidence-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .evidence-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 100%;
      padding: 7px 10px;
      border: 1px solid #d8e1f2;
      border-radius: 9px;
      background: #f5f8ff;
      color: var(--ink-soft);
      font-size: 12px;
    }
    .evidence-chip b { color: var(--brand); font-weight: 800; }
    .evidence-empty { color: var(--muted); font-size: 13px; }
    .evidence-grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 18px; }
    .cam-frame {
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #f8fafe;
    }
    .cam {
      display: grid;
      gap: 3px;
      width: min(100%, 310px);
      margin: 0 auto;
    }
    .cam div { aspect-ratio: 1; border-radius: 3px; }
    .cam-caption { margin: 9px 0 0; color: var(--muted); font-size: 11px; line-height: 1.55; }
    .cam-empty {
      min-height: 150px;
      display: grid;
      place-items: center;
      padding: 15px;
      border: 1px dashed var(--line-strong);
      border-radius: 14px;
      background: #f8fafe;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
      text-align: center;
    }
    .limitations { margin: 0; padding-left: 20px; color: var(--ink-soft); font-size: 13px; line-height: 1.8; }
    .limitations li + li { margin-top: 5px; }
    details.json-details {
      margin-top: 21px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: #fbfcff;
    }
    details.json-details summary { cursor: pointer; padding: 13px 15px; color: var(--ink-soft); font-size: 13px; font-weight: 800; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow: auto; border-top: 1px solid var(--line); padding: 14px; color: #3c4e6d; font: 12px/1.65 Consolas, monospace; }
    .history-panel { grid-column: 1 / -1; padding: 22px 25px 24px; }
    .history-head { display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 15px; }
    .history-head h2 { margin: 0; font-size: 19px; }
    .history-count { border-radius: 999px; padding: 5px 9px; background: #eef3ff; color: var(--brand); font-size: 12px; font-weight: 800; }
    .history-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 11px; }
    .history-item { min-width: 0; padding: 14px; border: 1px solid var(--line); border-radius: 13px; background: #fbfcff; }
    .history-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .history-label { font-size: 13px; font-weight: 900; }
    .history-label.fake { color: var(--danger); }
    .history-label.real { color: var(--safe); }
    .history-prob { color: var(--ink-soft); font-size: 12px; font-weight: 800; }
    .history-preview { display: -webkit-box; overflow: hidden; margin: 9px 0 0; color: var(--ink-soft); font-size: 12px; line-height: 1.55; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
    .history-meta { margin-top: 10px; color: var(--muted); font-size: 11px; }
    .history-empty { padding: 20px; border: 1px dashed var(--line-strong); border-radius: 13px; background: #fbfcff; color: var(--muted); text-align: center; font-size: 13px; }
    .footnote { width: min(1280px, calc(100% - 40px)); margin: -16px auto 30px; color: var(--muted); text-align: center; font-size: 11px; line-height: 1.6; }
    @media (max-width: 1040px) {
      .workspace { grid-template-columns: 1fr; }
      .result-panel { min-height: auto; }
      .history-panel { grid-column: auto; }
    }
    @media (max-width: 680px) {
      .topbar-inner, .workspace, .footnote { width: min(100% - 24px, 1280px); }
      .topbar-inner { align-items: flex-start; flex-direction: column; padding: 17px 0; }
      .top-badges { justify-content: flex-start; }
      .input-panel, .result-panel, .history-panel { padding: 18px; }
      .metrics, .evidence-grid { grid-template-columns: 1fr; }
      .prob-layout { grid-template-columns: 1fr; gap: 13px; }
      .history-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">明</div>
      <div>
        <h1>明鉴</h1>
        <p>多模态虚假新闻辅助研判工作台</p>
      </div>
    </div>
    <div class="top-badges" aria-label="系统能力状态">
      <span class="top-badge live">本地运行</span>
      <span class="top-badge">无 API Key</span>
      <span class="top-badge">辅助研判</span>
    </div>
  </div>
</header>

<main class="workspace" data-ui="workspace">
  <section class="panel input-panel">
    <div class="panel-heading">
      <div>
        <div class="eyebrow">01 · 输入待研判内容</div>
        <h2>提交新闻文本与配图</h2>
        <p>模型将结合文本语义与图像特征，给出可解释的风险研判结果。</p>
      </div>
      <div class="step" aria-hidden="true">1</div>
    </div>

    <label class="field-label" for="text">新闻文本 <span class="optional">必填</span></label>
    <div class="textarea-wrap">
      <textarea id="text" maxlength="5000" placeholder="粘贴或输入需要研判的新闻文本……">某地发生突发事件，官方尚未发布通报，网传图片与现场情况存在明显矛盾。</textarea>
      <span id="textCount" class="char-count">0 / 5000</span>
    </div>

    <label class="field-label" for="image">新闻配图 <span class="optional">可选</span></label>
    <label class="dropzone" id="fileDrop" data-ui="dropzone" for="image">
      <input id="image" type="file" accept="image/png,image/jpeg,image/webp,image/gif" hidden>
      <span class="drop-icon" aria-hidden="true">＋</span>
      <strong>拖拽图片到此处，或点击选择</strong>
      <small>支持 PNG / JPG / WEBP，单张不超过 8 MB</small>
      <span class="drop-action">选择图片</span>
    </label>
    <div id="imagePreview" class="image-preview" hidden>
      <img id="previewImg" alt="待分析图片预览">
      <div class="image-meta">
        <b id="imageName">未选择文件</b>
        <span id="imageSize"></span>
      </div>
      <button id="removeImage" class="icon-button" type="button" aria-label="移除图片">×</button>
    </div>
    <p class="status" id="status" role="status"></p>

    <div class="action-row">
      <button class="primary" id="run" type="button">开始研判</button>
      <button class="secondary" id="clear" type="button">重置工作台</button>
    </div>
    <p class="status">无图片时将自动执行文本单模态降级，图像贡献置零。图片仅在本机临时处理。</p>
  </section>

  <section class="panel result-panel">
    <div class="panel-heading">
      <div>
        <div class="eyebrow">02 · 研判结果</div>
        <h2>结论与证据解释</h2>
        <p>高风险结论会保留证据链，但最终事实判断仍需人工复核。</p>
      </div>
      <div class="step" aria-hidden="true">2</div>
    </div>

    <div id="empty" class="empty-state">
      <div class="empty-orb" aria-hidden="true">⌕</div>
      <h3>等待一次研判</h3>
      <p>提交新闻文本后，这里会显示真假概率、风险等级、文本证据与图像热图。</p>
    </div>

    <div id="result" data-ui="verdict" hidden>
      <div id="verdict" class="verdict-card">
        <div class="hero-top">
          <div>
            <span id="label" class="verdict-pill"></span>
            <p id="verdictCopy" class="verdict-copy"></p>
          </div>
          <span id="risk" class="risk-badge"></span>
        </div>
        <div class="prob-layout">
          <div>
            <div id="prob" class="prob">0.0%</div>
            <div class="prob-label">P(fake) · 模型原始概率</div>
          </div>
          <div class="prob-track">
            <div class="prob-track-head"><span>假新闻概率</span><span id="probHint">等待研判</span></div>
            <div class="meter"><span id="probFill"></span></div>
          </div>
        </div>
      </div>

      <div class="metrics">
        <div class="metric">
          <span class="metric-label">文本贡献</span>
          <b id="textContrib" class="metric-value">--</b>
          <div class="meter metric-meter"><span id="textMeter"></span></div>
        </div>
        <div class="metric">
          <span class="metric-label">图像贡献</span>
          <b id="imageContrib" class="metric-value">--</b>
          <div class="meter metric-meter image"><span id="imageMeter"></span></div>
        </div>
        <div class="metric">
          <span class="metric-label">输入模态</span>
          <b id="hasImage" class="metric-value">--</b>
          <span id="modePill" class="metric-label">等待分析</span>
        </div>
      </div>

      <section class="section-block">
        <h3 class="section-title">文本证据 <span>梯度归因排序</span></h3>
        <div id="textEvidence" class="evidence-list"></div>
      </section>

      <section class="section-block">
        <h3 class="section-title">图像证据 <span>Grad-CAM 相对关注热图</span></h3>
        <div class="evidence-grid">
          <div>
            <div id="cam" class="cam"></div>
            <div id="camEmpty" class="cam-empty">本次未提供图片，图像证据已降级为空。</div>
            <p class="cam-caption">热图经过组内归一化，仅表示模型关注区域的相对强度，不构成因果证明。</p>
          </div>
          <div>
            <h4 class="section-title">模态说明</h4>
            <p class="verdict-copy" id="modalityNote">等待研判后显示文本与图像的门控贡献估计。</p>
            <details class="json-details">
              <summary>查看结构化 JSON</summary>
              <pre id="json">{}</pre>
            </details>
          </div>
        </div>
      </section>

      <section class="section-block">
        <h3 class="section-title">限制说明 <span>使用前请阅读</span></h3>
        <ul id="limitations" class="limitations"></ul>
      </section>
    </div>
  </section>

  <section class="panel history-panel">
    <div class="history-head">
      <div>
        <div class="eyebrow">03 · 可追溯记录</div>
        <h2>本次会话历史</h2>
      </div>
      <span id="historyCount" class="history-count">0 条</span>
    </div>
    <div id="history" class="history-empty">暂无记录。完成一次研判后，历史会显示在这里。</div>
  </section>
</main>

<footer class="footnote">明鉴 MVP · 结果仅供辅助研判，不构成事实认定、法律意见或行政处置依据。</footer>
<script>
const $ = (id) => document.getElementById(id);
const DEFAULT_TEXT = '某地发生突发事件，官方尚未发布通报，网传图片与现场情况存在明显矛盾。';
const riskLabels = { high: '高风险', medium: '中风险', low: '低风险', uncertain: '不确定' };
let previewUrl = null;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
}

function pct(value, digits = 1) {
  const number = Number(value);
  return `${(Number.isFinite(number) ? number * 100 : 0).toFixed(digits)}%`;
}

function setStatus(message, tone = '') {
  const status = $('status');
  status.textContent = message || '';
  status.className = `status${tone ? ` ${tone}` : ''}`;
}

function setBusy(busy) {
  const button = $('run');
  button.disabled = busy;
  button.textContent = busy ? '正在研判…' : '开始研判';
}

function updateCount() {
  $('textCount').textContent = `${$('text').value.length} / 5000`;
}

function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  $('previewImg').removeAttribute('src');
  $('imagePreview').hidden = true;
  $('imageName').textContent = '未选择文件';
  $('imageSize').textContent = '';
}

function setPreview(file) {
  if (!file) { clearPreview(); return; }
  if (!file.type || !file.type.startsWith('image/')) {
    clearPreview();
    $('image').value = '';
    setStatus('请选择 PNG、JPG、WEBP 等图片文件。', 'error');
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    clearPreview();
    $('image').value = '';
    setStatus('图片超过 8 MB，请压缩后再试。', 'error');
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('previewImg').src = previewUrl;
  $('imageName').textContent = file.name;
  $('imageSize').textContent = `${humanSize(file.size)} · 已在本机准备`;
  $('imagePreview').hidden = false;
  setStatus('图片已载入，将与文本共同参与融合研判。', 'ok');
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function render(data) {
  const decision = data.decision || {};
  const risk = data.risk || {};
  const modalities = data.modalities || {};
  const hasImage = Boolean(modalities.has_image);
  const probability = Number(decision.probability_fake || 0);
  const level = risk.level || 'uncertain';

  $('empty').hidden = true;
  $('result').hidden = false;
  $('verdict').className = `verdict-card ${decision.label || 'uncertain'}`;
  $('label').textContent = `${decision.label_name_zh || '待判断'} · ${decision.label === 'fake' ? 'fake' : 'real'}`;
  $('label').className = `verdict-pill ${decision.label || ''}`;
  $('verdictCopy').textContent = decision.label === 'fake'
    ? '模型检测到较高虚假风险，建议优先核验来源、时间、地点与图片上下文。'
    : '模型未发现显著虚假风险，但仍建议结合权威信源进行人工确认。';
  $('risk').textContent = riskLabels[level] || level;
  $('risk').className = `risk-badge ${level}`;
  $('prob').textContent = pct(probability);
  $('probFill').style.width = pct(probability);
  $('probHint').textContent = risk.reason || '等待研判说明';
  $('textContrib').textContent = pct(modalities.text_contribution);
  $('imageContrib').textContent = pct(modalities.image_contribution);
  $('textMeter').style.width = pct(modalities.text_contribution);
  $('imageMeter').style.width = pct(modalities.image_contribution);
  $('hasImage').textContent = hasImage ? '图文联合' : '文本单模态';
  $('modePill').textContent = hasImage ? '已提供配图' : '缺图降级';
  $('modalityNote').textContent = modalities.note || '门控贡献是相对幅度估计，不代表因果关系。';

  const evidence = Array.isArray(data.text_evidence) ? data.text_evidence : [];
  $('textEvidence').innerHTML = evidence.length
    ? evidence.map((item) => {
        const token = escapeHtml(item.token);
        const direction = escapeHtml(item.direction || '贡献');
        return `<span class="evidence-chip" title="位置 ${escapeHtml(item.position)}，方向 ${direction}">${token} <b>${pct(item.importance)}</b></span>`;
      }).join('')
    : '<span class="evidence-empty">无有效文本证据。</span>';

  const cam = Array.isArray(data.image_evidence?.cam) ? data.image_evidence.cam : [];
  const camHasShape = hasImage && cam.length && Array.isArray(cam[0]) && cam[0].length;
  const maxCam = camHasShape ? Math.max(0, ...cam.flat().map((value) => Number(value) || 0)) : 0;
  if (camHasShape && maxCam > 0) {
    $('cam').hidden = false;
    $('camEmpty').hidden = true;
    $('cam').style.gridTemplateColumns = `repeat(${cam[0].length}, 1fr)`;
    $('cam').innerHTML = cam.flat().map((value) => {
      const normalized = Math.max(0, Math.min(1, Number(value) / maxCam));
      const lightness = Math.round(94 - 54 * normalized);
      return `<div style="background:hsl(4 82% ${lightness}%)" title="相对强度 ${(normalized * 100).toFixed(1)}%"></div>`;
    }).join('');
  } else {
    $('cam').hidden = true;
    $('camEmpty').hidden = false;
    $('camEmpty').textContent = hasImage
      ? '本次图像未形成显著关注区域，热图信号过低。'
      : '本次未提供图片，图像证据已降级为空。';
  }

  const limitations = Array.isArray(data.limitations) ? data.limitations : [];
  $('limitations').innerHTML = limitations.length
    ? limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join('')
    : '<li>当前无额外限制说明。</li>';
  $('json').textContent = JSON.stringify(data, null, 2);
}

function renderHistory(items) {
  const list = Array.isArray(items) ? items : [];
  $('historyCount').textContent = `${list.length} 条`;
  if (!list.length) {
    $('history').className = 'history-empty';
    $('history').textContent = '暂无记录。完成一次研判后，历史会显示在这里。';
    return;
  }
  $('history').className = 'history-grid';
  $('history').innerHTML = list.map((item) => {
    const label = item.label === 'fake' ? '疑似虚假' : '倾向真实';
    const level = riskLabels[item.risk_level] || item.risk_level || '不确定';
    return `<article class="history-item">
      <div class="history-top">
        <span class="history-label ${escapeHtml(item.label)}">${escapeHtml(label)}</span>
        <span class="history-prob">${pct(item.probability_fake)}</span>
      </div>
      <p class="history-preview">${escapeHtml(item.text_preview || '无文本预览')}</p>
      <div class="history-meta">${escapeHtml(item.sample_id || 'sample')} · ${escapeHtml(item.created_at || '')} · ${escapeHtml(level)}</div>
    </article>`;
  }).join('');
}

function resetWorkspace() {
  $('text').value = '';
  $('image').value = '';
  clearPreview();
  updateCount();
  $('result').hidden = true;
  $('empty').hidden = false;
  setStatus('工作台已清空。', 'ok');
}

async function refreshHistory() {
  try {
    const response = await fetch('/api/history');
    if (!response.ok) return;
    renderHistory((await response.json()).items);
  } catch (_) {
    $('historyCount').textContent = '0 条';
  }
}

$('text').addEventListener('input', updateCount);
$('image').addEventListener('change', (event) => setPreview(event.target.files[0]));
$('removeImage').addEventListener('click', () => {
  $('image').value = '';
  clearPreview();
  setStatus('已移除图片，将按文本单模态研判。', 'ok');
});

const dropzone = $('fileDrop');
['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $('image').files = transfer.files;
  setPreview(file);
});

$('run').addEventListener('click', async () => {
  const text = $('text').value.trim();
  if (!text) {
    setStatus('请先输入新闻文本。', 'error');
    $('text').focus();
    return;
  }
  setBusy(true);
  setStatus('正在分析文本与图像特征，请稍候……', 'loading');
  try {
    const file = $('image').files[0];
    const imageBase64 = file ? await fileToDataURL(file) : null;
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, image_base64: imageBase64, sample_id: 'web-demo' })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '请求失败');
    render(data);
    await refreshHistory();
    setStatus('研判完成。高风险结论请务必结合原始信源人工复核。', 'ok');
  } catch (error) {
    setStatus(`研判失败：${error.message || '未知错误'}`, 'error');
  } finally {
    setBusy(false);
  }
});

$('clear').addEventListener('click', resetWorkspace);
$('text').value = DEFAULT_TEXT;
updateCount();
refreshHistory();
</script>
</body>
</html>
'''


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


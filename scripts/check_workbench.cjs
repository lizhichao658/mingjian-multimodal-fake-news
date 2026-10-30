const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(process.argv[2] || path.join(__dirname, '../src/mingjian/web/workbench.html'), 'utf8');
let code = source.match(/<script>([\s\S]*?)<\/script>/)[1];
code = code.replace(/\}\)\(\);\s*$/, 'globalThis.uiTest = { state, buildRecord, conclusionFor, resultPayload, hasCamSignal, rememberRecord, recallRecord, renderHistory, renderResult, setEmptyResult, cancelWaiting, loadExample, updateHeatView, paintHeatLayer, selectImage }; })();');
const calls = { drawImage: 0, copied: [], blobs: [], pending: [], resizeObservers: [], pendingImages: [] };
class Element {
  constructor(tag = 'div') {
    Object.assign(this, { tagName: tag, children: [], listeners: {}, attributes: {}, dataset: {}, style: {}, value: '', hidden: false, open: false, textContent: '', width: 0, height: 0 });
    this.classList = { add() {}, remove() {} };
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  focus() {} scrollIntoView() {} select() {} remove() {} click() {}
  getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {}, setTransform() {}, clearRect() {}, drawImage() { calls.drawImage++; } }; }
  async decode() {}
}
const elements = new Map([...source.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], new Element()]));
const $ = id => elements.get(id);
$('history-filter').value = 'all';
$('heat-opacity').value = '75';
Object.assign($('evidence-image'), { complete: true, naturalWidth: 1672, naturalHeight: 941, clientWidth: 500, clientHeight: 281 });
const context = { console, AbortController, DOMException, Blob, URLSearchParams,
  ResizeObserver: class { constructor(callback) { calls.resizeObservers.push(callback); } observe() {} },
  Image: class { set src(value) { this._src = value; calls.pendingImages.push(this); } },
  URL: { createObjectURL: blob => { calls.blobs.push(blob); return 'blob:test'; }, revokeObjectURL() {} },
  setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: async (url, options = {}) => {
    if (url.endsWith('/api/health')) return { ok: true, json: async () => ({ status: 'ok', project: 'MingJian' }) };
    return new Promise((resolve, reject) => {
      calls.pending.push({ resolve, reject, body: JSON.parse(options.body) });
      options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    });
  },
  document: { getElementById: id => elements.get(id), querySelectorAll: () => [], createElement: tag => new Element(tag), createTextNode: text => ({ textContent: text }), body: new Element() },
  navigator: { clipboard: { writeText: async text => calls.copied.push(text) } },
  window: { location: { protocol: 'http:', search: '' }, devicePixelRatio: 1, addEventListener() {}, isSecureContext: true }
};
vm.createContext(context);
vm.runInContext(code, context);
const flush = () => new Promise(resolve => setImmediate(resolve));
const documentView = () => context.document.body.dataset.view;
const visibleOverview = () => ['result-status', 'result-reason', 'score-context', 'review-advice', 'result-source'].map(id => $(id).textContent).join('\n');
(async () => {
  await flush();
  const test = context.uiTest;
  assert.equal($('result-status').textContent, '待分析', 'HTTP root starts empty');
  assert.equal($('result-score').textContent, '—');
  assert.equal(test.state.history.length, 0);
  let assertions = 3;
  const record = { id: 'conflicting-record', title: '合成测试内容', text: '合成测试内容', probability: .557922, status: 'high', riskReason: '较高虚假风险', reviewAdvice: '疑似虚假，应优先处置', hasImage: false, isExample: false, date: '2026-10-24', time: '12:00:00', raw: { decision: { label: 'fake', label_name_zh: '疑似虚假' }, risk: { level: 'high', reason: '较高虚假风险' } } };
  test.renderResult(record);
  assert.equal($('result-status').textContent, '待复核');
  assert.equal($('result-score').textContent, '55.8%');
  assert.doesNotMatch(visibleOverview(), /高风险|较高虚假风险|疑似虚假|不确定/);
  assert.equal(test.state.current.status, 'uncertain');
  assert.equal(JSON.parse($('json-output').textContent).user_conclusion.label, '待复核');
  assert.equal(JSON.parse($('json-output').textContent).raw_model_output.decision.label, 'fake', 'raw binary stays available for traceability');
  await $('copy-summary').listeners.click[0]();
  await flush();
  assert.match(calls.copied.at(-1), /研判结论：待复核/);
  assert.doesNotMatch(calls.copied.at(-1), /高风险|较高虚假风险|疑似虚假|不确定/);
  $('download-json').listeners.click[0]();
  const downloaded = JSON.parse(await calls.blobs.at(-1).text());
  assert.equal(downloaded.user_conclusion.status, 'uncertain');
  assert.equal(downloaded.user_conclusion.label, '待复核');
  assert.equal(downloaded.raw_model_output.risk.level, 'high');
  assertions += 11;
  test.rememberRecord(record);
  test.renderHistory();
  assert.equal($('history-body').children[0].children[1].children[0].textContent, '待复核');
  $('history-filter').value = 'uncertain';
  test.renderHistory();
  assert.equal($('history-body').children.length, 1);
  assert.equal($('history-count').textContent, '1 条本次记录');
  assertions += 3;
  for (const probability of [.4, .6]) {
    test.renderResult({ ...record, probability, id: 'edge-' + probability });
    assert.equal($('result-status').textContent, '待复核');
    assert.equal(test.resultPayload(test.state.current).user_conclusion.label, '待复核');
    assertions += 2;
  }
  test.renderResult({ ...record, probability: .85, status: 'low', riskReason: '低风险' });
  assert.equal($('result-status').textContent, '高风险', 'stale record.status cannot override probability');
  test.renderResult({ ...record, probability: .2, status: 'high' });
  assert.equal($('result-status').textContent, '低风险');
  assertions += 2;
  for (const action of [() => test.renderResult(record), () => test.recallRecord(record), () => test.loadExample('example-uncertain', false), () => test.setEmptyResult()]) {
    $('technical-details').open = true;
    $('json-wrap').hidden = false;
    $('toggle-json').attributes['aria-expanded'] = 'true';
    action();
    assert.equal($('technical-details').open, false);
    assert.equal($('json-wrap').hidden, true);
    assert.equal($('toggle-json').attributes['aria-expanded'], 'false');
    assertions += 3;
  }
  test.loadExample('example-uncertain', false);
  await flush();
  assert.equal($('heat-layer').hidden, false);
  assert.equal($('heat-toggle').checked, true);
  assert.ok(calls.drawImage > 0, 'example heat is painted over the embedded image');
  assert.match($('evidence-image').src, /^data:image\/webp;base64,/);
  assert.match($('result-source').textContent, /效果示例.*非本次模型结果/);
  assert.match($('form-feedback').textContent, /固定演示数值.*AI.*非本次模型推理/);
  assert.equal(test.state.history.length, 1, 'example never enters real history');
  assertions += 7;
  // Starting from an example submits the photo actually shown in the form and uses the fresh API result.
  const shownExampleImage = $('evidence-image').src;
  assert.equal(test.state.restoredImage, shownExampleImage);
  const exampleRequest = $('news-form').listeners.submit[0]({ preventDefault() {} });
  await flush();
  assert.equal(calls.pending.at(-1).body.image_base64, shownExampleImage);
  const freshCam = [[.4, .8], [.5, 1]];
  calls.pending.at(-1).resolve({ ok: true, json: async () => ({ decision: { probability_fake: .23456, label: 'real' }, risk: { level: 'low' }, modalities: { has_image: true }, image_evidence: { cam: freshCam } }) });
  await exampleRequest;
  assert.equal(test.state.current.probability, .23456);
  assert.equal($('result-score').textContent, '23.5%');
  assert.equal($('result-status').textContent, '低风险');
  assert.equal(test.state.current.isExample, false);
  assert.equal(test.state.current.demoHeat, false);
  assert.equal(JSON.stringify(test.state.current.cam), JSON.stringify(freshCam));
  assertions += 8;
  const zero = { ...record, id: 'zero-cam', hasImage: true, imageUrl: 'data:image/png;base64,AAAA', cam: [[0, 0], [0, 0]], demoHeat: false };
  const beforeDraw = calls.drawImage;
  test.renderResult(zero);
  await flush();
  assert.equal($('heat-layer').hidden, true);
  assert.equal($('heat-toggle').disabled, true);
  assert.equal($('image-grid-detail').hidden, true);
  assert.equal($('heat-toggle').checked, false);
  assert.match($('image-view-description').textContent, /非零信号.*仅展示原图/);
  assert.equal(calls.drawImage, beforeDraw, 'zero signal never creates a synthetic heatmap');
  assertions += 6;
  assert.equal(test.hasCamSignal([[1, 0], [0]]), false);
  assert.equal(test.hasCamSignal([[1, 0], [0, .5]]), true);
  assertions += 2;
  // Retain request cancellation and stale-response protection after the presentation rewrite.
  test.setEmptyResult();
  $('news-text').value = '第一条用于请求竞态核验的合成内容';
  const submit = $('news-form').listeners.submit[0];
  const previous = submit({ preventDefault() {} });
  await flush();
  test.cancelWaiting();
  $('news-text').value = '第二条用于请求竞态核验的合成内容';
  const latest = submit({ preventDefault() {} });
  await flush();
  calls.pending.at(-1).resolve({ ok: true, json: async () => ({ decision: { probability_fake: .557922, label: 'fake' }, risk: { level: 'high', reason: '较高虚假风险' }, modalities: { has_image: false } }) });
  await Promise.all([previous, latest]);
  assert.equal(test.state.current.text, '第二条用于请求竞态核验的合成内容');
  assert.equal($('result-status').textContent, '待复核');
  assert.equal($('analyze-button').disabled, false);
  assert.equal(test.state.activeRequest, null);
  assertions += 4;
  // Actual container resizes repaint the backing canvas, even without window resize.
  test.loadExample('example-uncertain', false);
  await flush();
  Object.assign($('evidence-image'), { clientWidth: 320, clientHeight: 180 });
  calls.resizeObservers[0]();
  await flush();
  assert.equal($('heat-layer').width, 320);
  assert.equal($('heat-layer').height, 180);
  assert.equal($('heat-layer').hidden, false);
  assertions += 3;
  // A failed drawing context must not leave a success caption or visible overlay.
  const getContext = $('heat-layer').getContext;
  $('heat-layer').getContext = () => null;
  test.updateHeatView();
  await flush();
  assert.equal($('heat-layer').hidden, true);
  assert.equal($('heat-toggle').disabled, true);
  assert.equal($('heat-legend').hidden, true);
  assert.equal($('heat-toggle').checked, false);
  assert.match($('image-view-description').textContent, /叠加失败.*仅显示原图/);
  assertions += 5;
  $('heat-layer').getContext = getContext;
  $('heat-toggle').checked = true;
  test.updateHeatView();
  await flush();
  assert.equal($('heat-layer').hidden, false);
  assertions += 1;
  // A delayed image decode must not paint over the next record or its caption.
  let resolveOldDecode;
  $('evidence-image').complete = false;
  $('evidence-image').decode = () => new Promise(resolve => { resolveOldDecode = resolve; });
  test.updateHeatView();
  const drawBeforeSwitch = calls.drawImage;
  test.renderResult(record);
  resolveOldDecode();
  await flush();
  assert.equal(calls.drawImage, drawBeforeSwitch);
  assert.equal($('heat-layer').hidden, true);
  assert.doesNotMatch($('image-view-description').textContent, /叠加失败/);
  assertions += 3;
  $('evidence-image').complete = true;
  $('evidence-image').decode = async () => {};
  // Reading a new image blocks submit; reset and example recall invalidate the late decode.
  const pendingImage = test.selectImage({ name: 'synthetic-pending.png', type: 'image/png', size: 64 });
  assert.equal(test.state.imageLoading, true);
  assert.equal($('analyze-button').disabled, true);
  const requestCount = calls.pending.length;
  await submit({ preventDefault() {} });
  assert.equal(calls.pending.length, requestCount);
  $('reset-workbench').listeners.click[0]();
  calls.pendingImages.shift().onload();
  await pendingImage;
  assert.equal(test.state.selectedFile, null);
  assert.equal(test.state.imageUrl, null);
  assert.equal($('analyze-button').disabled, false);
  assertions += 6;
  const pendingAfterExample = test.selectImage({ name: 'synthetic-replaced.png', type: 'image/png', size: 64 });
  test.loadExample('example-uncertain', false);
  const currentExampleImage = test.state.restoredImage;
  calls.pendingImages.shift().onload();
  await pendingAfterExample;
  assert.equal(test.state.selectedFile, null);
  assert.equal(test.state.restoredImage, currentExampleImage);
  assert.equal(documentView(), 'example');
  assertions += 3;
  console.log('Shared workbench: ' + assertions + ' assertions passed (conclusion, copy, export, history, boundaries, drawer reset, heat overlay, zero signal, cancellation race).');
  process.exit(0);
})().catch(error => { console.error(error); process.exit(1); });

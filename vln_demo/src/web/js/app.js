// Boot: fetch the config, build the panels, hang them in the dock, start the poll.
//
// Order matters in one place only -- CFG has to be filled before any panel is
// built, because the dropdowns are rendered from it.

import { $ } from './util.js';
import { CFG, onStatus, startPolling } from './store.js';
import { getJSON, cmd } from './api.js';
import { initTheme } from './theme.js';
import { initLock } from './lock.js';
import { Dock } from './dock.js';
import { DEFAULT, buildPanels } from './layout.js';
import { modelLabel, assetDisplayLabel } from './controls.js';

const STATE_ZH = {
  idle: '● 空闲', running: '● 模型运行中', loading: '● 加载场景中',
  manual: '● 手动驾驶', done: '● 已完成', stopped: '● 已停止', error: '● 出错',
};

function header(s) {
  const st = $('state');
  st.textContent = STATE_ZH[s.status] || s.status;
  st.className = 'pill s-' + s.status;
  $('p-model').textContent = modelLabel();
  if (s.backend_label) $('p-backend').textContent = assetDisplayLabel({ label: s.backend_label });
  $('p-scene').textContent = s.scene_title || s.scene;
  $('pill-scene').title = s.scene_title ? s.scene_title + ' · ' + s.scene : s.scene;
  $('p-ep').textContent = s.episode_id;
  $('p-step').textContent = s.step;
  // NaN distance means "this episode has no goal" (the 3DRS and ABotN sets have
  // none), and NaN !== NaN is how it arrives from JSON.
  $('p-dist').textContent = s.dist != null && s.dist === s.dist ? s.dist.toFixed(2) + ' m' : '-';
  if ($('distance-label')) $('distance-label').textContent =
    s.evaluation?.kind === 'object_region' ? '距参考终点' : '距目标';
  $('p-res').hidden = !s.result;
  if (s.result) $('p-res').textContent = s.result;
  // The views repaint themselves off the MJPEG streams; all the poll owes them is
  // the rate read-out.
  $('fps').textContent = s.fps ? s.fps.toFixed(1) + ' fps' : '';
}

async function waitForBrain() {
  const boot = $('boot');
  const state = $('state');
  state.textContent = '● 等待大脑就绪';
  state.className = 'pill s-loading';
  for (;;) {
    try {
      const readiness = await getJSON('/readiness', { signal: AbortSignal.timeout(5000) });
      if (boot) boot.textContent = readiness.message || '正在等待大脑服务就绪…';
      if (readiness.label) $('p-model').textContent = readiness.label;
      if (readiness.ready === true) return;
    } catch (_) {
      if (boot) boot.textContent = '暂时无法连接服务，正在重试；大脑就绪后自动进入控制台。';
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

waitForBrain().then(() => getJSON('/config.json')).then((cfg) => {
  Object.assign(CFG, cfg);

  // The panels are built detached and attached by render(); nothing between
  // here and there may touch the document for a panel's own ids.
  const dock = new Dock($('dock'), buildPanels(), DEFAULT);
  dock.render();

  const boot = $('boot');
  if (boot) boot.remove();

  initTheme($('btn-theme'));
  // Before the panels' own listeners see anything: initLock installs a
  // capture-phase guard, and a guard added after a click is already in flight
  // would let that one through.
  initLock($('btn-lock'), cmd);
  $('btn-layout').addEventListener('click', () => {
    if (confirm('把窗口布局恢复成默认？')) dock.reset();
  });

  onStatus(header);
  startPolling(250);
}).catch(fail);

/** Boot failed. Say so on the page -- not just in the console.
 *
 * This used to write into #boot, which by then had already been removed, so a
 * throw anywhere in boot left the page sitting on 「连接中…」 with no panels and
 * no message: the one failure mode that looks exactly like a hung server.
 */
function fail(e) {
  console.error('[vln] boot failed', e);
  const boot = $('boot');
  const msg = '控制台前端启动失败：' + (e && e.message ? e.message : e)
            + '（详情见浏览器控制台）';
  if (boot) { boot.textContent = msg; return }
  const st = $('state');
  if (st) { st.textContent = '● 前端出错'; st.className = 'pill s-error'; st.title = msg }
  const dock = $('dock');
  if (dock) {
    const p = document.createElement('div');
    p.className = 'boot err';
    p.textContent = msg;
    dock.replaceChildren(p);
  }
}

// Handy from the console when a layout gets into a shape that is hard to undo.
window.vlnResetLayout = () => { localStorage.removeItem('vln.layout.v1'); location.reload() };

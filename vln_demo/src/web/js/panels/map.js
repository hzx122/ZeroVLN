// 俯视地图 -- the same MJPEG treatment as the first-person view, plus
// double-click to jump.
//
// The click is sent as a fraction of the image (0..1 across, 0..1 down), never
// as pixels. The server resizes the map to a fixed height and the browser
// resizes it again to fit the panel, so pixels would be wrong twice over;
// fractions survive both. Inverting them back to a world point is the server's
// job -- see map_click_world() in vln_model.py, which knows about the rot90 and
// about the three different map projections.
//
// The image fills the panel with object-fit: contain (panels.css), so the
// element box is the whole frame and the picture is letterboxed inside it. The
// fraction therefore has to be measured against the picture, not the element --
// that is pictureRect(), and it is the whole reason a click still lands where
// you aimed it now that the map is scaled up instead of sitting at 420px in the
// middle of the frame.

import { el } from '../util.js';
import { cmd } from '../api.js';
import { blocked } from '../lock.js';
import { stream } from './view.js';
import { onStatus } from '../store.js';

/** Where the picture actually is inside an object-fit: contain <img>.
 *
 * contain picks the largest scale that fits both axes and centres the result, so
 * this is that arithmetic read back out. Falls back to the element box before
 * the first frame arrives, when naturalWidth is still 0. */
export function pictureRect(img) {
  const r = img.getBoundingClientRect();
  const nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw || !nh || !r.width || !r.height) return r;
  const s = Math.min(r.width / nw, r.height / nh);
  const w = nw * s, h = nh * s;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2,
           width: w, height: h };
}

export function build() {
  const legend = el('span.pv-foot', { hidden: true });
  onStatus((s) => {
    const e = s.evaluation;
    legend.hidden = e?.kind !== 'object_region';
    if (!legend.hidden) {
      legend.textContent = e.available
        ? '绿色填充 = 成功区域（物体包围盒外扩1m，含边界）；深绿框 = 目标物体；橙圈 = 参考终点。当前位置：'
          + (e.success ? '区域内' : '区域外')
        : '成功区域不可用：' + e.reason + '；本次无法按物体区域判定。';
    }
  });
  const img = stream('map', '/stream_map.mjpg');
  const stage = el('div.pv-stage.clickable', {}, [
    img, el('span.pv-tag', { text: 'MAP' }),
  ]);

  img.addEventListener('dblclick', (e) => {
    // 双击落在 <img> 上，不是表单控件，捕获拦截放行了它 —— 传送最该被锁住，所以
    // 这里在画那个红点之前先问一次锁（不然锁着还是会闪一下涟漪像是生效了）。
    if (blocked('goto')) return;
    const r = pictureRect(img);
    if (!r.width || !r.height) return;
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    // In the letterbox rather than on the map: there is no world point there,
    // and clamping would silently teleport to an edge the user did not click.
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const box = stage.getBoundingClientRect();
    ping(stage, e.clientX - box.left, e.clientY - box.top);
    cmd('goto', { x: fx.toFixed(5), y: fy.toFixed(5) });
  });
  // Double-clicking an image otherwise selects it, which leaves the map looking
  // highlighted for the rest of the session.
  img.addEventListener('mousedown', (e) => { if (e.detail > 1) e.preventDefault() });

  return el('div.dk-pane', {}, [
    stage,
    legend,
    el('span.pv-foot', { text: '双击地图 = 传送到那个点（朝向不变，轨迹从新位置重新开始）' }),
  ]);
}

function ping(stage, x, y) {
  const dot = el('span.pv-ping');
  dot.style.left = x + 'px';
  dot.style.top = y + 'px';
  stage.appendChild(dot);
  setTimeout(() => dot.remove(), 700);
}

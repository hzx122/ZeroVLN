// 第一视角 -- MJPEG stream plus double-click-to-move.
//
// The poll does not touch it: repainting from /status was the old frame-rate
// ceiling (four repaints a second however fast the simulator ran).

import { el } from '../util.js';
import { cmd } from '../api.js';
import { blocked } from '../lock.js';

/** A stream that reconnects itself.
 *
 * A dropped stream -- reload, proxy timeout, server restart -- leaves a frozen
 * image with no hint that it is stale, so retry rather than waiting for someone
 * to press refresh.
 */
export function stream(id, path) {
  const img = el('img', { id, src: path, alt: '' });
  img.addEventListener('error', () => {
    setTimeout(() => { img.src = path + '?r=' + Date.now() }, 1000);
  });
  return img;
}

export function build() {
  const img = stream('rgb', '/stream.mjpg');
  const stage = el('div.pv-stage.clickable', {}, [
    img,
    el('span.pv-tag', { text: 'RGB' }),
  ]);

  img.addEventListener('dblclick', (event) => {
    if (blocked('screen point move')) return;
    const rect = pictureRect(img);
    if (!rect.width || !rect.height) return;
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    // Ignore the object-fit letterbox instead of silently moving to an edge.
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const stageRect = stage.getBoundingClientRect();
    ping(stage, event.clientX - stageRect.left, event.clientY - stageRect.top);
    cmd('screen_point_move', {
      x: Math.round(fx * 1000), y: Math.round(fy * 1000),
    });
  });
  img.addEventListener('mousedown', (event) => {
    if (event.detail > 1) event.preventDefault();
  });

  return el('div.dk-pane', {}, [
    stage,
    el('div.pv-foot', {}, [
      el('span', { text: '双击第一视角中任意有效深度位置，可直接执行视觉点导航' }),
    ]),
  ]);
}

function pictureRect(img) {
  const rect = img.getBoundingClientRect();
  const nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw || !nh || !rect.width || !rect.height) return rect;
  const scale = Math.min(rect.width / nw, rect.height / nh);
  const width = nw * scale, height = nh * scale;
  return {
    left: rect.left + (rect.width - width) / 2,
    top: rect.top + (rect.height - height) / 2,
    width, height,
  };
}

function ping(stage, x, y) {
  const dot = el('span.pv-ping');
  dot.style.left = x + 'px';
  dot.style.top = y + 'px';
  stage.appendChild(dot);
  setTimeout(() => dot.remove(), 700);
}

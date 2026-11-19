// 手动控制 -- hold W/A/S/D to drive.
//
// Manual driving is continuous: the server walks for as long as a key is down, so
// what we owe it is an honest answer to "which keys are down right now". We send
// the whole set, not deltas, and keep re-sending it while anything is held. The
// server releases everything if the heartbeat stops (HOLD_TIMEOUT), which is what
// stops the robot when this tab loses focus, sleeps, or the wifi drops mid-press
// -- a missed keyup would otherwise mean a robot that never stops.

import { el, trim } from '../util.js';
import { CFG } from '../store.js';
import { cmd } from '../api.js';
import { blocked } from '../lock.js';

const KEYS = ['w', 'a', 's', 'd'];
const BEAT = 150;          // ms; comfortably under the server's 600 ms timeout
const KEYMAP = {
  w: 'w', arrowup: 'w', s: 's', arrowdown: 's',
  a: 'a', arrowleft: 'a', d: 'd', arrowright: 'd',
};

export function build() {
  const held = {};
  const btn = {};
  let beat = null;

  const send = () => {
    // 锁上时立刻松开所有键并停掉心跳，而不是每 150ms 撞一次锁：服务端在 lock 那
    // 一刻已经清了 HOLD，这里再抱着按下状态只会让方向键看着还亮着。
    if (blocked('manual hold')) {
      for (const k of KEYS) { delete held[k]; btn[k].classList.remove('on') }
      if (beat) { clearInterval(beat); beat = null }
      return;
    }
    cmd('hold', { k: Object.keys(held).sort().join('') });
    // The pad doubles as a key indicator, so a stuck key (or one this tab never
    // saw released) is visible rather than a mystery.
    for (const k of KEYS) btn[k].classList.toggle('on', !!held[k]);
  };
  const on = (k) => {
    if (held[k]) return;                 // autorepeat, or a second pointer
    held[k] = 1; send();
    if (!beat) beat = setInterval(send, BEAT);
  };
  const off = (k) => {
    if (!held[k]) return;
    delete held[k]; send();
    if (!Object.keys(held).length && beat) { clearInterval(beat); beat = null }
  };
  const allOff = () => KEYS.forEach(off);

  const key = (k, glyph, label) => {
    const b = el('button', { html: glyph + '<span class="k">' + label + '</span>' });
    // pointer events, not click: the pad has to support press-and-hold too, and
    // pointercancel is what fires when a touch turns into a scroll.
    b.addEventListener('pointerdown', (e) => { e.preventDefault(); on(k) });
    for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
      b.addEventListener(ev, () => off(k));
    }
    btn[k] = b;
    return b;
  };

  // Laid out as the physical keys are: W above S, A and D flanking S. Same shape
  // as the arrow-key inverted-T, which is the other thing hands already know --
  // any other arrangement makes you translate between what you see and what you
  // press.
  const pad = el('div.pad', {}, [
    el('span'), key('w', '↑', 'W 前进'), el('span'),
    key('a', '↺', 'A 左转'), key('s', '↓', 'S 后退'), key('d', '↻', 'D 右转'),
  ]);

  document.addEventListener('keydown', (e) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
    const k = e.key.toLowerCase();
    if (k !== 'x' && !KEYMAP[k]) return;
    // 键盘走到这里时 target 是 body，不是表单控件，lock.js 的捕获拦截放行了它 ——
    // 所以驾驶键要自己问一次锁，否则锁着还能用 W/A/S/D 开车。
    if (blocked('manual key ' + k)) { e.preventDefault(); return }
    if (k === 'x') { cmd('manual', { a: 'x' }); e.preventDefault(); return }
    e.preventDefault();
    if (e.repeat) return;                // the OS autorepeat is not extra intent
    on(KEYMAP[k]);
  });
  document.addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    if (KEYMAP[k]) { e.preventDefault(); off(KEYMAP[k]) }
  });
  // Losing focus swallows the keyup, so treat it as one.
  window.addEventListener('blur', allOff);
  document.addEventListener('visibilitychange', () => { if (document.hidden) allOff() });

  const h = CFG.hold || { speed: 1, turn: 45 };
  return el('div.dk-pane.dk-pad', {}, [
    pad,
    el('div.hint.mid', {
      html: '先点「手动」进入手动模式，再<b>按住</b>按键驾驶：<b>按多久就走多远</b>'
        + '（约 ' + trim(h.speed, 2) + ' m/s、' + trim(h.turn, 1) + '°/s），松手就停<br>'
        + 'W+A 同时按住就是边走边转；X 退出手动。后退朝向不变（摄像头朝前，退时看不见身后）',
    }),
  ]);
}

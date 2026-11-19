// 🔒 前端锁 —— 一个后台 agent 在开着这台控制台时，把人从界面上挡出去。
//
// 场景很具体：agent 正在调一个回合，有人打开同一个页面顺手换了个场景 / 改了温度，
// agent 完全看不见这件事发生过，只看到自己的实验莫名其妙变了。按下锁之后，页面上
// 一切能改变控制台状态的操作都被拒绝，并当场弹出提示说明是谁在用。
//
// 三道拦截，缺一不可：
//
//   1. 这里的捕获阶段监听。它在事件走到面板自己的 handler 之前就掐掉，所以像资产
//      集下拉框那种「只在本地暂存、点加载才提交」的控件也动不了 —— 只在 cmd() 里
//      拦是拦不住它的。
//   2. api.js 的 cmd()/postSysPrompt()：真正发请求的那一层，兜住上面漏掉的路径
//      （比如地图双击这种不落在表单控件上的交互）。
//   3. 服务端 vln_model.py 的 _cmd()/do_POST()。锁的**真值在服务端**，前端这两道
//      只负责给出提示：一个开着没刷新的旧标签页、一条手敲的 /cmd URL，都绕不过第
//      三道。前端拦截失效最坏是弹不出提示，不是锁被绕开。
//
// 放行的只有锁按钮自己（不然锁上就解不开了）和纯观看类操作 —— 切标签页、拖面板、
// 滚日志：它们碰不到控制台状态，挡掉只会让人没法看着 agent 干活。
//
// 这个模块**不 import api.js**，方向是反的（api.js import 这里的 blocked）。锁按钮
// 要发的那条 lock/unlock 由 app.js 把 cmd 传进 initLock —— 一个参数换掉一圈循环
// import，比赌打包器/浏览器怎么解环要稳。

import { el } from './util.js';
import { S, onStatus } from './store.js';

export const LOCK_MSG = '后台 Agent 正在调试，只允许 Agent 自主操作，已禁止用户前端任何交互';

let locked = false;
let btn = null;
let toastEl = null;
let toastTimer = null;

export function isLocked() { return locked }

/** 弹错误提示。重复触发只是把同一个框的计时器续上 —— 手动驾驶按住 W 时心跳
 *  150ms 一次，一次一个框会把屏幕铺满。 */
export function warn(text) {
  if (!toastEl) {
    toastEl = el('div.toast');
    document.body.appendChild(toastEl);
  }
  toastEl.textContent = '🔒 ' + (text || LOCK_MSG);
  toastEl.hidden = false;
  toastEl.className = 'toast on';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.className = 'toast'; toastEl.hidden = true }, 3200);
}

/** 锁着就拦下并弹提示。返回 true = 已拦截，调用方别再往下走。 */
export function blocked(what) {
  if (!locked) return false;
  warn();
  if (what) console.warn('[vln] 前端已锁定，忽略：' + what);
  return true;
}

function paint() {
  document.documentElement.dataset.locked = locked ? '1' : '0';
  if (!btn) return;
  btn.textContent = locked ? '🔒' : '🔓';
  btn.className = locked ? 'icon lock-on' : 'icon';
  btn.title = locked
    ? '已锁定：只有后台 Agent 能操作，点这里解锁'
    : '锁定前端：按下之后除本按钮外，页面上一切操作都会被拒绝（留给后台 Agent 调试用）';
}

/** 服务端才是权威：状态从 /status 来，所以多标签页、刷新之后都一致。 */
function sync(s) {
  if (!('locked' in s)) return;          // 老服务端没有这个字段，别把锁抹掉
  const want = !!s.locked;
  if (want === locked) return;
  locked = want;
  paint();
}

/** 捕获阶段拦截：先于面板自己的 handler 跑。
 *
 * 挡的是表单控件（按钮 / 下拉框 / 输入框），也就是这页上所有能改状态的东西；
 * dock 的标签条和分隔线是 div，不在其列，于是锁着也还能切面板、看日志。 */
function guard(e) {
  if (!locked) return;
  const t = e.target;
  if (!t) return;
  if (t === btn || (t.closest && t.closest('#btn-lock'))) return;
  const tag = (t.tagName || '').toUpperCase();
  const isCtl = /^(BUTTON|SELECT|INPUT|TEXTAREA|OPTION)$/.test(tag)
    || !!(t.closest && t.closest('button'));
  if (!isCtl) return;
  e.preventDefault();
  e.stopPropagation();
  if (e.stopImmediatePropagation) e.stopImmediatePropagation();
  warn();
}

export function initLock(button, send) {
  btn = button;
  paint();

  button.addEventListener('click', () => {
    // 乐观地先画上：/status 要 250ms 才回来，那之间的一次点击不该看着像没反应。
    // 真值随下一拍 sync() 回正 —— 服务端要是拒了，这里也会被改回去。
    locked = !locked;
    paint();
    send(locked ? 'lock' : 'unlock');     // api.js 的 cmd()，对 lock/unlock 放行
  });

  for (const ev of ['pointerdown', 'mousedown', 'click', 'dblclick',
                    'change', 'input', 'keydown']) {
    document.addEventListener(ev, guard, true);
  }

  onStatus(sync);
  if (S.status) sync(S);
}

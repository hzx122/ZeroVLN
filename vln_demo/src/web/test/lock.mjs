// 🔒 前端锁：按下之后，页面上什么都不该发得出去。
//
// 锁有三道拦截（见 js/lock.js 的模块注释），这个测试只钉得住其中两道半：
//
//   * cmd() 那道 —— 完全能测，也是最要紧的一道：所有面板发指令都走它。
//   * 按钮和状态同步 —— 能测：/status 的 locked 是真值来源，锁的状态跨标签页、
//     跨刷新都从那里来，本地那个乐观翻转必须被它纠正。
//   * 捕获阶段那道 —— **测不到**。shim 的 `document.addEventListener` 是空实现，
//     它也不做事件冒泡/捕获，于是 lock.js 装的 guard 在这里根本不会被调用。这道
//     只挡「本地暂存、点加载才提交」的控件（资产集/场景下拉框），漏了它锁仍然是
//     牢的（那种控件最终还是要点「加载」，而加载走 cmd()），只是人会看到下拉框
//     动了一下才被拒。宁可在这写清楚，也不要拿一个假的 dispatch 造出「测过了」
//     的错觉。
//
// 服务端那道（vln_model.py 的 _cmd/do_POST）是锁真正的把关处，不归前端测试管。

import './shim.mjs';
import { readFileSync } from 'fs';

const base = process.argv[2] + '/js/';
for (const id of ['state', 'p-model', 'p-backend', 'p-scene', 'pill-scene', 'p-ep',
                  'p-step', 'p-dist', 'p-res', 'fps', 'btn-theme', 'btn-layout',
                  'btn-lock', 'dock', 'boot']) {
  const n = document.createElement('div'); n.id = id; document.body.appendChild(n);
}

const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));

const cmds = [];
let served = false;              // what /status says the lock is
globalThis.fetch = async (url, opt) => {
  if (url.startsWith('/cmd')) {
    cmds.push(url);
    // 服务端的行为照抄 _cmd()：lock/unlock 放行并改状态，其余锁着就拒。
    const m = /[?&]do=([^&]*)/.exec(url);
    const doo = m ? decodeURIComponent(m[1]) : '';
    if (doo === 'lock' || doo === 'unlock') served = doo === 'lock';
    else if (served) return { ok: true, status: 200, text: async () => 'locked: ...' };
  }
  if (url === '/sysprompt') cmds.push('POST /sysprompt');
  if (url === '/react-reference') cmds.push('POST /react-reference');
  const body = url.startsWith('/readiness') ? { ready: true }
             : url.startsWith('/config.json') ? CFG_J
             : url.startsWith('/status') ? { ...STATUS, locked: served }
             : url.startsWith('/prompt') ? { text: 'p', sys_text: 's', tpl: 'en_r1' } : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};

const errs = [];
process.on('uncaughtException', (e) => errs.push(String(e)));
process.on('unhandledRejection', (e) => errs.push(String(e)));

await import(base + 'app.js');
const api = await import(base + 'api.js');
const lock = await import(base + 'lock.js');
await new Promise((r) => setTimeout(r, 200));

const g = (id) => document.getElementById(id);
const poll = () => new Promise((r) => setTimeout(r, 320));   // > 250ms status tick
const step = (name, fn) => fn().then(
  () => console.log('  ok   ' + name),
  (e) => { errs.push(name + ': ' + e.message); console.log('  FAIL ' + name + ' -- ' + e.message) });
const must = (cond, msg) => { if (!cond) throw new Error(msg) };
const since = (n) => cmds.slice(n);

console.log('lock:');

await step('起来是开的', async () => {
  must(!lock.isLocked(), 'should boot unlocked');
  must(g('btn-lock').textContent === '🔓', 'button shows ' + g('btn-lock').textContent);
});

await step('点一下就发 do=lock，并标到 <html> 上', async () => {
  const n = cmds.length;
  g('btn-lock').fire('click');
  await poll();
  must(since(n).some((c) => c.includes('do=lock')), 'no lock command: ' + since(n).join(' '));
  must(lock.isLocked(), 'isLocked() still false');
  must(document.documentElement.getAttribute('data-locked') === '1', 'no data-locked=1');
  must(g('btn-lock').textContent === '🔒', 'button did not repaint');
});

await step('锁着时 cmd() 一条都发不出去', async () => {
  const n = cmds.length;
  for (const a of ['start', 'stop', 'reset', 'switch', 'set_model', 'set_temp',
                   'set_frompose', 'hold', 'manual', 'goto', 'clearlog']) {
    await api.cmd(a, { x: 1 });
  }
  must(since(n).length === 0, '这些指令漏了出去：' + since(n).join(' '));
});

await step('锁着时 /sysprompt 也发不出去', async () => {
  const n = cmds.length;
  const r = await api.postSysPrompt('hello');
  must(since(n).length === 0, 'sysprompt leaked: ' + since(n).join(' '));
  must(r && r.locked === true, 'postSysPrompt should report locked, got ' + JSON.stringify(r));
});

await step('锁着时参考图也发不出去', async () => {
  const n = cmds.length;
  const r = await api.postReactReference({ kind: 'clear' });
  must(since(n).length === 0, 'react-reference leaked: ' + since(n).join(' '));
  must(r && r.locked === true, 'postReactReference should report locked');
});

await step('弹出的提示就是那句原话', async () => {
  const toast = document.body.children.find((c) => String(c.className).startsWith('toast'));
  must(toast, 'no .toast in the document');
  must(toast.textContent.includes('后台 Agent 正在调试'), 'toast says: ' + toast.textContent);
  must(toast.textContent.includes('已禁止用户前端任何交互'), 'toast says: ' + toast.textContent);
  must(!toast.hidden, 'toast is hidden while it should be showing');
});

await step('地图双击不发 goto', async () => {
  const map = await import(base + 'panels/map.js');
  const pane = map.build();
  const img = pane.all.find((n) => n.id === 'map');
  const n = cmds.length;
  img.fire('dblclick', { clientX: 400, clientY: 300 });
  await poll();
  must(since(n).length === 0, 'goto leaked: ' + since(n).join(' '));
});

await step('服务端说解锁了，页面就跟着解锁', async () => {
  served = false;                       // 别的标签页 / agent 解的锁
  await poll();
  must(!lock.isLocked(), 'page still locked after the server unlocked');
  must(g('btn-lock').textContent === '🔓', 'button still shows ' + g('btn-lock').textContent);
});

await step('解锁后指令又通了', async () => {
  const n = cmds.length;
  await api.cmd('stop');
  must(since(n).some((c) => c.includes('do=stop')), 'stop did not go through after unlock');
});

await step('服务端说锁上了，页面就跟着锁上（另一个标签页按的）', async () => {
  served = true;
  await poll();
  must(lock.isLocked(), 'page did not follow the server into the locked state');
  const n = cmds.length;
  await api.cmd('start');
  must(since(n).length === 0, 'start leaked after a server-side lock');
});

console.log('\n' + (errs.length ? 'FAILURES:\n' + errs.join('\n')
                                : 'lock ok: 锁上之后没有一条指令发得出去，状态跟着服务端走'));
process.exit(errs.length ? 1 : 0);

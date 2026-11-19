// One shared server-side PTY. The server starts `pi` automatically; all
// subsequent interaction still travels as ordinary terminal bytes.

import { el } from '../util.js';

const post = (payload) => fetch('/terminal', {
  method: 'POST',
  body: JSON.stringify(payload),
  headers: { 'Content-Type': 'application/json; charset=utf-8' },
}).then((response) => response.json());

function decode64(text) {
  if (!text) return new Uint8Array();
  const raw = atob(text);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

function encode64(text) {
  let raw = '';
  for (let i = 0; i < text.length; i++) raw += String.fromCharCode(text.charCodeAt(i) & 255);
  return btoa(raw);
}

export function build() {
  const host = el('div.terminal-host');
  const notice = el('div.hint.flat', { id: 'workspace-reset-notice', hidden: true });
  notice.style.overflowWrap = 'anywhere';
  const pane = el('div.dk-pane.terminal-pane', {}, [host, notice]);
  const restart = el('button', { text: '重启终端', title: '结束当前 shell 并重新打开 bash' });
  const reset = el('button', { id: 'workspace-reset', text: '清空工作区',
    title: '先按时间戳备份旧工作区，再只保留固定 AGENTS.md 与隐藏 .pi 配置链接，并启动全新 pi-agent' });
  reset.disabled = true; // older servers must not expose a nonworking reset action

  // The dependency is loaded as a plain script before app.js. Keeping this
  // guard makes a missing vendor file fail inside this panel instead of taking
  // down the whole VLN console.
  if (typeof globalThis.Terminal !== 'function') {
    host.appendChild(el('div.terminal-unavailable', {
      text: '终端组件加载失败，请刷新页面。',
    }));
    restart.disabled = reset.disabled = true;
    return { el: pane, tools: [restart, reset] };
  }

  const term = new globalThis.Terminal({
    cursorBlink: true,
    cursorStyle: 'block',
    disableStdin: false,
    scrollback: 5000,
    fontSize: 13,
    lineHeight: 1.15,
    fontFamily: 'Menlo, Monaco, Consolas, "Liberation Mono", monospace',
    theme: {
      background: '#0f1117', foreground: '#d7dae0', cursor: '#f2f2f2',
      selectionBackground: '#294f78', black: '#16181d', red: '#e06c75',
      green: '#98c379', yellow: '#e5c07b', blue: '#61afef',
      magenta: '#c678dd', cyan: '#56b6c2', white: '#d7dae0',
    },
  });
  term.open(host);

  let generation = 0;
  let cursor = 0;
  let stopped = false;
  let sendChain = Promise.resolve();
  let lastCols = 0;
  let lastRows = 0;
  let resetting = false;
  let resetSupported = false;
  let terminalEpoch = 0;

  const send = (payload) => {
    if (resetting) return Promise.resolve();
    sendChain = sendChain.then(() => post(payload)).catch(() => {});
    return sendChain;
  };

  const fit = () => {
    if (pane.hidden || !host.clientWidth || !host.clientHeight) return;
    // xterm exposes its measured cell size through the render service. Fall
    // back to the configured 13px monospace metrics during its first frame.
    const cell = term._core && term._core._renderService
      && term._core._renderService.dimensions
      && term._core._renderService.dimensions.css.cell;
    const cw = cell && cell.width > 0 ? cell.width : 7.9;
    const ch = cell && cell.height > 0 ? cell.height : 16;
    const cols = Math.max(20, Math.floor((host.clientWidth - 12) / cw));
    const rows = Math.max(5, Math.floor((host.clientHeight - 10) / ch));
    if (cols === lastCols && rows === lastRows) return;
    lastCols = cols; lastRows = rows;
    term.resize(cols, rows);
    send({ op: 'resize', cols, rows });
  };

  term.onData((data) => send({ op: 'input', data }));
  term.onBinary((data) => send({ op: 'input', binary: encode64(data) }));

  const poll = async () => {
    if (stopped) return;
    const epoch = terminalEpoch;
    try {
      const response = await fetch('/terminal?since=' + cursor
        + '&generation=' + generation + '&wait=15');
      const reply = await response.json();
      if (!reply.ok) throw new Error(reply.err || 'terminal poll failed');
      if (resetting || epoch !== terminalEpoch) {
        setTimeout(poll, 200);
        return;
      }
      resetSupported = reply.workspace_reset_supported === true;
      reset.disabled = !resetSupported;
      if (reply.reset || reply.generation !== generation) term.reset();
      generation = reply.generation;
      cursor = reply.cursor;
      const bytes = decode64(reply.output);
      if (bytes.length) term.write(bytes);
      requestAnimationFrame(fit);
      poll();
    } catch (_) {
      setTimeout(poll, 1000);
    }
  };

  const reopen = async (clearWorkspace) => {
    if (resetting) return;
    resetting = true;
    terminalEpoch++;
    restart.disabled = reset.disabled = true;
    notice.hidden = false;
    notice.textContent = clearWorkspace ? '正在结束 pi-agent、备份并重置工作区…' : '正在重启终端…';
    try {
      await sendChain;
      const reply = await post({ op: clearWorkspace ? 'reset_workspace' : 'restart',
        generation, cols: term.cols, rows: term.rows });
      if (!reply.ok) throw new Error(reply.err || 'restart failed');
      generation = reply.generation;
      cursor = reply.cursor;
      term.reset();
      const bytes = decode64(reply.output);
      if (bytes.length) term.write(bytes);
      term.focus();
      notice.textContent = clearWorkspace
        ? '工作区已重置；旧内容备份在：' + reply.workspace_reset.backup
        : '终端已重启';
    } catch (error) {
      notice.textContent = (clearWorkspace ? '工作区重置失败：' : '重启失败：') + error.message;
    } finally {
      resetting = false;
      restart.disabled = false;
      reset.disabled = !resetSupported;
    }
  };
  restart.addEventListener('click', () => reopen(false));
  reset.addEventListener('click', () => reopen(true));

  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => requestAnimationFrame(fit)).observe(host);
  } else {
    window.addEventListener('resize', fit);
  }
  requestAnimationFrame(fit);
  poll();

  return { el: pane, tools: [restart, reset] };
}

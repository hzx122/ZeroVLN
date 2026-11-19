import './shim.mjs';

const base = process.argv[2] + '/js/';
globalThis.requestAnimationFrame = window.requestAnimationFrame;

let instance = null;
class FakeTerminal {
  constructor() {
    instance = this;
    this.cols = 80; this.rows = 24; this.writes = []; this.resets = 0;
  }
  open() {}
  onData(fn) { this.data = fn }
  onBinary(fn) { this.binary = fn }
  write(data) { this.writes.push(Buffer.from(data).toString()) }
  writeln(data) { this.writes.push(data) }
  reset() { this.resets++ }
  resize(cols, rows) { this.cols = cols; this.rows = rows }
  focus() { this.focused = true }
}
globalThis.Terminal = FakeTerminal;

const posts = [];
let polls = 0;
const pendingPolls = [];
globalThis.fetch = async (url, opts = {}) => {
  if (opts.method === 'POST') {
    const payload = JSON.parse(opts.body);
    posts.push(payload);
    return { json: async () => payload.op === 'reset_workspace'
      ? { ok: true, generation: 3, cursor: 0, output: '',
          workspace_reset: { backup: '/workspace-backups/timestamp' } }
      : payload.op === 'restart'
      ? { ok: true, generation: 2, cursor: 7,
          output: Buffer.from('prompt pi').toString('base64') }
      : { ok: true } };
  }
  polls++;
  if (polls > 1) return new Promise((resolve) => pendingPolls.push(resolve));
  return { json: async () => ({
    ok: true, generation: 1, cursor: 5, reset: true,
    output: Buffer.from('shell').toString('base64'), running: true,
    workspace_reset_supported: true,
  }) };
};

const terminal = await import(base + 'panels/terminal.js');
const panel = terminal.build();
await new Promise((resolve) => setTimeout(resolve, 20));
const errors = [];
if (!instance || !instance.writes.join('').includes('shell')) errors.push('PTY output not rendered');
instance.data('pi\r');
await new Promise((resolve) => setTimeout(resolve, 10));
if (!posts.some((p) => p.op === 'input' && p.data === 'pi\r')) errors.push('keyboard input not sent');
panel.tools[0].fire('click');
await new Promise((resolve) => setTimeout(resolve, 10));
if (!posts.some((p) => p.op === 'restart')) errors.push('restart not sent');
if (!instance.writes.join('').includes('prompt pi')) errors.push('restart output not rendered');
if (!instance.focused) errors.push('terminal not focused after restart');
panel.tools[1].fire('click');
panel.tools[1].fire('click'); // an accidental double click must not rotate twice
await new Promise((resolve) => setTimeout(resolve, 10));
const resets = posts.filter((p) => p.op === 'reset_workspace');
if (resets.length !== 1 || resets[0].generation !== 2) errors.push('reset should use current generation exactly once');
if (!panel.el.textContent.includes('/workspace-backups/timestamp')) errors.push('backup path not shown');

// An old in-flight poll must not overwrite a newly reset terminal.
pendingPolls.shift()({ json: async () => ({ ok: true, generation: 1, cursor: 5,
  output: Buffer.from('stale-output').toString('base64') }) });
await new Promise((resolve) => setTimeout(resolve, 250));
if (instance.writes.join('').includes('stale-output')) errors.push('stale poll overwrote reset');
// A backend process reload may legitimately restart generation numbering.
pendingPolls.shift()({ json: async () => ({ ok: true, generation: 1, cursor: 5,
  output: Buffer.from('new-server').toString('base64'), workspace_reset_supported: true }) });
await new Promise((resolve) => setTimeout(resolve, 10));
if (!instance.writes.join('').includes('new-server')) errors.push('backend reload did not reconnect');

console.log(errors.length ? 'terminal failures: ' + errors.join('; ')
                          : 'terminal ok: output + input + one-click restart');
process.exit(errors.length ? 1 : 0);

import './shim.mjs';
import { readFileSync } from 'fs';
const base = process.argv[2] + '/js/';
for (const id of ['state','p-model','p-backend','p-scene','pill-scene','p-ep','p-step',
                  'p-dist','p-res','fps','btn-theme','btn-layout','btn-lock','dock','boot']) {
  const n = document.createElement('div'); n.id = id; document.body.appendChild(n);
}
const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));
let prompts = 0;
globalThis.fetch = async (url) => {
  if (url.startsWith('/prompt')) prompts++;
  const body = url.startsWith('/readiness') ? { ready: true }
             : url.startsWith('/config.json') ? CFG_J
             : url.startsWith('/status') ? STATUS
             : url.startsWith('/prompt') ? { text: 'u', sys_text: 's', tpl: 'en_r1' } : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};
await import(base + 'app.js');
await new Promise((r) => setTimeout(r, 200));

const dock = document.getElementById('dock');
const tabs = dock.querySelectorAll('.dk-tab');
console.log('tabs:', tabs.map((t) => t.textContent.trim()).join(' | '));
console.log('/prompt fetches while 提示词 is the hidden tab:', prompts, prompts === 0 ? '(correct)' : '(should be 0)');

const pt = tabs.find((t) => t.textContent.includes('提示词'));
if (!pt) { console.log('FAIL: no 提示词 tab'); process.exit(1) }
pt.fire('mousedown', { button: 0, clientX: 10, clientY: 10, detail: 1, preventDefault(){} });
pt.fire('click', { button: 0 });
await new Promise((r) => setTimeout(r, 350));
console.log('/prompt fetches after activating it:', prompts, prompts > 0 ? '(correct)' : '(should be > 0)');
process.exit(prompts > 0 ? 0 : 1);

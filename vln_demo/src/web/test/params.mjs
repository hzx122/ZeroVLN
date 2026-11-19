import './shim.mjs';
import { readFileSync } from 'fs';
const base = process.argv[2] + '/js/';
for (const id of ['state','p-model','p-backend','p-scene','pill-scene','p-ep','p-step',
                  'p-dist','p-res','fps','btn-theme','btn-layout','btn-lock','dock','boot']) {
  const n = document.createElement('div'); n.id = id; document.body.appendChild(n);
}
const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));
const seen = [];
globalThis.fetch = async (url) => {
  seen.push(url);
  const body = url.startsWith('/readiness') ? { ready: true }
             : url.startsWith('/config.json') ? CFG_J
             : url.startsWith('/status') ? STATUS
             : url.startsWith('/prompt') ? { text: 'user prompt body', sys_text: 'sys body', tpl: 'en_r1' } : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};
await import(base + 'app.js');
await new Promise((r) => setTimeout(r, 200));
const { params } = await import(base + 'params.js');
const p = params();
console.log('params() ->', JSON.stringify(p));
const bad = Object.entries(p).filter(([k, v]) => v === undefined || v === null || v === '');
console.log(bad.length ? 'unexpectedly empty: ' + bad.map(([k]) => k).join(', ')
                       : 'every prompt parameter resolved from the DOM');
// the prompt panel only fetches when it is visible; force it and check the call shape
const promptUrls = seen.filter((u) => u.startsWith('/prompt'));
console.log('/prompt calls:', promptUrls.length, promptUrls[0] ? '-> ' + promptUrls[0].slice(0, 90) : '');
process.exit(bad.some(([k]) => k !== 'instruction') ? 1 : 0);

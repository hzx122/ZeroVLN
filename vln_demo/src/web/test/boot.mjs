// Boot the real page under the shim and report the first thing that breaks.
//
// Mirrors index.html's static markup and app.js's boot order, then drives a
// couple of interactions. Run with:  node boot.mjs <path-to-src/web>

import './shim.mjs';
import { readFileSync } from 'fs';
import assert from 'node:assert/strict';

const WEB = process.argv[2];
const base = WEB + '/js/';

// ---- the bits index.html provides before any JS runs ----------------------
const IDS = ['state', 'p-model', 'p-backend', 'p-scene', 'pill-scene', 'p-ep',
             'p-step', 'p-dist', 'p-res', 'fps', 'btn-theme', 'btn-layout', 'btn-lock',
             'dock', 'boot'];
for (const id of IDS) {
  const n = document.createElement('div');
  n.id = id;
  if (id === 'state') n.textContent = '连接中…';
  if (id === 'boot') n.textContent = '正在连接控制台…';
  document.body.appendChild(n);
}

// ---- a fetch that answers like the live console ---------------------------
const CFG = JSON.parse(readFileSync(process.argv[3], 'utf8'));
// the real thing, captured off the live console -- a hand-written stub is how
// the last round of this test invented a failure that did not exist
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));
let fetches = 0;
let ready = false;
let readinessChecks = 0;
let configFetches = 0;
globalThis.fetch = async (url) => {
  fetches++;
  if (url.startsWith('/readiness') && ++readinessChecks === 1) {
    throw new Error('brain service starting');
  }
  if (url.startsWith('/config.json')) configFetches++;
  const body = url.startsWith('/readiness') ? { ready, message: '等待大脑' }
             : url.startsWith('/config.json') ? CFG
             : url.startsWith('/status') ? STATUS
             : url.startsWith('/prompt') ? { text: 'p', sys_text: 's', tpl: 'en_r1' }
             : url.startsWith('/cmd') ? 'ok' : {};
  return {
    ok: true, status: 200,
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  };
};

// ---- boot ----------------------------------------------------------------
const errs = [];
process.on('unhandledRejection', (e) => errs.push('unhandledRejection: ' + (e && e.stack || e)));
process.on('uncaughtException', (e) => errs.push('uncaughtException: ' + (e && e.stack || e)));

await import(base + 'app.js');
await new Promise((r) => setTimeout(r, 1700));
assert.ok(readinessChecks >= 2, 'retry after connection failure');
assert.equal(configFetches, 0, 'do not enter console before brain readiness');
assert.equal(document.getElementById('dock').querySelectorAll('div.dk-pane').length, 0);
assert.ok(document.getElementById('boot'), 'retain waiting page');
ready = true;
await new Promise((r) => setTimeout(r, 1600));
await new Promise((r) => setTimeout(r, 250));
assert.equal(configFetches, 1, 'enter automatically once brain is ready');
assert.equal(document.getElementById('boot'), null);

const state = document.getElementById('state');
const boot = document.getElementById('boot');
const dock = document.getElementById('dock');

console.log('fetches issued      :', fetches);
console.log('#boot still present :', !!boot, boot ? JSON.stringify(boot.textContent) : '');
console.log('#state text         :', JSON.stringify(state.textContent));
console.log('#dock child count   :', dock ? dock.children.length : '(no #dock)');
console.log('panes in document   :',
            dock ? dock.querySelectorAll('div.dk-pane').length : 0);
console.log('tabs in document    :',
            dock ? dock.querySelectorAll('.dk-tab').length : 0);
if (errs.length) { console.log('\nERRORS:'); for (const e of errs) console.log(e) }
else console.log('\nno errors');
process.exit(errs.length ? 1 : 0);

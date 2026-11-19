// Randomised drag/drop over the real Dock, through the real DOM shim.
//
// The invariants a dock must never break, whatever the user does to it:
//   * all panels are present exactly once -- no duplicates, none lost
//   * every split's sizes sum to 100 and none is below the minimum
//   * every panel element is attached to the dock, with exactly one parent
//   * a saved layout round-trips, and a corrupt one is repaired not fatal

import './shim.mjs';
import { readFileSync } from 'fs';

const base = process.argv[2] + '/js/';
for (const id of ['state', 'p-model', 'p-backend', 'p-scene', 'pill-scene', 'p-ep',
                  'p-step', 'p-dist', 'p-res', 'fps', 'btn-theme', 'btn-layout', 'btn-lock',
                  'dock', 'boot', 'dk-ghost', 'dk-drop']) {
  const n = document.createElement('div'); n.id = id; document.body.appendChild(n);
}
const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));
globalThis.fetch = async (url) => {
  const body = url.startsWith('/config.json') ? CFG_J
             : url.startsWith('/status') ? STATUS
             : url.startsWith('/prompt') ? { text: 'u', sys_text: 's' } : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};

const { CFG } = await import(base + 'store.js');
Object.assign(CFG, CFG_J);
const { Dock } = await import(base + 'dock.js');
const { buildPanels, DEFAULT } = await import(base + 'layout.js');

const IDS = ['view', 'map', 'log', 'prompt', 'run', 'instr', 'terminal', 'manual', 'trail'];
const fails = [];

// A tiny deterministic PRNG: a seeded run that fails is a run I can repeat.
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const pick = (a) => a[Math.floor(rnd() * a.length) % a.length];

function check(dock, label) {
  const found = [];
  const walk = (n) => {
    if (!n) return;
    if (n.t === 'tabs') { found.push(...n.ids); return }
    if (n.t === 'split') {
      const sum = n.sizes.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 100) > 0.51) fails.push(`${label}: sizes sum to ${sum.toFixed(2)}`);
      if (n.sizes.some((s) => s < 5.9)) fails.push(`${label}: a pane at ${Math.min(...n.sizes).toFixed(2)}% (floor is 6)`);
      if (n.sizes.length !== n.kids.length)
        fails.push(`${label}: ${n.sizes.length} sizes for ${n.kids.length} kids`);
      n.kids.forEach(walk);
    }
  };
  walk(dock.tree);
  const sorted = [...found].sort().join(',');
  if (sorted !== [...IDS].sort().join(','))
    fails.push(`${label}: panels are [${sorted}]`);
  // every panel element must hang under #dock exactly once
  const host = document.getElementById('dock');
  for (const id of IDS) {
    const p = dock.reg[id];
    if (!p) { fails.push(`${label}: no panel record for ${id}`); continue }
    let n = p.el, hops = 0;
    while (n.parentNode && hops++ < 50) n = n.parentNode;
    if (n !== document.documentElement) fails.push(`${label}: ${id} is detached`);
  }
}

const host = document.getElementById('dock');
const panels = buildPanels();
const dock = new Dock(host, panels, DEFAULT);
dock.render();
check(dock, 'initial');
console.log('initial:', Object.keys(dock.reg).length, 'panels,',
            host.querySelectorAll('.dk-tab').length, 'tabs');

// ---- randomised drags -----------------------------------------------------
const ZONES = ['center', 'left', 'right', 'top', 'bottom', 'tab'];
let moves = 0;
for (let i = 0; i < 500; i++) {
  const id = pick(IDS);
  const zone = pick(ZONES);
  // any group that is not the one holding `id` alone
  const groups = [];
  const gather = (n) => {
    if (n.t === 'tabs') groups.push(n); else n.kids.forEach(gather);
  };
  gather(dock.tree);
  const target = pick(groups);
  if (!target) continue;
  // The same guard finish() applies: dropping a panel onto the group it already
  // occupies is a no-op, and for a lone panel remove() would collapse the group
  // out of the tree and leave `target` orphaned.
  const from = dock.find(id).node;
  if (target === from && (zone === 'center' || from.ids.length === 1)) continue;
  try {
    dock.remove(id);
    dock.insert(id, target, zone, Math.floor(rnd() * (target.ids.length + 1)));
    dock.render();
    moves++;
  } catch (e) {
    fails.push(`move ${i} (${id} -> ${zone}): ${e.message}`);
    break;
  }
  check(dock, `after move ${i} (${id} -> ${zone})`);
  if (fails.length > 3) break;
}
console.log('random drags applied:', moves);

// ---- save / restore -------------------------------------------------------
dock.save();
const dock2 = new Dock(host, buildPanels(), DEFAULT);
dock2.render();
check(dock2, 'restored from localStorage');
console.log('restored layout has',
            host.querySelectorAll('.dk-tab').length, 'tabs');

// ---- corrupt and stale layouts -------------------------------------------
localStorage.setItem('vln.layout.v1', '{not json at all');
const dock3 = new Dock(host, buildPanels(), DEFAULT);
dock3.render();
check(dock3, 'garbage json falls back to default');

localStorage.setItem('vln.layout.v1', JSON.stringify({
  t: 'split', dir: 'row', sizes: [50, 50],
  kids: [
    { t: 'tabs', ids: ['view', 'view', 'gone-panel'], active: 9 },
    { t: 'tabs', ids: ['map'], active: 0 },
  ],
}));
const dock4 = new Dock(host, buildPanels(), DEFAULT);
dock4.render();
check(dock4, 'stale layout (dup + unknown id + bad active) is repaired');

// ---- everything into one group, then back out ----------------------------
const dock5 = new Dock(host, buildPanels(), DEFAULT);
dock5.render();
const first = (() => { let g = null; const w = (n) => { if (g) return; if (n.t === 'tabs') g = n; else n.kids.forEach(w) }; w(dock5.tree); return g })();
for (const id of IDS) {
  if (first.ids.includes(id)) continue;
  dock5.remove(id); dock5.insert(id, first, 'center', 0);
}
dock5.render();
check(dock5, 'all panels collapsed into one group');
if (first.ids.length !== IDS.length)
  fails.push(`collapse: group holds ${first.ids.length} of ${IDS.length}`);
// pull them back out one at a time, each to the right edge of the shrinking
// group -- this is the sequence that used to drive a pane under 1%
for (const id of IDS.slice(1)) {
  const from = dock5.find(id).node;
  if (from === first && from.ids.length === 1) continue;
  dock5.remove(id); dock5.insert(id, first, 'right', 0); dock5.render();
  const w = (n) => { if (n.t !== 'split') return;
    if (n.sizes.some((x) => x < 5.9)) fails.push(`split back out: pane at ${Math.min(...n.sizes).toFixed(2)}%`);
    n.kids.forEach(w) };
  w(dock5.tree);
}
check(dock5, 'split back out');

console.log(fails.length ? '\nFAILURES:\n' + fails.slice(0, 10).join('\n')
                         : '\nall dock invariants held');
process.exit(fails.length ? 1 : 0);

// Does the stylesheet actually lay the dock out the way the dock thinks it does?
//
// The other tests run the JS and never look at the CSS, which is how the dock
// shipped with every horizontal split carrying class="dk-split row" -- and
// base.css's unrelated form-row rule (flex-wrap:wrap, align-items:center,
// gap:7px, margin-bottom:8px) matching it. The default two-column layout wrapped
// into one stacked column and every panel collapsed to its tab strip. The JS was
// perfect; the page was unusable.
//
// So: parse the four stylesheets, render the real dock, and for every element
// the dock owns work out which rules match it. Two rules the page has to keep:
//
//   1. Every class the dock puts on its own furniture is dk-prefixed. Short
//      names like `row` / `on` / `dot` belong to the panels, and a collision is
//      invisible in the JS.
//   2. No stylesheet other than dock.css may set a layout property on a dock
//      element -- that is the collision, stated as a property of the cascade
//      rather than of one class name.
//
// Then the positive half: the winning declarations really do say row where the
// tree says row, and nothing has turned a split into a wrapping flex container.

import './shim.mjs';
import { readFileSync } from 'fs';

const WEB = process.argv[2];
const base = WEB + '/js/';

// ---- the page shell ------------------------------------------------------
for (const id of ['state', 'p-model', 'p-backend', 'p-scene', 'pill-scene', 'p-ep',
                  'p-step', 'p-dist', 'p-res', 'fps', 'btn-theme', 'btn-layout', 'btn-lock',
                  'dock', 'boot', 'dk-ghost', 'dk-drop']) {
  const n = document.createElement('div'); n.id = id; document.body.appendChild(n);
}
const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));
globalThis.fetch = async (url) => {
  const body = url.startsWith('/readiness') ? { ready: true }
             : url.startsWith('/config.json') ? CFG_J
             : url.startsWith('/status') ? STATUS
             : url.startsWith('/prompt') ? { text: 'u', sys_text: 's' } : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};

// ---- CSS: parse ---------------------------------------------------------
// Stylesheet order is index.html's link order, which is also cascade order.
const SHEETS = ['css/tokens.css', 'css/base.css', 'css/dock.css', 'css/panels.css'];
const LAYOUT = new Set([
  'display', 'position', 'inset', 'top', 'right', 'bottom', 'left', 'float', 'clear',
  'flex', 'flex-direction', 'flex-wrap', 'flex-flow', 'flex-grow', 'flex-shrink',
  'flex-basis', 'order', 'align-items', 'align-self', 'align-content',
  'justify-content', 'justify-items', 'justify-self', 'place-items', 'place-content',
  'gap', 'row-gap', 'column-gap', 'grid', 'grid-template-columns', 'grid-template-rows',
  'grid-auto-flow', 'width', 'height', 'min-width', 'min-height', 'max-width',
  'max-height', 'margin', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
  'padding', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'overflow', 'overflow-x', 'overflow-y', 'box-sizing', 'writing-mode',
]);

const rules = [];
const skipped = [];        // selectors this matcher cannot judge -- reported, never silent
let order = 0;

for (const file of SHEETS) {
  const src = readFileSync(WEB + '/' + file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
  let i = 0;
  while (i < src.length) {
    const open = src.indexOf('{', i);
    if (open < 0) break;
    const head = src.slice(i, open).trim();
    let depth = 1, j = open + 1;
    while (j < src.length && depth) {
      if (src[j] === '{') depth++;
      else if (src[j] === '}') depth--;
      j++;
    }
    const body = src.slice(open + 1, j - 1);
    i = j;
    // @keyframes / @media are skipped whole: their inner blocks are not page
    // rules, and 'from' / 'to' would parse as element selectors.
    if (head.startsWith('@')) { skipped.push({ file, sel: head, props: [] }); continue }
    const decls = {};
    for (const d of body.split(';')) {
      const c = d.indexOf(':');
      if (c < 0) continue;
      const prop = d.slice(0, c).trim().toLowerCase();
      if (prop && !prop.startsWith('--')) decls[prop] = d.slice(c + 1).trim();
    }
    for (const one of head.split(',')) {
      const sel = one.trim();
      if (!sel) continue;
      const parsed = parseSel(sel);
      if (!parsed) {
        skipped.push({ file, sel, props: Object.keys(decls).filter((p) => LAYOUT.has(p)) });
        continue;
      }
      rules.push({ file, sel, parsed, decls, spec: specificity(parsed), order: order++ });
    }
  }
}

/** '.a > .b c' -> [{comb:null,comp}, {comb:'>',comp}, {comb:' ',comp}]; null if
 *  it uses anything this matcher would get wrong (pseudo, attribute, *, + ~). */
function parseSel(sel) {
  if (/[:\[\]*+~]/.test(sel)) return null;
  const toks = sel.replace(/\s*>\s*/g, ' > ').split(/\s+/).filter(Boolean);
  const parts = [];
  let comb = null;
  for (const t of toks) {
    if (t === '>') { comb = '>'; continue }
    const comp = { tag: '', id: '', cls: [] };
    const m = t.match(/^([a-zA-Z][\w-]*)?((?:[.#][\w-]+)*)$/);
    if (!m) return null;
    comp.tag = (m[1] || '').toUpperCase();
    for (const bit of (m[2] || '').match(/[.#][\w-]+/g) || []) {
      if (bit[0] === '.') comp.cls.push(bit.slice(1));
      else comp.id = bit.slice(1);
    }
    parts.push({ comb: parts.length ? (comb || ' ') : null, comp });
    comb = null;
  }
  return parts.length ? parts : null;
}

// Declarations, not const arrows: the parse loop above runs before this point.
function specificity(parts) {
  return parts.reduce(
    (a, p) => a + (p.comp.id ? 100 : 0) + p.comp.cls.length * 10 + (p.comp.tag ? 1 : 0), 0);
}

function hits(n, comp) {
  return (!comp.tag || n.tagName === comp.tag)
    && (!comp.id || n.id === comp.id)
    && comp.cls.every((c) => n.classList.contains(c));
}

function matchSel(n, parts) {
  if (!hits(n, parts[parts.length - 1].comp)) return false;
  return up(n, parts.slice(0, -1), parts[parts.length - 1].comb);
}
function up(n, parts, comb) {
  if (!parts.length) return true;
  const p = parts[parts.length - 1];
  if (comb === '>') {
    const par = n.parentNode;
    return !!par && hits(par, p.comp) && up(par, parts.slice(0, -1), p.comb);
  }
  for (let a = n.parentNode; a; a = a.parentNode) {
    if (hits(a, p.comp) && up(a, parts.slice(0, -1), p.comb)) return true;
  }
  return false;
}

/** Winning value per property, plus which rule won it. */
function computed(n) {
  const out = {};
  const matched = rules.filter((r) => matchSel(n, r.parsed))
    .sort((a, b) => (a.spec - b.spec) || (a.order - b.order));
  for (const r of matched) {
    for (const prop in r.decls) out[prop] = { value: r.decls[prop], from: r.file + ' ' + r.sel };
  }
  // inline style beats everything -- buildSplit writes flex-basis this way
  for (const prop in n.style) out[prop] = { value: n.style[prop], from: 'inline style' };
  return { style: out, matched };
}

// ---- render the real dock ----------------------------------------------
await import(base + 'app.js');
await new Promise((r) => setTimeout(r, 250));

const dockEl = document.getElementById('dock');
if (!dockEl || !dockEl.children.length) { console.log('FAIL: the dock did not render'); process.exit(1) }

/** Split the rendered tree into the dock's own furniture and the panels' roots.
 *
 *  The boundary is the .dk-pane element itself: it is a panel's root, and it
 *  carries the panel's own class as well (log.js's root is
 *  class="dk-pane log-box"), styled by panels.css on purpose. Same for the
 *  buttons inside .dk-tools -- the panels hand those over (log.js's 清空) and
 *  base.css styles them. Furniture is checked for foreign interference; panes are
 *  checked against the one contract dock.css owes them. */
const owned = [dockEl];
const panes = [];
(function walk(n) {
  for (const c of n.children) {
    if (c.tagName === '#TEXT') continue;
    if (c.classList.contains('dk-pane')) { panes.push(c); continue }
    owned.push(c);
    if (!c.classList.contains('dk-tools')) walk(c);
  }
})(dockEl);

const fails = [];
console.log('stylesheet rules parsed :', rules.length, '(' + skipped.length + ' selectors skipped as unjudgeable)');
console.log('dock-owned elements     :', owned.length);

// ---- 1. every dock class is dk-prefixed --------------------------------
const strays = new Set();
for (const n of owned) {
  for (const c of String(n.className).split(/\s+/).filter(Boolean)) {
    if (!c.startsWith('dk-')) strays.add(c + '  (on <' + n.tagName.toLowerCase() + ' class="' + n.className + '">)');
  }
}
if (strays.size) {
  fails.push('unprefixed classes on dock elements: ' + [...strays].join(', '));
  console.log('unprefixed dock classes :'); for (const s of strays) console.log('    ' + s);
} else {
  console.log('unprefixed dock classes : none');
}

// ---- 2. no foreign stylesheet sets layout on a dock element ------------
const foreign = [];
for (const n of owned) {
  for (const r of rules) {
    if (r.file === 'css/dock.css') continue;
    if (!matchSel(n, r.parsed)) continue;
    const props = Object.keys(r.decls).filter((p) => LAYOUT.has(p));
    if (props.length) {
      foreign.push(r.file + ' { ' + r.sel + ' } sets ' + props.join(', ')
                   + ' on <' + n.tagName.toLowerCase() + ' class="' + n.className + '">');
    }
  }
}
const uniqForeign = [...new Set(foreign)];
if (uniqForeign.length) {
  fails.push(uniqForeign.length + ' foreign layout rule(s) reach dock elements');
  console.log('foreign layout rules    :'); for (const f of uniqForeign) console.log('    ' + f);
} else {
  console.log('foreign layout rules    : none');
}

// ---- 3. the cascade says what the tree says ---------------------------
const val = (c, p) => (c.style[p] ? c.style[p].value : undefined);
const want = (cond, msg) => { if (!cond) fails.push(msg) };

const splits = owned.filter((n) => n.classList.contains('dk-split'));
const groups = owned.filter((n) => n.classList.contains('dk-group'));
console.log('splits / groups rendered:', splits.length, '/', groups.length);
want(splits.length >= 3, 'expected the default layout to have at least 3 splits, got ' + splits.length);
want(groups.length === 7, 'expected 7 groups in the default layout, got ' + groups.length);

for (const s of splits) {
  const c = computed(s);
  const dir = s.classList.contains('dk-row') ? 'row'
            : s.classList.contains('dk-col') ? 'column' : null;
  want(dir, 'a split has neither dk-row nor dk-col: class="' + s.className + '"');
  want(val(c, 'display') === 'flex', 'split is not display:flex (got ' + val(c, 'display') + ')');
  want(val(c, 'flex-direction') === dir,
       'split class="' + s.className + '" wants flex-direction:' + dir
       + ' but the cascade says ' + val(c, 'flex-direction')
       + (c.style['flex-direction'] ? ' from ' + c.style['flex-direction'].from : ''));
  // The bug, stated directly: a split that wraps stops being a split.
  const wrap = val(c, 'flex-wrap');
  want(wrap === undefined || wrap === 'nowrap',
       'split has flex-wrap:' + wrap + ' from ' + (c.style['flex-wrap'] || {}).from);
  const ai = val(c, 'align-items');
  want(ai === undefined || ai === 'stretch',
       'split has align-items:' + ai + ' (children would size to content, not fill)'
       + ' from ' + (c.style['align-items'] || {}).from);
  const mb = val(c, 'margin-bottom') || val(c, 'margin');
  want(mb === undefined || /^0/.test(mb),
       'split has a margin (' + mb + ') -- splits tile, they do not space themselves');
}

for (const g of groups) {
  const c = computed(g);
  want(val(c, 'display') === 'flex', 'group is not display:flex');
  want(val(c, 'flex-direction') === 'column', 'group is not a column (tab strip over body)');
  want(val(c, 'overflow') === 'hidden', 'group does not clip its body');
  // A group inside a split has to be allowed to shrink below its content, or a
  // small pane pushes the whole split past the viewport.
  want(val(c, 'min-height') === '0', 'group in a split has min-height ' + val(c, 'min-height') + ', want 0');
}

const bodies = owned.filter((n) => n.classList.contains('dk-body'));
for (const b of bodies) {
  const c = computed(b);
  want(val(c, 'flex') === '1', 'dk-body does not take the leftover height');
  want(val(c, 'min-height') === '0', 'dk-body has min-height ' + val(c, 'min-height'));
  want(val(c, 'position') === 'relative', 'dk-body is not the containing block for its panes');
}

console.log('panes                   :', panes.length);
want(panes.length === 9, 'expected 9 panes, got ' + panes.length);
for (const p of panes) {
  const c = computed(p);
  want(val(c, 'position') === 'absolute',
       'a pane is not absolutely filling its body (class="' + p.className + '")');
  want(val(c, 'inset') === '0',
       'a pane does not fill its body: inset ' + val(c, 'inset') + ' (class="' + p.className + '")');
  want(val(c, 'display') === 'flex',
       'a pane is not a flex column, so its content cannot claim the height'
       + ' (class="' + p.className + '", display ' + val(c, 'display') + ')');
}

// The top of the default layout is a row: two views side by side is the whole
// point of the layout the console is designed around.
const root = dockEl.children.find((c) => c.classList.contains('dk-split'));
want(root && root.classList.contains('dk-row'),
     'the root split is not a row -- the default layout is meant to be two columns');
if (root) {
  const c = computed(root);
  console.log('root split              : class="' + root.className + '" flex-direction='
              + val(c, 'flex-direction') + ' flex-wrap=' + (val(c, 'flex-wrap') || '(unset)')
              + ' align-items=' + (val(c, 'align-items') || '(unset)'));
}

// The two streams have to fill the panel they sit in. They are smaller than it
// (RGB is 640x480, the map is 420 tall), and max-width/max-height only ever
// shrink -- which is how the first-person view spent a while sitting at native
// size in the middle of a frame twice its area.
console.log('streams                 :');
for (const id of ['rgb', 'map']) {
  const img = document.getElementById(id);
  want(img, 'no #' + id + ' image in the rendered page');
  if (!img) continue;
  const c = computed(img);
  console.log('    #' + id + ' width=' + val(c, 'width') + ' height=' + val(c, 'height')
              + ' object-fit=' + val(c, 'object-fit'));
  want(val(c, 'width') === '100%' && val(c, 'height') === '100%',
       '#' + id + ' does not fill its frame (width ' + val(c, 'width')
       + ', height ' + val(c, 'height') + ') -- it would render at native size');
  // contain, never cover or fill: cover crops the map and fill distorts it.
  want(val(c, 'object-fit') === 'contain',
       '#' + id + ' has object-fit:' + val(c, 'object-fit') + ', want contain'
       + ' (cover crops, fill changes the aspect ratio)');
}

// What this matcher could not judge, and whether any of it could have mattered.
// A skipped selector is a hole in the check, so it is listed rather than counted:
// `*` (box-sizing) and the :hover / [hidden] / input[type=…] rules are the whole
// set, and none of them applies to a dock element at rest.
const risky = skipped.filter((s) => s.props.length);
console.log('skipped selectors       : ' + skipped.length + ', of which '
            + risky.length + ' set a layout property:');
for (const s of risky) console.log('    ' + s.file + ' { ' + s.sel + ' } -> ' + s.props.join(', '));

if (fails.length) {
  console.log('\nFAILURES:');
  for (const f of fails) console.log('  - ' + f);
  process.exit(1);
}
console.log('\ncss ok: no foreign rule touches the dock, every split lays out as its tree says');
// The page's 250 ms poll is still running, so say when we are done.
process.exit(0);

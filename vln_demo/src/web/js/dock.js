// A VSCode-shaped dock: nested splits, tab groups, draggable sashes, and
// drag-and-drop that either docks a panel against an edge or tabs it into a
// group.
//
// The tree is the only state:
//
//   split = {t:'split', dir:'row'|'col', sizes:[pct,...], kids:[node,...]}
//   group = {t:'tabs',  ids:['view','map'], active:0}
//
// Two rules the rest of the file follows from:
//
//  * Panel elements are created once, by their own module, and only ever
//    re-parented -- never rebuilt. The first-person view is an open MJPEG
//    stream and the log has a scroll position; both would be lost.
//  * A sash drag writes flex-basis directly and does not re-render. Rebuilding
//    the DOM 60 times a second while dragging would tear the streams.

import { el, clamp, ls } from './util.js';

const KEY = 'vln.layout.v1';
const MIN_PCT = 6;          // a panel can be squeezed to 6% of its split, no less
const EDGE = 0.28;          // within this fraction of an edge, dock instead of tab
const DRAG_SLOP = 5;        // px before a tab press becomes a drag rather than a click

let uid = 0;
function tag(n) {
  n._id = 'n' + (++uid);
  if (n.t === 'split') n.kids.forEach(tag);
  return n;
}

export class Dock {
  /** @param panels [{id, title, el, tools?}] -- element and tools built by the panel */
  constructor(root, panels, defaultTree) {
    this.root = root;
    this.def = defaultTree;
    this.reg = {};
    for (const p of panels) this.reg[p.id] = p;
    this.maxed = null;
    this.drag = null;
    this.ghost = document.getElementById('dk-ghost');
    this.drop = document.getElementById('dk-drop');
    this.tree = this.normalize(this.saved()) || tag(JSON.parse(JSON.stringify(this.def)));
  }

  // ---- persistence -------------------------------------------------------
  saved() {
    const raw = ls.get(KEY, '');
    if (!raw) return null;
    try { return JSON.parse(raw) } catch (e) { return null }
  }

  save() {
    // _el / _id are runtime handles, so serialise a clean copy
    const strip = (n) => n.t === 'split'
      ? { t: 'split', dir: n.dir, sizes: n.sizes.map((s) => +s.toFixed(2)), kids: n.kids.map(strip) }
      : { t: 'tabs', ids: n.ids.slice(), active: n.active };
    ls.set(KEY, JSON.stringify(strip(this.tree)));
  }

  reset() {
    ls.del(KEY);
    this.maxed = null;
    this.tree = tag(JSON.parse(JSON.stringify(this.def)));
    this.render();
  }

  /** Make a stored tree safe to render: drop unknown panels, dedupe, fix sizes.
   *
   * A saved layout outlives the code that wrote it. Panels get renamed and
   * added, and a tree that references a panel twice would re-parent the same
   * element into two places. Rather than version-gate the whole thing, repair
   * it: unknown ids are dropped, duplicates keep their first appearance, and a
   * panel the saved layout never heard of is appended in a group of its own so
   * a new feature is visible instead of silently missing. */
  normalize(t) {
    if (!t) return null;
    const known = new Set(Object.keys(this.reg));
    const seen = new Set();
    const walk = (n) => {
      if (!n || typeof n !== 'object') return null;
      if (n.t === 'tabs') {
        const ids = [];
        for (const i of n.ids || []) {
          // seen has to be updated inside the loop, not after it: a group that
          // lists the same panel twice would otherwise keep both copies and the
          // element would be re-parented into two places at once.
          if (known.has(i) && !seen.has(i)) { seen.add(i); ids.push(i) }
        }
        if (!ids.length) return null;
        return { t: 'tabs', ids, active: clamp(n.active | 0, 0, ids.length - 1) };
      }
      if (n.t !== 'split') return null;
      const kids = (n.kids || []).map(walk).filter(Boolean);
      if (!kids.length) return null;
      if (kids.length === 1) return kids[0];
      let sizes = (n.sizes || []).slice(0, kids.length).map((s) => (+s > 0 ? +s : 1));
      while (sizes.length < kids.length) sizes.push(100 / kids.length);
      const sum = sizes.reduce((a, b) => a + b, 0);
      sizes = sizes.map((s) => (s * 100) / sum);
      return { t: 'split', dir: n.dir === 'col' ? 'col' : 'row', sizes, kids };
    };
    let out = walk(t);
    if (!out) return null;
    const missing = [...known].filter((i) => !seen.has(i));
    // One-time migration for layouts saved before the terminal panel existed:
    // it belongs next to 导航指令, not in the generic "new panels" group that
    // normalize appends at the far right. Once a layout contains terminal its
    // saved location wins like every other panel.
    const termAt = missing.indexOf('terminal');
    if (termAt >= 0) {
      const besideInstruction = (n) => {
        if (n.t === 'tabs') {
          if (!n.ids.includes('instr')) return false;
          n.ids.push('terminal');
          return true;
        }
        return n.kids.some(besideInstruction);
      };
      if (besideInstruction(out)) missing.splice(termAt, 1);
    }
    if (missing.length) {
      out = { t: 'split', dir: 'row', sizes: [76, 24],
              kids: [out, { t: 'tabs', ids: missing, active: 0 }] };
    }
    return tag(out);
  }

  // ---- tree lookup -------------------------------------------------------
  /** Which group holds a panel: {node, parent, idx, i}. */
  find(id) {
    let out = null;
    const walk = (n, parent, idx) => {
      if (n.t === 'tabs') {
        const i = n.ids.indexOf(id);
        if (i >= 0) out = { node: n, parent, idx, i };
        return;
      }
      n.kids.forEach((k, j) => walk(k, n, j));
    };
    walk(this.tree, null, 0);
    return out;
  }

  /** Where a node sits: {parent, idx}; parent is null for the root. */
  findNode(node) {
    let out = { parent: null, idx: 0 };
    const walk = (n, parent, idx) => {
      if (n === node) { out = { parent, idx }; return true }
      return n.t === 'split' && n.kids.some((k, j) => walk(k, n, j));
    };
    walk(this.tree, null, 0);
    return out;
  }

  // ---- render ------------------------------------------------------------
  render() {
    this.root.replaceChildren(this.build(this.tree));
    this.applyMax();
    this.save();
  }

  build(node) {
    return node.t === 'split' ? this.buildSplit(node) : this.buildGroup(node);
  }

  buildSplit(node) {
    // 'dk-row' / 'dk-col', not 'row' / 'col': every class the dock puts on its
    // own furniture is dk-prefixed, because base.css owns the short names. A
    // split used to come out as class="dk-split row", which also matched
    // base.css's form-row rule -- flex-wrap:wrap, align-items:center, gap:7px --
    // so the two-column default layout wrapped into a single stacked column and
    // every panel collapsed to its tab strip. See test/css.mjs.
    const d = el('div.dk-split.dk-' + node.dir);
    d.dataset.nid = node._id;
    node._el = d;
    node.kids.forEach((k, i) => {
      if (i) d.appendChild(this.sash(node, i - 1));
      const c = this.build(k);
      c.style.flex = '1 1 ' + node.sizes[i] + '%';
      d.appendChild(c);
    });
    return d;
  }

  buildGroup(node) {
    const g = el('div.dk-group');
    g.dataset.nid = node._id;
    node._el = g;
    const strip = el('div.dk-tabs');
    node.ids.forEach((id, i) => {
      const p = this.reg[id];
      const t = el('div.dk-tab' + (i === node.active ? '.dk-on' : ''),
                   { title: p.title + '　（拖动可以移到别处，双击最大化）' },
                   [el('span.dk-dot'), el('span', { text: p.title })]);
      t.addEventListener('pointerdown', (e) => this.tabDown(e, node, id, t));
      t.addEventListener('click', () => {
        if (this.moved) return;                  // that press was a drag
        if (node.active !== i) { node.active = i; this.render() }
      });
      strip.appendChild(t);
    });
    // Whatever the active panel wants in the strip, and nothing else. There is no
    // maximise button: it sat on all seven tab strips to offer something a
    // double-click on the tab already does.
    const active = this.reg[node.ids[node.active]];
    const own = (active && active.tools) || [];
    if (own.length) {
      const tools = el('div.dk-tools');
      for (const b of own) tools.appendChild(b);
      strip.appendChild(tools);
    }
    strip.addEventListener('dblclick', (e) => {
      if (e.target.closest('.dk-tab')) this.toggleMax(node);
    });

    const body = el('div.dk-body');
    node.ids.forEach((id, i) => {
      const p = this.reg[id];
      p.el.hidden = i !== node.active;
      body.appendChild(p.el);                    // a move, not a copy
    });
    g.append(strip, body);
    return g;
  }

  toggleMax(node) {
    this.maxed = this.maxed === node._id ? null : node._id;
    this.render();
  }

  applyMax() {
    for (const e of this.root.querySelectorAll('.dk-hasmax, .dk-maxpath')) {
      e.classList.remove('dk-hasmax', 'dk-maxpath');
    }
    if (!this.maxed) return;
    let cur = this.root.querySelector('[data-nid="' + this.maxed + '"]');
    if (!cur) { this.maxed = null; return }
    while (cur.parentElement && cur.parentElement.classList.contains('dk-split')) {
      cur.classList.add('dk-maxpath');
      cur.parentElement.classList.add('dk-hasmax');
      cur = cur.parentElement;
    }
  }

  // ---- sash --------------------------------------------------------------
  sash(node, i) {
    const s = el('div.dk-sash');
    s.addEventListener('pointerdown', (e) => {
      if (e.button) return;
      e.preventDefault();
      const row = node.dir === 'row';
      const box = node._el.getBoundingClientRect();
      const total = (row ? box.width : box.height) || 1;
      const a = node.sizes[i], b = node.sizes[i + 1];
      const start = row ? e.clientX : e.clientY;
      const ea = node.kids[i]._el, eb = node.kids[i + 1]._el;
      s.classList.add('dk-drag');
      document.body.classList.add('dk-sizing');
      const move = (ev) => {
        const d = (((row ? ev.clientX : ev.clientY) - start) / total) * 100;
        const na = clamp(a + d, MIN_PCT, a + b - MIN_PCT);
        node.sizes[i] = na;
        node.sizes[i + 1] = a + b - na;
        ea.style.flex = '1 1 ' + node.sizes[i] + '%';
        eb.style.flex = '1 1 ' + node.sizes[i + 1] + '%';
      };
      const up = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        s.classList.remove('dk-drag');
        document.body.classList.remove('dk-sizing');
        this.save();
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
    return s;
  }

  // ---- drag a panel ------------------------------------------------------
  tabDown(e, group, id, tabEl) {
    if (e.button) return;
    const x0 = e.clientX, y0 = e.clientY;
    this.moved = false;
    const move = (ev) => {
      if (!this.drag) {
        if (Math.hypot(ev.clientX - x0, ev.clientY - y0) < DRAG_SLOP) return;
        this.moved = true;
        this.begin(id, group, tabEl);
      }
      this.over(ev);
    };
    const up = (ev) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      if (this.drag) this.finish(ev);
      // let the click handler that follows know it was a drag, then forget
      setTimeout(() => { this.moved = false }, 0);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  begin(id, group, tabEl) {
    this.drag = { id, from: group, tabEl, hit: null };
    tabEl.classList.add('dk-drag');
    document.body.classList.add('dk-dragging');
    this.ghost.textContent = this.reg[id].title;
    this.ghost.hidden = false;
  }

  over(ev) {
    this.ghost.style.left = ev.clientX + 'px';
    this.ghost.style.top = ev.clientY - 22 + 'px';
    const hit = this.hitTest(ev.clientX, ev.clientY);
    this.drag.hit = hit;
    for (const g of this.root.querySelectorAll('.dk-group.dk-over')) {
      g.classList.remove('dk-over');
    }
    if (!hit) { this.drop.hidden = true; return }
    hit.groupEl.classList.add('dk-over');
    const r = hit.groupEl.getBoundingClientRect();
    let box = { left: r.left, top: r.top, width: r.width, height: r.height };
    if (hit.zone === 'left') box.width = r.width / 2;
    else if (hit.zone === 'right') { box.left = r.left + r.width / 2; box.width = r.width / 2 }
    else if (hit.zone === 'top') box.height = r.height / 2;
    else if (hit.zone === 'bottom') { box.top = r.top + r.height / 2; box.height = r.height / 2 }
    else if (hit.zone === 'tab') {
      // Over the tab strip the preview shrinks to an insertion caret: the panel
      // is going to become a tab here, and where in the row is the only choice
      // left to show.
      box = { left: hit.caret - 1.5, top: hit.strip.top + 4,
              width: 3, height: hit.strip.height - 7 };
    }
    Object.assign(this.drop.style, {
      left: box.left + 'px', top: box.top + 'px',
      width: box.width + 'px', height: box.height + 'px',
    });
    this.drop.hidden = false;
  }

  /** Which group is under the cursor, and where in it. */
  hitTest(x, y) {
    const under = document.elementFromPoint(x, y);
    const groupEl = under && under.closest ? under.closest('.dk-group') : null;
    if (!groupEl) return null;
    const node = this.byId(groupEl.dataset.nid);
    if (!node) return null;
    const stripEl = groupEl.querySelector('.dk-tabs');
    const strip = stripEl.getBoundingClientRect();
    if (y <= strip.bottom) {
      // Insert before the first tab whose midpoint is past the cursor, which is
      // also how the panel gets reordered inside its own group.
      const tabs = [...stripEl.querySelectorAll('.dk-tab')].map((t) => t.getBoundingClientRect());
      let at = tabs.findIndex((t) => x < t.left + t.width / 2);
      if (at < 0) at = tabs.length;
      const caret = at < tabs.length ? tabs[at].left
                  : tabs.length ? tabs[tabs.length - 1].right : strip.left + 4;
      return { node, groupEl, zone: 'tab', at, caret, strip };
    }
    const r = groupEl.getBoundingClientRect();
    const fx = (x - r.left) / (r.width || 1);
    const fy = (y - r.top) / (r.height || 1);
    const d = { left: fx, right: 1 - fx, top: fy, bottom: 1 - fy };
    const near = Object.keys(d).sort((a, b) => d[a] - d[b])[0];
    return { node, groupEl, zone: d[near] < EDGE ? near : 'center' };
  }

  byId(nid) {
    let out = null;
    const walk = (n) => {
      if (n._id === nid) { out = n; return }
      if (n.t === 'split') n.kids.forEach(walk);
    };
    walk(this.tree);
    return out;
  }

  finish(ev) {
    const { id, from, tabEl, hit } = this.drag;
    this.drag = null;
    tabEl.classList.remove('dk-drag');
    document.body.classList.remove('dk-dragging');
    this.ghost.hidden = true;
    this.drop.hidden = true;
    for (const g of this.root.querySelectorAll('.dk-group.dk-over')) {
      g.classList.remove('dk-over');
    }
    if (!hit) { this.render(); return }
    // Dropping a panel back where it already is: nothing to do, and going
    // through remove/insert would tear down a group that is about to be rebuilt
    // identically (and drop the tree entirely if it was the last panel).
    const alone = from.ids.length === 1;
    if (hit.node === from && (hit.zone === 'center' || alone)) { this.render(); return }
    this.remove(id);
    this.insert(id, hit.node, hit.zone, hit.at);
    this.maxed = null;
    this.render();
  }

  // ---- tree mutation -----------------------------------------------------
  remove(id) {
    const f = this.find(id);
    if (!f) return;
    f.node.ids.splice(f.i, 1);
    f.node.active = clamp(f.node.active, 0, Math.max(0, f.node.ids.length - 1));
    if (f.node.ids.length) return;
    // The group is empty: take it out and hand its space to a neighbour, then
    // collapse a split that is down to one child.
    const { parent, idx } = this.findNode(f.node);
    if (!parent) return;
    parent.kids.splice(idx, 1);
    const freed = parent.sizes.splice(idx, 1)[0];
    if (parent.sizes.length) {
      parent.sizes[Math.min(idx, parent.sizes.length - 1)] += freed;
    }
    if (parent.kids.length === 1) {
      const only = parent.kids[0];
      const up = this.findNode(parent);
      if (up.parent) up.parent.kids[up.idx] = only;
      else this.tree = only;
    }
  }

  insert(id, target, zone, at) {
    if (zone === 'center' || zone === 'tab') {
      // `at` was measured before remove() shifted the tabs along, so clamp.
      const i = zone === 'tab' ? clamp(at | 0, 0, target.ids.length) : target.ids.length;
      target.ids.splice(i, 0, id);
      target.active = i;
      return;
    }
    const dir = zone === 'left' || zone === 'right' ? 'row' : 'col';
    const before = zone === 'left' || zone === 'top';
    const fresh = tag({ t: 'tabs', ids: [id], active: 0 });
    const { parent, idx } = this.findNode(target);
    if (parent && parent.dir === dir) {
      // Same direction as the split it lives in: become a sibling rather than
      // nesting another split, so the tree stays flat and the sashes line up.
      const at = before ? idx : idx + 1;
      const half = parent.sizes[idx] / 2;
      parent.kids.splice(at, 0, fresh);
      if (half >= MIN_PCT) {
        parent.sizes[idx] = half;
        parent.sizes.splice(at, 0, half);
        return;
      }
      // Halving would leave a pane too small to grab by its tab strip -- which
      // is how a panel becomes unreachable, since dragging it out again needs
      // that strip. Dropping into an already-thin slot rebalances the whole
      // split instead. A sash drag has the same floor; insert should not be the
      // one way round it.
      parent.sizes.splice(at, 0, half);
      const even = 100 / parent.sizes.length;
      for (let i = 0; i < parent.sizes.length; i++) parent.sizes[i] = even;
      return;
    }
    const wrap = tag({
      t: 'split', dir, sizes: before ? [40, 60] : [60, 40],
      kids: before ? [fresh, target] : [target, fresh],
    });
    if (parent) parent.kids[idx] = wrap;
    else this.tree = wrap;
  }

  /** Bring a panel to the front of whatever group it is in. */
  focus(id) {
    const f = this.find(id);
    if (!f || f.node.active === f.i) return;
    f.node.active = f.i;
    this.render();
  }
}

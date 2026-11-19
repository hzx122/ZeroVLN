// A DOM small enough to read and big enough to boot the console's page.
//
// The one behaviour that matters here: getElementById walks down from
// documentElement, so an element that has not been attached yet is NOT found --
// exactly like a browser. Shims that keep a flat id registry hide precisely the
// class of bug this is meant to catch.

class Node {
  constructor(tag) {
    this.tagName = (tag || '').toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this._attrs = {};
    // dataset writes through to data-* attributes, as in a browser: theme.js
    // sets dataset.theme and the stylesheet selects on [data-theme].
    this.dataset = new Proxy(this._attrs, {
      get: (t, k) => t['data-' + dashed(k)],
      set: (t, k, v) => { t['data-' + dashed(k)] = String(v); return true },
      deleteProperty: (t, k) => { delete t['data-' + dashed(k)]; return true },
      has: (t, k) => ('data-' + dashed(k)) in t,
    });
    this._cls = '';
    this._text = '';
    this._listeners = {};
    this.hidden = false;
    this.disabled = false;
    this._value = '';
    this.checked = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.title = '';
    this.id = '';
  }
  get className() { return this._cls }
  set className(v) { this._cls = String(v == null ? '' : v) }
  get classList() {
    const self = this;
    const list = () => self._cls.split(/\s+/).filter(Boolean);
    return {
      add: (...c) => { self._cls = [...new Set([...list(), ...c])].join(' ') },
      remove: (...c) => { self._cls = list().filter((x) => !c.includes(x)).join(' ') },
      toggle: (c, on) => { on ? self.classList.add(c) : self.classList.remove(c) },
      contains: (c) => list().includes(c),
    };
  }
  get textContent() {
    return this.children.length
      ? this.children.map((c) => c.textContent).join('') : this._text;
  }
  set textContent(v) { this.children = []; this._text = String(v == null ? '' : v) }
  get innerHTML() { return '' }
  set innerHTML(v) { this.children = []; this._text = '' }
  get isConnected() {
    let n = this;
    while (n.parentNode) n = n.parentNode;
    return n === document || n === document.documentElement || n._isRoot === true;
  }
  get offsetParent() { return this.isConnected && !this.hidden ? this.parentNode : null }
  get offsetLeft() { return 0 }
  get offsetTop() { return 0 }
  get options() { return this.children.filter((c) => c.tagName === 'OPTION') }
  get value() {
    if (this.tagName !== 'SELECT') return this._value;
    const opts = this.options;
    const sel = opts.find((o) => o.selected) || opts[0];
    return sel ? (sel._attrs.value != null ? sel._attrs.value : sel.textContent) : '';
  }
  set value(v) {
    this._value = String(v == null ? '' : v);
    if (this.tagName !== 'SELECT') return;
    // Assigning an unmatched value is a no-op in a browser, not a silent set.
    const opts = this.options;
    const hit = opts.find((o) =>
      (o._attrs.value != null ? o._attrs.value : o.textContent) === this._value);
    if (hit) for (const o of opts) o.selected = o === hit;
  }
  get firstChild() { return this.children[0] || null }
  get lastChild() { return this.children[this.children.length - 1] || null }
  get nextSibling() {
    const s = this.parentNode ? this.parentNode.children : [];
    return s[s.indexOf(this) + 1] || null;
  }
  appendChild(c) {
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this; this.children.push(c); return c;
  }
  append(...cs) { for (const c of cs) this.appendChild(typeof c === 'string' ? text(c) : c) }
  insertBefore(c, ref) {
    if (c.parentNode) c.parentNode.removeChild(c);
    const i = ref ? this.children.indexOf(ref) : this.children.length;
    c.parentNode = this;
    this.children.splice(i < 0 ? this.children.length : i, 0, c);
    return c;
  }
  removeChild(c) {
    const i = this.children.indexOf(c);
    if (i >= 0) { this.children.splice(i, 1); c.parentNode = null }
    return c;
  }
  replaceChildren(...cs) {
    for (const c of [...this.children]) this.removeChild(c);
    this.append(...cs);
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this) }
  setAttribute(k, v) {
    if (k === 'class') { this.className = v; return }
    this._attrs[k] = String(v);
    // A present boolean attribute means true, whatever its value string is --
    // el() writes them as setAttribute(k, ''), and reading them back as '' is
    // how a shim invents bugs that a browser does not have.
    if (BOOL_ATTRS.has(k)) { this[k] = true; return }
    if (!k.startsWith('data-')) this[k] = v;
  }
  getAttribute(k) {
    if (k === 'class') return this.className;
    return k in this._attrs ? this._attrs[k] : (k in this ? this[k] : null);
  }
  removeAttribute(k) { delete this._attrs[k]; delete this[k] }
  addEventListener(t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn) }
  removeEventListener(t, fn) {
    this._listeners[t] = (this._listeners[t] || []).filter((f) => f !== fn);
  }
  dispatchEvent(e) {
    e.target = e.target || this;
    for (const fn of this._listeners[e.type] || []) fn.call(this, e);
    return true;
  }
  fire(type, extra = {}) { return this.dispatchEvent({ type, target: this, ...extra }) }
  getBoundingClientRect() {
    return { left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600 };
  }
  focus() {}
  blur() {}
  select() {}
  closest(sel) {
    for (let n = this; n; n = n.parentNode) if (matches(n, sel)) return n;
    return null;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null }
  querySelectorAll(sel) {
    const out = [];
    const walk = (n) => {
      for (const c of n.children) { if (matches(c, sel)) out.push(c); walk(c) }
    };
    walk(this);
    return out;
  }
  get all() {
    const out = [];
    const walk = (n) => { for (const c of n.children) { out.push(c); walk(c) } };
    walk(this);
    return out;
  }
}

const BOOL_ATTRS = new Set(['checked', 'selected', 'disabled', 'hidden', 'multiple']);

const dashed = (k) => String(k).replace(/[A-Z]/g, (c) => '-' + c.toLowerCase());

function text(s) { const n = new Node('#text'); n._text = s; return n }

/** Supports exactly what the page uses: 'tag', '.cls', 'tag.cls', and lists. */
/** `tag`, `.cls`, `#id` and any combination of them, comma-separated. Anything
 *  else -- attribute, pseudo, combinator -- matches nothing, the same way it did
 *  before it was parsed at all: a selector this cannot judge must not silently
 *  come back true. */
function matches(n, sel) {
  return String(sel).split(',').some((one) => {
    const m = one.trim().match(/^([a-zA-Z][\w-]*)?((?:[.#][\w-]+)*)$/);
    if (!m || (!m[1] && !m[2])) return false;
    if (m[1] && n.tagName !== m[1].toUpperCase()) return false;
    for (const bit of (m[2] || '').match(/[.#][\w-]+/g) || []) {
      if (bit[0] === '.') { if (!n.classList.contains(bit.slice(1))) return false }
      else if (n.id !== bit.slice(1)) return false;
    }
    return true;
  });
}

const document = {
  _isRoot: true,
  createElement: (t) => new Node(t),
  createTextNode: text,
  createDocumentFragment: () => new Node('#fragment'),
  getElementById(id) {
    const hit = (n) => {
      for (const c of n.children) {
        if (c.id === id) return c;
        const d = hit(c);
        if (d) return d;
      }
      return null;
    };
    return hit(document.documentElement);
  },
  querySelector(s) { return document.documentElement.querySelector(s) },
  querySelectorAll(s) { return document.documentElement.querySelectorAll(s) },
  addEventListener() {},
  removeEventListener() {},
  get hidden() { return false },
};
document.documentElement = new Node('html');
document.documentElement._isRoot = true;
document.documentElement.parentNode = null;
document.body = new Node('body');
document.documentElement.appendChild(document.body);
document.head = new Node('head');
document.documentElement.appendChild(document.head);

const store = new Map();
const localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};

const window = {
  document, localStorage,
  addEventListener() {}, removeEventListener() {},
  requestAnimationFrame: (fn) => setTimeout(fn, 0),
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  location: { reload() {} },
  confirm: () => true,
  alert: () => {},
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
};

class CustomEvent {
  constructor(type, opts = {}) { this.type = type; this.detail = opts.detail }
}

globalThis.document = document;
globalThis.window = window;
globalThis.localStorage = localStorage;
globalThis.CustomEvent = CustomEvent;
globalThis.Element = Node;
// Node 26 exposes navigator through a getter-only global. defineProperty works
// there and on older Node releases, while direct assignment no longer does.
Object.defineProperty(globalThis, 'navigator', {
  value: { userAgent: 'shim' }, configurable: true, writable: true,
});

export { document, window, Node, matches };

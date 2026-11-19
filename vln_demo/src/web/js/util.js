// Tiny DOM helpers. Everything here is used by three or more modules; anything
// used by one lives in that module.

export const $ = (id) => document.getElementById(id);

/** getElementById restricted to one subtree -- and, crucially, working before
 *  that subtree is in the document.
 *
 *  Panels are built detached and only then hung in the dock, so a panel that
 *  wires itself up with document.getElementById finds nothing: the ids it is
 *  looking for are its own, and the document has never seen them. Anything a
 *  panel looks up inside itself goes through here instead.
 */
export function within(root, id) {
  if (!root) return null;
  if (root.id === id) return root;
  const kids = root.children || [];
  for (let i = 0; i < kids.length; i++) {
    const hit = within(kids[i], id);
    if (hit) return hit;
  }
  return null;
}

/** el('div.row', {title:'x'}, [child|string, ...]) */
export function el(spec, attrs, kids) {
  const [tag, ...cls] = String(spec).split('.');
  const n = document.createElement(tag || 'div');
  if (cls.length) n.className = cls.join(' ');
  for (const k in (attrs || {})) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, '');
    else n.setAttribute(k, v);
  }
  for (const c of [].concat(kids || [])) {
    if (c == null) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
}

export function esc(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Fixed decimals without the trailing-zero noise: 1.00 -> "1", 0.60 -> "0.6". */
export function trim(v, n) {
  return Number(v).toFixed(n).replace(/\.?0+$/, '') || '0';
}

export function labelRow(label, ...kids) {
  return el('div.row', {}, [el('label', { text: label }), ...kids]);
}

export function select(id, items, cur) {
  const s = el('select', { id });
  for (const it of items) {
    const o = el('option', { value: it.key, text: it.label });
    if (it.key === cur) o.selected = true;
    s.appendChild(o);
  }
  return s;
}

/** localStorage that never throws (private mode, quota, disabled storage). */
export const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v } catch (e) { return d } },
  set(k, v) { try { localStorage.setItem(k, v) } catch (e) { /* ignore */ } },
  del(k) { try { localStorage.removeItem(k) } catch (e) { /* ignore */ } },
};

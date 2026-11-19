// Every call to the console goes through here, so the URL shapes live in one
// place and a route rename is one edit.
//
// It is also where the 🔒 lock gets its second line of defence: every command
// the page can issue funnels through cmd(), so one check here covers the paths
// that are not a form control (the map's double-click-to-teleport, manual
// driving's 150 ms heartbeat) and would slip past lock.js's capture guard.
// lock/unlock themselves are exempt -- they are how the lock is undone.

import { blocked } from './lock.js';

function qs(o) {
  return Object.keys(o || {})
    .map((k) => encodeURIComponent(k) + '=' + encodeURIComponent(o[k]))
    .join('&');
}

export function cmd(action, extra) {
  if (action !== 'lock' && action !== 'unlock' && blocked('cmd ' + action)) {
    return Promise.resolve();
  }
  const q = qs(extra);
  return fetch('/cmd?do=' + encodeURIComponent(action) + (q ? '&' + q : ''))
    .catch(() => {});
}

export function getJSON(url, options) {
  return fetch(url, options).then((r) => r.json());
}

/** Atomic first-person snapshot used by the bbox picker. */
export function getFrameSnapshot() {
  return getJSON('/frame-snapshot?v=' + Date.now());
}

export function prompt(params) {
  return getJSON('/prompt?' + qs(params));
}

/** Empty body = restore the default; see prompts.agent_system_prompt(). */
export function postSysPrompt(text) {
  if (blocked('sysprompt')) {
    return Promise.resolve({ ok: false, locked: true });
  }
  return fetch('/sysprompt', {
    method: 'POST',
    body: text,
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  }).then((r) => r.json());
}

/** Optional first-turn visual reference for the next ReAct run. */
export function postReactReference(payload) {
  if (blocked('react-reference')) {
    return Promise.resolve({ ok: false, locked: true });
  }
  return fetch('/react-reference', {
    method: 'POST',
    body: JSON.stringify(payload || {}),
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  }).then((r) => r.json());
}

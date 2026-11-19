// The one status poll, fanned out to whoever asked.
//
// Panels do not fetch /status themselves: it carries the event log as an
// increment keyed by `since`, so a second reader would silently steal events
// from the first. One poll, one `since`, many subscribers.

export const CFG = {};        // /config.json, filled once by app.js
export const S = {};          // the latest /status snapshot

const subs = [];
let since = 0;
let timer = null;
let pollingRun = 0;
let resetEpoch = 0;

export function onStatus(fn) {
  subs.push(fn);
  if (S.status) fn(S);        // late subscribers still see the current state
}

/** Drop an in-flight snapshot after the user clears the log. */
export function resetSince() { since = 0; resetEpoch++ }

export function startPolling(ms) {
  const run = ++pollingRun;
  if (timer) clearTimeout(timer);
  const tick = async () => {
    const epoch = resetEpoch;
    try {
      const response = await fetch('/status?since=' + since);
      const s = await response.json();
      if (run !== pollingRun || epoch !== resetEpoch) return;
      // A restarted backend begins a new event sequence.
      if (s.since < since) s.reset_log = true;
      since = s.since;
      Object.assign(S, s);
      for (const fn of subs) {
        try { fn(s) } catch (e) { console.error(e) }
      }
    } catch (_) {
      // Retry after failure, with only one status request in flight.
    } finally {
      if (run === pollingRun) timer = setTimeout(tick, ms);
    }
  };
  tick();
}

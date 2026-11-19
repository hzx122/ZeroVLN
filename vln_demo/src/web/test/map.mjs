// Double-clicking the bird's-eye view has to land where you aimed.
//
// The map image fills its panel with object-fit: contain, so the element box is
// the whole frame and the picture is letterboxed inside it. The server wants a
// fraction of the *picture*; the browser hands us a point in the *element*. That
// arithmetic is pictureRect(), and it is invisible until someone clicks a map
// that is not the shape of its panel -- which is every map, since the server
// renders them at a fixed 420px height and the panel is whatever the window is.
//
// Back when the picture sat at its native size in the middle of the frame the
// two boxes were identical and the naive (e.clientX - r.left) / r.width was
// right. It is now wrong by exactly the letterbox, which is what this pins.

import './shim.mjs';
import { readFileSync } from 'fs';

const base = process.argv[2] + '/js/';
const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const sent = [];
globalThis.fetch = async (url) => {
  if (url.startsWith('/cmd')) sent.push(url);
  return { ok: true, status: 200, json: async () => ({}), text: async () => '{}' };
};

const store = await import(base + 'store.js');
if (store.CFG && !Object.keys(store.CFG).length) Object.assign(store.CFG, CFG_J);
const map = await import(base + 'panels/map.js');

const errs = [];
const step = (name, fn) => {
  try { fn(); console.log('  ok   ' + name) }
  catch (e) { errs.push(name + ': ' + e.message); console.log('  FAIL ' + name + ' -- ' + e.message) }
};
const near = (got, want, what) => {
  if (Math.abs(got - want) > 1e-4) {
    throw new Error(what + ': want ' + want + ', got ' + got.toFixed(5));
  }
};

// A 600x420 map (the server's fixed height, some scene's width) in an 800x600
// panel. contain scales by min(800/600, 600/420) = 1.3333, so the picture is
// 800x560 with 20px of letterbox above and below.
const FRAME = { left: 100, top: 50, width: 800, height: 600 };
const pane = map.build();
const img = pane.querySelector('#map');
if (!img) { console.log('FAIL: the map panel has no #map image'); process.exit(1) }
img.naturalWidth = 600;
img.naturalHeight = 420;
img.getBoundingClientRect = () => ({
  left: FRAME.left, top: FRAME.top, width: FRAME.width, height: FRAME.height,
  right: FRAME.left + FRAME.width, bottom: FRAME.top + FRAME.height,
});

console.log('pictureRect inside a letterboxed frame:');
step('the picture is as wide as the frame and 560 tall', () => {
  const r = map.pictureRect(img);
  near(r.width, 800, 'width');
  near(r.height, 560, 'height');
  near(r.left, 100, 'left');
  near(r.top, 70, 'top');            // 50 + (600 - 560) / 2
});
step('no frame yet -> fall back to the element box', () => {
  const bare = map.build().querySelector('#map');
  bare.getBoundingClientRect = img.getBoundingClientRect;
  const r = map.pictureRect(bare);   // naturalWidth is 0 before the first frame
  near(r.height, 600, 'height');
});

const dbl = (x, y) => { sent.length = 0; img.fire('dblclick', { clientX: x, clientY: y }); return sent[0] };
const frac = (url) => {
  const q = new URLSearchParams(url.slice(url.indexOf('?') + 1));
  return { x: +q.get('x'), y: +q.get('y') };
};

console.log('double-click -> fraction of the picture:');
step('the middle of the picture is 0.5, 0.5', () => {
  const f = frac(dbl(100 + 400, 70 + 280));
  near(f.x, 0.5, 'x'); near(f.y, 0.5, 'y');
});
step('the top-left corner of the picture is 0, 0', () => {
  const f = frac(dbl(100, 70));
  near(f.x, 0, 'x'); near(f.y, 0, 'y');
});
step('the bottom-right corner of the picture is 1, 1', () => {
  const f = frac(dbl(100 + 800, 70 + 560));
  near(f.x, 1, 'x'); near(f.y, 1, 'y');
});
// The one the old code got wrong: a quarter of the way down the *picture* is
// 0.25, not 0.2833 (which is where it is in the frame).
step('a quarter down the picture is 0.25, not a quarter down the frame', () => {
  const f = frac(dbl(100 + 400, 70 + 140));
  near(f.y, 0.25, 'y');
  const naive = (70 + 140 - FRAME.top) / FRAME.height;
  if (Math.abs(naive - 0.25) < 1e-3) throw new Error('this case cannot tell the two apart');
});

console.log('clicks that are not on the map:');
step('in the letterbox above the picture -> nothing is sent', () => {
  const url = dbl(100 + 400, 50 + 5);
  if (url) throw new Error('sent ' + url + ' for a click 15px above the picture');
});
step('in the letterbox below the picture -> nothing is sent', () => {
  const url = dbl(100 + 400, 50 + 595);
  if (url) throw new Error('sent ' + url + ' for a click below the picture');
});
step('a click on the last row of the picture still counts', () => {
  if (!dbl(100 + 400, 70 + 559)) throw new Error('rejected a click that is on the map');
});

// A frame the same shape as the picture has no letterbox, and the fractions have
// to come out identical to the pre-contain behaviour.
console.log('a frame that matches the picture exactly:');
step('no letterbox -> fraction is the fraction of the frame', () => {
  img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 600, height: 420,
                                       right: 600, bottom: 420 });
  const f = frac(dbl(150, 105));
  near(f.x, 0.25, 'x'); near(f.y, 0.25, 'y');
});

console.log();
if (errs.length) {
  console.log('FAILURES:');
  for (const e of errs) console.log('  - ' + e);
  process.exit(1);
}
console.log('map ok: a double-click is a fraction of the picture, letterbox and all');
process.exit(0);

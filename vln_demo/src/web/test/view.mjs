// Double-clicking the first-person image sends a normalized direct point.

import './shim.mjs';

const base = process.argv[2] + '/js/';
const sent = [];
globalThis.fetch = async (url) => {
  if (url.startsWith('/cmd')) sent.push(url);
  return { ok: true, status: 200, json: async () => ({}), text: async () => 'ok' };
};

const view = await import(base + 'panels/view.js');
const pane = view.build();
const img = pane.querySelector('#rgb');
if (!img) { console.log('FAIL: no #rgb image'); process.exit(1) }
img.naturalWidth = 640;
img.naturalHeight = 480;
img.getBoundingClientRect = () => ({
  left: 100, top: 50, width: 800, height: 600, right: 900, bottom: 650,
});

const errors = [];
function check(name, fn) {
  try { fn(); console.log('  ok   ' + name) }
  catch (error) { errors.push(name + ': ' + error.message) }
}
function click(x, y) {
  sent.length = 0;
  img.fire('dblclick', { clientX: x, clientY: y });
  return sent[0];
}
function params(url) {
  return new URLSearchParams(url.slice(url.indexOf('?') + 1));
}

check('centre becomes normalized point 500,500', () => {
  const query = params(click(500, 350));
  if (query.get('do') !== 'screen_point_move') throw new Error('wrong command');
  if (query.get('x') !== '500' || query.get('y') !== '500') {
    throw new Error('wrong normalized coordinates');
  }
});
check('corner becomes normalized point 0,0', () => {
  const query = params(click(100, 50));
  if (query.get('x') !== '0' || query.get('y') !== '0') {
    throw new Error('wrong corner coordinates');
  }
});
if (errors.length) {
  for (const error of errors) console.log('FAIL ' + error);
  process.exit(1);
}
console.log('view ok: double-click sends a normalized direct point');

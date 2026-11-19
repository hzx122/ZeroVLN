import './shim.mjs';

const base = process.argv[2] + '/js/';
const snapshot = {
  id: 7, turn: 2, driver: 'react', message_count: 1,
  total_message_count: 4, image_count: 1,
  messages: [
    { index: 4, role: 'user', parts: [
      { type: 'text', text: 'before image' },
      { type: 'image', url: '/debug-image/9.jpg', width: 640, height: 480 },
      { type: 'text', text: 'after image' },
    ] },
  ],
};

let fetches = 0;
globalThis.fetch = async (url) => {
  fetches++;
  return { ok: true, status: 200,
           json: async () => ({ ok: true, request: snapshot }) };
};

const store = await import(base + 'store.js');
Object.assign(store.S, {
  status: 'running', reset_log: false,
  stream: { active: true, text: 'STREAMING OUTPUT', n: 2, dt: 0.4 },
  events: [{ t: 'request', i: 1, request_id: 7, turn: 2, driver: 'react',
             message_count: 1, total_message_count: 4, image_count: 1 }],
});
const log = await import(base + 'panels/log.js');
const panel = log.build().el;
const errors = [];
if (panel.querySelector('details')) errors.push('request input must not be collapsible');
await new Promise((resolve) => setTimeout(resolve, 20));

// One turn reads top-to-bottom as input, then the live output for that input.
if (panel.children.length < 2
    || !panel.children[0].classList.contains('request')
    || !panel.children[1].classList.contains('live')) {
  errors.push('input/live output are not alternating in DOM order');
}

const messages = panel.querySelectorAll('.req-message');
const images = panel.querySelectorAll('.req-image');
if (fetches !== 1) errors.push('snapshot fetch count=' + fetches);
if (messages.length !== 1) errors.push('incremental message cards=' + messages.length);
if (images.length !== 1) errors.push('image previews=' + images.length);
if (!panel.textContent.includes('before image')
    || !panel.textContent.includes('after image')
    || panel.textContent.includes('previous reply')
    || panel.textContent.includes('SYSTEM FULL TEXT')) {
  errors.push('incremental ordered input is wrong');
}
if (images[0] && images[0].src !== '/debug-image/9.jpg') {
  errors.push('wrong image URL=' + images[0].src);
}
if (panel.textContent.includes('（增量）')) errors.push('incremental title suffix remains');

// 高频 golden 帧是唯一例外：默认折叠且不请求图片，手动展开后才按需加载。
Object.assign(store.S, {
  stream: { active: false, text: '', n: 0, dt: 0 },
  events: [{ t: 'request', i: 2, request_id: 8, turn: 9, driver: 'golden',
             message_count: 1, total_message_count: 1, image_count: 1 }],
});
const goldenPanel = log.build().el;
const goldenDetails = goldenPanel.querySelector('details');
if (!goldenDetails || goldenDetails.open) errors.push('golden request must start collapsed');
if (fetches !== 1) errors.push('collapsed golden fetched eagerly');
if (goldenDetails) {
  goldenDetails.open = true;
  goldenDetails.fire('toggle');
  await new Promise((resolve) => setTimeout(resolve, 20));
  if (fetches !== 2) errors.push('expanded golden did not fetch exactly once');
}

console.log(errors.length ? 'request failures: ' + errors.join('; ')
                          : 'request ok: normal expanded + golden lazy collapsed');
process.exit(errors.length ? 1 : 0);

import './shim.mjs';
import { readFileSync } from 'fs';

const base = process.argv[2] + '/js/';
for (const id of ['state','p-model','p-backend','p-scene','pill-scene','p-ep','p-step',
                  'p-dist','p-res','fps','btn-theme','btn-layout','btn-lock','dock','boot']) {
  const node = document.createElement('div'); node.id = id; document.body.appendChild(node);
}
const config = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const fixture = JSON.parse(readFileSync(process.argv[4], 'utf8'));
const requests = [];
const commands = [];
let reference = { active: false, kind: '', name: '', bbox: null, version: 0 };

globalThis.FileReader = class {
  readAsDataURL(file) {
    this.result = file.data || 'data:image/png;base64,iVBORw0KGgo=';
    setTimeout(() => this.onload && this.onload(), 0);
  }
};

globalThis.fetch = async (url, options = {}) => {
  if (url.startsWith('/frame-snapshot')) {
    return { ok: true, status: 200, json: async () => ({
      ok: true,
      image_data: 'data:image/jpeg;base64,FRAME',
      width: 640,
      height: 480,
      source: { scene: fixture.scene, episode_id: fixture.episode_id, seq: fixture.seq },
    }) };
  }
  if (url === '/react-reference') {
    const payload = JSON.parse(options.body || '{}');
    requests.push(payload);
    reference = payload.kind === 'clear'
      ? { active: false, kind: '', name: '', bbox: null, version: reference.version + 1 }
      : { active: true, kind: payload.kind,
          name: payload.kind === 'upload' ? payload.name : 'Qwen bbox',
          bbox: payload.bbox || null, version: reference.version + 1 };
    return { ok: true, status: 200, json: async () => ({ ok: true, reference }) };
  }
  if (url.startsWith('/cmd')) commands.push(url);
  const body = url.startsWith('/readiness') ? { ready: true }
             : url.startsWith('/config.json') ? config
             : url.startsWith('/status') ? { ...fixture, driver: 'react', react_reference: reference }
             : url.startsWith('/prompt') ? { text: 'p', sys_text: 's', tpl: 'react' } : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};

const errors = [];
process.on('uncaughtException', (err) => errors.push(String(err)));
process.on('unhandledRejection', (err) => errors.push(String(err)));
await import(base + 'app.js');
await new Promise((resolve) => setTimeout(resolve, 200));

const get = (id) => document.getElementById(id);
const values = () => ['x1', 'y1', 'x2', 'y2'].map((n) => Number(get('react-ref-' + n).value));
if (get('instr').value !== (fixture.instruction || '')) errors.push('default instruction must be selectable text');
const instruction = get('instr');
const { copyInstruction } = await import(base + 'panels/instr.js');
let copied = '';
instruction.select = () => {};
instruction.setSelectionRange = () => {};
document.execCommand = (command) => { copied = instruction.value; return command === 'copy'; };
instruction.value = '';
if (!await copyInstruction(instruction, 'default on HTTP') || copied !== 'default on HTTP') {
  errors.push('HTTP clipboard fallback failed');
}
instruction.value = 'my custom instruction';
instruction.fire('input');
await new Promise((resolve) => setTimeout(resolve, 300));
if (instruction.value !== 'my custom instruction') errors.push('poll overwrote custom instruction');
fixture.episode_id = String(fixture.episode_id) + '-next';
fixture.instruction = 'new episode instruction';
await new Promise((resolve) => setTimeout(resolve, 300));
if (instruction.value !== fixture.instruction) errors.push('episode switch did not update instruction');

if (get('react-ref-file')) errors.push('filesystem file input should not exist');
for (const name of ['x1', 'y1', 'x2', 'y2']) {
  const input = get('react-ref-' + name);
  if (!input || Number(input.min) !== 0 || Number(input.max) !== 1000) {
    errors.push(name + ' is not a 0..1000 input');
  }
}

// Drawing on an 800x600 shim image should normalize both axes to 0..1000.
get('react-ref-draw').fire('click');
await new Promise((resolve) => setTimeout(resolve, 0));
if (get('bbox-picker').hidden) errors.push('bbox picker did not open');
const stage = get('bbox-picker-stage');
const pickerImage = get('bbox-picker-image');
pickerImage.naturalWidth = 640;
pickerImage.naturalHeight = 480;
const square = { left: 0, top: 0, right: 800, bottom: 800, width: 800, height: 800 };
pickerImage.getBoundingClientRect = () => square;
stage.getBoundingClientRect = () => square;
// 640x480 inside 800x800 is 800x600 with 100px bars top and bottom.
stage.fire('pointerdown', { clientX: 80, clientY: 160, button: 0,
                            pointerId: 1, preventDefault() {} });
stage.fire('pointermove', { clientX: 640, clientY: 580, pointerId: 1 });
stage.fire('pointerup', { clientX: 640, clientY: 580, pointerId: 1 });
if (JSON.stringify(values()) !== JSON.stringify([100, 100, 800, 800])) {
  errors.push('drawn bbox mismatch: ' + JSON.stringify(values()));
}

get('bbox-picker-close').fire('click');
get('instr-start').fire('click');
await new Promise((resolve) => setTimeout(resolve, 30));
const expectedBBox = {
  kind: 'bbox', bbox: [100, 100, 800, 800],
  source: { scene: fixture.scene, episode_id: fixture.episode_id, seq: fixture.seq },
};
if (JSON.stringify(requests[0]) !== JSON.stringify(expectedBBox)) {
  errors.push('bbox request mismatch: ' + JSON.stringify(requests[0]));
}
if (!commands.some((url) => url.includes('do=start'))) errors.push('start command missing after bbox');

// Clipboard image replaces the bbox and is uploaded only when Start is pressed.
let pastePrevented = false;
get('instr').fire('paste', {
  clipboardData: { files: [{ name: 'clipboard.png', type: 'image/png', size: 1024,
                             data: 'data:image/png;base64,AAAA' }] },
  preventDefault() { pastePrevented = true; },
});
await new Promise((resolve) => setTimeout(resolve, 10));
if (!pastePrevented) errors.push('image paste was not consumed');
if (get('react-ref-thumb').hidden) errors.push('pasted image preview stayed hidden');
if (values().some((v) => v !== 0)) errors.push('pasting image did not clear bbox');

get('instr-start').fire('click');
await new Promise((resolve) => setTimeout(resolve, 30));
if (!requests[1] || requests[1].kind !== 'upload'
    || requests[1].name !== 'clipboard.png'
    || requests[1].image_data !== 'data:image/png;base64,AAAA') {
  errors.push('clipboard upload request mismatch: ' + JSON.stringify(requests[1]));
}

get('react-ref-clear').fire('click');
await new Promise((resolve) => setTimeout(resolve, 10));
if (!requests[2] || requests[2].kind !== 'clear') errors.push('clear request missing');

console.log(errors.length ? 'reference failures: ' + errors.join('; ')
                          : 'reference ok: clipboard image + 0..1000 drag bbox + start sync');
process.exit(errors.length ? 1 : 0);

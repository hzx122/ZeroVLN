// Exercise the actual pi-agent tools against a private controller, without an LLM.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createVlnTools } from '../../../third_party/pi-agent/.pi/extensions/vln-console.ts';

const baseUrl = process.env.PI_VLN_URL;
const originalFetch = globalThis.fetch;
globalThis.fetch = (input, options = {}) => {
  const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
  if (new URL(url).origin === new URL(baseUrl).origin) {
    const headers = new Headers(options.headers);
    headers.set('X-VLN-Batch-Token', process.env.VLN_BATCH_TOKEN);
    options = { ...options, headers };
  }
  return originalFetch(input, options);
};
const tools = createVlnTools({ baseUrl, observationDir: path.resolve('observation'),
  mapDir: path.resolve('map'), pollIntervalMs: 100 });
const signal = AbortSignal.timeout(650000);
const images = [];
async function execute(tool, params) {
  const result = await tool.execute('protocol-canary', params, signal);
  assert(!result.isError, JSON.stringify(result));
  const image = result.content.find(x => x.type === 'image');
  assert(image && Buffer.from(image.data, 'base64').length > 1000, `${tool.name}: no image`);
  if (result.details?.savedPath) {
    assert(fs.readFileSync(result.details.savedPath).equals(Buffer.from(image.data, 'base64')));
    images.push(result.details.savedPath);
  }
  return result;
}
await execute(tools.observe, { save_path: path.resolve('observation/observation-001.jpg') });
const moved = await execute(tools.localMove, { instruction: '向前移动到当前画面可见的地面。', point: [500, 700] });
assert(moved.details.motion.steps > 0, 'point must actually move');
await execute(tools.map, { save_path: path.resolve('map/map-001.jpg') });
assert.equal(new Set(images).size, images.length, 'terminal observation must not overwrite initial observation');
console.log(JSON.stringify({ actual_pi_tools: ['vln_observe', 'vln_move', 'vln_map'],
  saved_images: images.length, motion: moved.details.motion }));

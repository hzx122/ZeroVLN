// Exercise the registered provider through the same SDK/auth route as pi-agent.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { ModelRuntime } from '../../third_party/pi-agent/packages/coding-agent/dist/index.js';
import projectModels from '../config/pi/project-models.ts';
const [provider, id, output, url = ''] = process.argv.slice(2);
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'vln-provider-probe-'));
try {
  const runtime = await ModelRuntime.create({authPath:path.join(temp,'auth.json'),modelsPath:null,
    modelsStorePath:path.join(temp,'models'),refreshOnCreate:false});
  projectModels({registerProvider(name, config) {
    runtime.registerProvider(name, name === provider && url ? {...config,
      models:config.models.map(m => m.id === id ? {...m,baseUrl:url} : m)} : config);
  }});
  const model = runtime.getModel(provider,id);
  assert(model, 'model must be registered');
  const reply = await runtime.completeSimple(model, {messages:[{role:'user',timestamp:Date.now(),
    content:'Reply with a short greeting.'}]}, {maxTokens:256,signal:AbortSignal.timeout(600000)});
  fs.writeFileSync(output,JSON.stringify({provider,id,baseUrl:model.baseUrl,reply},null,2));
  assert(reply.stopReason === 'stop' && reply.content.some(c => c.type === 'text' && c.text.trim()),
    'registered provider did not return a complete text response');
} finally { fs.rmSync(temp,{recursive:true,force:true}); }

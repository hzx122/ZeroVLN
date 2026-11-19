// Explicit runtime only: no parent/global AGENTS, no discovered extensions or sessions.
import fs from 'node:fs';
import path from 'node:path';
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager }
  from '../../third_party/pi-agent/packages/coding-agent/dist/index.js';
import vlnConsole from '../../third_party/pi-agent/.pi/extensions/vln-console.ts';
import projectModels from '../config/pi/project-models.ts';
import { fileURLToPath } from 'node:url';
import { installToolBudget } from './tool_budget.mjs';

const cwd = process.cwd();
const runtime = path.join(cwd, '.runtime');
const controller = process.env.PI_VLN_URL;
const modelUrl = process.env.VLN_BATCH_MODEL_URL;
const modelProvider = process.env.VLN_BATCH_MODEL_PROVIDER || 'vllm-next-local';
const modelId = process.env.VLN_BATCH_MODEL_ID || 'qwen3.8-flash-next';
if (!controller || (!modelUrl && modelProvider !== 'anonymous-data')) throw new Error('explicit per-episode controller and model URLs required');
const rawFetch = globalThis.fetch;
globalThis.fetch = (input, options = {}) => {
  const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
  if (new URL(url).origin === new URL(controller).origin) {
    const headers = new Headers(options.headers);
    headers.set('X-VLN-Batch-Token', process.env.VLN_BATCH_TOKEN);
    options = { ...options, headers };
  }
  return rawFetch(input, options);
};
fs.mkdirSync(runtime, { recursive: true });
const projectSettings = JSON.parse(fs.readFileSync(path.join(cwd, '.pi/settings.json'), 'utf8'));
const agentsContent = fs.readFileSync(path.join(cwd, 'AGENTS.md'), 'utf8');
// Record exactly what this agent loaded even if the live repository changes later.
fs.writeFileSync(path.join(runtime, 'AGENTS.loaded.md'), agentsContent);
// Same project defaults (including compaction/retry), with resources loaded explicitly
// so no prior workspace, parent AGENTS or global session can leak into this episode.
const settingsManager = SettingsManager.inMemory({ ...projectSettings,
  packages: [], extensions: [], skills: [], prompts: [], themes: [] });
const modelRuntime = await ModelRuntime.create({ authPath: path.join(runtime, 'auth.json'),
  modelsPath: null, modelsStorePath: path.join(runtime, 'models-store'), refreshOnCreate: false });
// Reuse the project's provider definitions verbatim; only the remote address changes.
const registerModels = (pi) => projectModels({ ...pi, registerProvider(name, config) {
  pi.registerProvider(name, modelUrl && name === modelProvider ? { ...config,
    models: config.models.map(model => model.id === modelId ? { ...model, baseUrl: modelUrl } : model),
  } : config);
} });
registerModels({ registerProvider: (name, config) => modelRuntime.registerProvider(name, config) });
const loader = new DefaultResourceLoader({ cwd, agentDir: runtime, settingsManager,
  noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
  agentsFilesOverride: () => ({ agentsFiles: [{ path: path.join(cwd, 'AGENTS.md'),
    content: agentsContent }] }),
  additionalSkillPaths: [fileURLToPath(new URL('../../third_party/pi-agent/.pi/skills/vln-visual-reference', import.meta.url))],
  extensionFactories: [vlnConsole, registerModels],
});
await loader.reload();
const sessionManager = SessionManager.create(cwd, path.join(runtime, 'sessions'));
const { session } = await createAgentSession({ cwd, agentDir: runtime, settingsManager, modelRuntime,
  model: modelRuntime.getModel(modelProvider, modelId), thinkingLevel: 'medium',
  resourceLoader: loader, sessionManager });
const maxToolCalls = Number(process.env.VLN_BATCH_MAX_TOOL_CALLS || 100);
const budget = installToolBudget(session, maxToolCalls);
let turns = 0;
let termination = 'error';
let lastStopReason = null;
try {
  session.subscribe((event) => {
    if (event.type === 'message_end' || event.type === 'tool_execution_end') {
      process.stdout.write(JSON.stringify(event) + '\n');
    }
    if (event.type === 'turn_end') turns++;
    if (event.type === 'message_end' && event.message.role === 'assistant') lastStopReason = event.message.stopReason;
  });
  if (process.env.VLN_BATCH_DRY_RUN === '1') {
    console.log(JSON.stringify({ initialized: true, context: loader.getAgentsFiles().agentsFiles.map(x => x.path),
      model: modelId, provider: modelProvider, tools: session.getActiveToolNames(),
      skills: loader.getSkills().skills.map(x => x.name),
      modelMaxTokens: modelRuntime.getModel(modelProvider, modelId).maxTokens,
      compaction: settingsManager.getCompactionSettings() }));
    termination = 'dry_run';
  } else {
    try {
      await session.prompt(process.env.VLN_BATCH_INSTRUCTION);
    } catch (error) {
      if (!budget.reached) throw error;
    }
    termination = budget.reached ? 'tool_limit' : lastStopReason === 'stop' ? 'normal' : (lastStopReason || 'incomplete');
  }
} finally {
  fs.writeFileSync(path.join(runtime, 'agent-result.json'), JSON.stringify({ termination, lastStopReason, toolCalls: budget.calls, maxToolCalls, turns, session: sessionManager.getSessionFile() }));
  session.dispose();
}

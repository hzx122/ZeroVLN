import './shim.mjs';
import { readFileSync } from 'fs';
const base = process.argv[2] + '/js/';
for (const id of ['state','p-model','p-backend','p-scene','pill-scene','p-ep','p-step',
                  'p-dist','p-res','fps','btn-theme','btn-layout','btn-lock','dock','boot']) {
  const n = document.createElement('div'); n.id = id; document.body.appendChild(n);
}
const CFG_J = JSON.parse(readFileSync(process.argv[3], 'utf8'));
if (!CFG_J.models.some((model) => model.key === 'lightnav0')) {
  CFG_J.models.push({ key: 'lightnav0', label: 'LightNav-0',
                      desc: 'independent policy', fm: false, lightnav: true });
}
if (!CFG_J.drivers.some((driver) => driver.key === 'lightnav')) {
  CFG_J.drivers.push({ key: 'lightnav', label: 'LightNav-0·轨迹', desc: 'waypoints' });
}
// Exercise the unified-backend / switchable-capability view without requiring
// a dedicated live fixture. Non-capability asset sets must remain unfiltered.
CFG_J.capability_dimensions = [
  { key: 'perception', label: 'Perception', count: 1 },
  { key: 'grounding', label: 'Grounding', count: 1 },
];
CFG_J.episode_dimensions = {
  'sage-839883': { '0_0': 'perception', '0_1': 'grounding' },
};
CFG_J.assets.push({ key: 'rxr_train_guide', label: 'RxR-CE / Train Guide', desc: 'train',
  scenes: [['rxr-train-guide-a', 'a (2)'], ['rxr-train-guide-b', 'b (1)']] });
Object.assign(CFG_J.scene_of, { 'rxr-train-guide-a': 'rxr_train_guide',
                              'rxr-train-guide-b': 'rxr_train_guide' });
Object.assign(CFG_J.episodes, { 'rxr-train-guide-a': ['1', '2'], 'rxr-train-guide-b': ['3'] });
CFG_J.episode_languages = { 'rxr-train-guide-a': { '1': 'en-US', '2': 'hi-IN' },
                            'rxr-train-guide-b': { '3': 'te-IN' } };
const STATUS = JSON.parse(readFileSync(process.argv[4], 'utf8'));
const cmds = [];
const alerts = [];
window.alert = (message) => alerts.push(String(message));
globalThis.fetch = async (url) => {
  if (url.startsWith('/cmd')) cmds.push(url);
  const body = url.startsWith('/readiness') ? { ready: true }
             : url.startsWith('/config.json') ? CFG_J
             : url.startsWith('/status') ? STATUS
             : url.startsWith('/prompt') ? { text: 'p', sys_text: 's', tpl: 'en_r1' }
             : url.startsWith('/traj_cases') ? {
                 ok: true, scene: 'sage-839883', traj: '0', cases: [
                   { task_id: '0_0', instruction_type: 'Add_Causality',
                     start_label: 'plant_1', goal_label: 'red_wine_glass_4',
                     instruction: 'Take the plant from the kitchen and place it next to the red wine glass in the living room.' },
                 ] }
             : {};
  return { ok: true, status: 200, json: async () => body,
           text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) };
};
const errs = [];
process.on('uncaughtException', (e) => errs.push(String(e)));
process.on('unhandledRejection', (e) => errs.push(String(e)));
await import(base + 'app.js');
const { assetDisplayLabel } = await import(base + 'controls.js');
await new Promise((r) => setTimeout(r, 200));

const g = (id) => document.getElementById(id);
const step = (name, fn) => {
  try { fn(); console.log('  ok   ' + name) }
  catch (e) { errs.push(name + ': ' + e.message); console.log('  FAIL ' + name + ' -- ' + e.message) }
};
console.log('every control found in the document after render:');
for (const id of ['assets','scene','ep','load','model','driver','mode','tpl','nact',
                  'capability','row-capability',
                  'maxtools','temp','frompose','instr','react-ref-x1','react-ref-y1',
                  'react-ref-x2','react-ref-y2','react-ref-draw','load-hint','asset-desc',
                  'walk-golden',
                  'traj-all','traj-cases','traj-cases-list','traj-cases-close',
                  'traj-cases-title','traj-cases-n']) {
  const hit = !!g(id);
  console.log('  ' + (hit ? 'ok  ' : 'MISS') + ' #' + id);
  if (!hit) errs.push('missing #' + id);
}
console.log('interactions:');
step('expert playback speed changes display stride, not path step', () => {
  g('golden_stride').value = '8'; g('golden_stride').fire('change');
  if (!cmds.some((c) => c.includes('do=set_golden_stride') && c.includes('n=8')))
    throw new Error('missing playback stride command');
  if (Number(g('golden_step').value) !== .1)
    throw new Error('playback speed changed physical path resolution');
});
step('dataset labels separate split and role from simulator and scene counts', () => {
  const cases = [
    ['R2R / MP3D (habitat)', 'R2R-CE / Val Unseen'],
    ['RxR / MP3D (habitat)', 'RxR-CE / Val Unseen / Guide'],
    ['R2R-CE / Train (MP3D)', 'R2R-CE / Train'],
    ['RxR-CE / Train Guide (MP3D)', 'RxR-CE / Train / Guide'],
    ['RxR-CE / Train Follower (MP3D)', 'RxR-CE / Train / Follower'],
    ['R2R-CE / Val Seen', 'R2R-CE / Val Seen'],
    ['Custom dataset', 'Custom dataset'],
  ];
  for (const [label, expected] of cases) {
    const asset = { key: 'native-key', label };
    if (assetDisplayLabel(asset) !== expected) throw new Error(label);
    if (asset.key !== 'native-key' || asset.label !== label) throw new Error('catalogue mutated');
  }
  for (const option of g('assets').children) {
    if (!option.textContent.endsWith(' 场景）')) throw new Error('scene count has no unit');
  }
});
step('training language filter keeps full catalogue and native IDs', () => {
  g('assets').value = 'rxr_train_guide'; g('assets').fire('change');
  if (g('row-language').hidden) throw new Error('missing language selector');
  g('language').value = 'te-IN'; g('language').fire('change');
  if (g('scene').value !== 'rxr-train-guide-b' || g('ep').value !== '3')
    throw new Error('language without episodes in the old scene did not select a valid scene');
  g('load').fire('click');
  if (!cmds.some((c) => c.includes('episode_id=3') && c.includes('rxr-train-guide-b')))
    throw new Error('display labels changed the native episode ID');
  g('language').value = ''; g('language').fire('change');
  if (g('scene').children.length !== 2) throw new Error('All languages lost a scene');
  if (CFG_J.episodes['rxr-train-guide-a'].length !== 2) throw new Error('catalogue mutated');
});
step('switch asset set -> 3drs', () => {
  g('assets').value = '3drs'; g('assets').fire('change');
  if (!g('scene').children.length) throw new Error('scene list empty');
  if (g('scene').children.length !== 143) throw new Error('want 143 scenes, got ' + g('scene').children.length);
  if (!g('row-language').hidden) throw new Error('language filter leaked to another asset');
});
step('pick a scene refills episodes', () => {
  const s = g('scene'); s.value = s.children[3].value; s.fire('change');
  if (g('load-hint').hidden) throw new Error('pending hint not shown after a staged pick');
});
step('capability view stays off for non-capability assets', () => {
  if (!g('row-capability').hidden) throw new Error('capability row shown for 3drs');
});
step('driver -> classic greys the tool budget', () => {
  g('driver').value = 'legacy'; g('driver').fire('change');
  if (!g('maxtools').disabled) throw new Error('maxtools should be disabled for the legacy driver');
  if (g('nact').disabled) throw new Error('nact should be live for the legacy driver');
});
step('driver -> react flips it back', () => {
  g('driver').value = 'react'; g('driver').fire('change');
  if (g('maxtools').disabled) throw new Error('maxtools should be live for react');
  if (!g('nact').disabled) throw new Error('nact should be dead for react');
});
step('model -> LightNav locks its matching driver', () => {
  g('model').value = 'lightnav0'; g('model').fire('change');
  if (g('driver').value !== 'lightnav') throw new Error('driver was not switched to lightnav');
  if (!g('driver').disabled) throw new Error('LightNav driver should be locked');
});
step('model -> generic unlocks driver and returns to react', () => {
  const generic = CFG_J.models.find((model) => !model.fm && !model.lightnav);
  g('model').value = generic.key; g('model').fire('change');
  if (g('driver').value !== 'react') throw new Error('driver did not return to react');
  if (g('driver').disabled) throw new Error('generic model should unlock driver');
});
step('load button issues switch', () => {
  g('load').fire('click');
  if (!cmds.some((c) => c.includes('do=switch'))) throw new Error('no switch command');
});
step('from-pose checkbox updates the server-side setting', () => {
  g('frompose').checked = false; g('frompose').fire('change');
  if (!cmds.some((c) => c.includes('do=set_frompose&on=0')))
    throw new Error('no set_frompose command');
});
step('expert path rejects a non-SAGE backend with a visible message', () => {
  const before = cmds.length;
  g('walk-golden').fire('click');
  if (cmds.length !== before) throw new Error('non-SAGE command reached the server');
  if (!alerts.some((message) => message.includes('只支持 SAGE')))
    throw new Error('missing SAGE-only alert');
});
const { S } = await import(base + 'store.js');
step('expert path is sent on SAGE', () => {
  S.backend = 'sage';
  g('walk-golden').fire('click');
  if (!cmds.some((c) => c.includes('do=walk_golden')))
    throw new Error('SAGE command was not sent');
});
step('theme toggle', () => {
  const before = document.documentElement.getAttribute('data-theme');
  g('btn-theme').fire('click');
  if (document.documentElement.getAttribute('data-theme') === before)
    throw new Error('theme did not change');
});

// traj-all picker (sage): the per-trajectory case list opens, lists the mock's
// two cases, and picking one issues a switch -- the handler is async (fetch),
// so wait for it rather than asserting synchronously.
try {
  g('assets').value = 'sage'; g('assets').fire('change');
  if (g('row-capability').hidden) throw new Error('capability row hidden on sage');
  if (g('ep').children.length !== 1 || g('ep').value !== '0_0')
    throw new Error('perception view did not filter episodes: scene=' + g('scene').value
                    + ' dimension=' + g('capability').value
                    + ' episodes=' + g('ep').children.length + '/' + g('ep').value);
  g('capability').value = 'grounding'; g('capability').fire('change');
  if (g('ep').children.length !== 1 || g('ep').value !== '0_1')
    throw new Error('grounding view did not filter episodes');
  g('capability').value = 'perception'; g('capability').fire('change');
  if (g('traj-all').hidden) throw new Error('traj-all not visible on a sage scene');
  g('traj-all').fire('click');
  await new Promise((r) => setTimeout(r, 40));
  if (g('traj-cases').hidden) throw new Error('picker did not open');
  const list = g('traj-cases-list');
  if (list.children.length !== 1)
    throw new Error('want 1 current-dimension case, got ' + list.children.length);
  list.children[0].fire('click');
  await new Promise((r) => setTimeout(r, 10));
  if (!cmds.some((c) => c.includes('do=switch') && c.includes('episode_id=0_0')))
    throw new Error('no switch to case 0_0');
  if (!g('traj-cases').hidden) throw new Error('picker not closed after a pick');
  console.log('  ok   traj-all picker flow (sage)');
} catch (e) {
  errs.push('traj-all flow: ' + e.message);
  console.log('  FAIL traj-all flow -- ' + e.message);
}

console.log('\ncommands issued:', cmds.map((c) => c.split('?')[1].split('&')[0]).join(' '));
console.log(errs.length ? '\nFAILURES:\n' + errs.join('\n') : '\nall interaction checks passed');
process.exit(errs.length ? 1 : 0);

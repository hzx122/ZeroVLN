// 任务设置 -- every dropdown the console has, and the rules about when a pick
// takes effect.
//
// Two of those rules are load-bearing:
//
//  * Scene and episode are the only two-step controls on the page. Loading an
//    episode takes seconds, so scrubbing the dropdown would thrash the sim;
//    picking stages the choice and 加载 commits it. They are therefore the only
//    controls whose value may legitimately differ from server state, which is
//    what `epDirty` is for -- without it the 250 ms poll would overwrite a pick
//    before the user reached the button.
//  * ReAct drives with tools, so the per-turn template and the action count mean
//    nothing there, and the tool budget means nothing anywhere else. Greying out
//    beats letting a dead control look effective.

import { el, within, labelRow, select } from './util.js';
import { CFG, S, onStatus } from './store.js';
import { cmd, getJSON } from './api.js';
import { emit } from './bus.js';
import { checkpointPicker } from './action-head.js';

const descOf = (list) => Object.fromEntries(list.map((x) => [x.key, x.desc || '']));
const setOf = (key) => CFG.assets.find((a) => a.key === key);

// Older console processes keep their catalogue in memory until restarted.
export function assetDisplayLabel(asset) {
  const legacy = {
    'R2R / MP3D (habitat)': 'R2R-CE / Val Unseen',
    'RxR / MP3D (habitat)': 'RxR-CE / Val Unseen / Guide',
    'R2R-CE / Train (MP3D)': 'R2R-CE / Train',
    'RxR-CE / Train Guide (MP3D)': 'RxR-CE / Train / Guide',
    'RxR-CE / Train Follower (MP3D)': 'RxR-CE / Train / Follower',
  };
  return legacy[asset.label] || asset.label;
}

let epDirty = false;
let curScene = '';      // what the server actually has loaded
let curEp = '';
let trajAllBtn = null;  // built in build(); visible only for sage scenes
let headPicker = null;
// CFG is populated after ES modules are evaluated, so derive these in build().
let capabilityItems = [];

/** ``'0_12'`` -> ``'0'``; an id without a trailing index stays as-is. */
const trajOf = (id) => String(id || '').replace(/_\d+$/, '').split('__', 1)[0];

/** Which trajectory the "全部用例" button targets: the loaded episode when it
 *  belongs to the selected scene (the real run), else the staged pick in the
 *  Episode dropdown -- so switching scene but not yet loading must not reuse
 *  the previous scene's id. */
function curTraj() {
  const sc = root && q('scene') ? q('scene').value : '';
  const ids = episodeIds(sc);
  const loaded = ids.indexOf(S.episode_id) >= 0 ? S.episode_id : '';
  const pick = ids.indexOf(q('ep').value) >= 0 ? q('ep').value : '';
  return loaded || pick;
}

function activeDimension() {
  const key = root && q('assets') ? q('assets').value : '';
  return capabilityItems.length && hasCapabilityForSet(key) && q('capability')
    ? q('capability').value : '';
}

function hasCapabilityForSet(key) {
  const set = setOf(key);
  return !!((set && set.scenes) || []).some(([scene]) =>
    Object.keys((CFG.episode_dimensions || {})[scene] || {}).length);
}

function episodeDimension(scene, episodeId) {
  return ((CFG.episode_dimensions || {})[scene] || {})[episodeId] || '';
}

function episodeLanguage(scene, episodeId) {
  return ((CFG.episode_languages || {})[scene] || {})[episodeId] || '';
}

function activeLanguage() {
  return root && q('row-language') && !q('row-language').hidden
    ? q('language').value : '';
}

function fillLanguages(key) {
  const previous = q('language').value;
  const counts = {};
  for (const [scene] of (setOf(key)?.scenes || [])) {
    for (const id of CFG.episodes[scene] || []) {
      const language = episodeLanguage(scene, id);
      if (language) counts[language] = (counts[language] || 0) + 1;
    }
  }
  const labels = { 'en-US': 'English (US)', 'en-IN': 'English (India)',
                   'hi-IN': 'Hindi', 'te-IN': 'Telugu' };
  q('row-language').hidden = !Object.keys(counts).length;
  q('language').replaceChildren(
    el('option', { value: '', text: 'All languages' }),
    ...Object.keys(counts).sort().map((key) => el('option', {
      value: key, text: (labels[key] || key) + ' (' + counts[key] + ')',
    })),
  );
  if (counts[previous]) q('language').value = previous;
}

function episodeIds(scene) {
  const ids = CFG.episodes[scene] || [];
  const dimension = activeDimension();
  const language = activeLanguage();
  return ids.filter((id) =>
    (!dimension || episodeDimension(scene, id) === dimension) &&
    (!language || episodeLanguage(scene, id) === language));
}

// The panel's own root. Every lookup below is scoped to it rather than going
// through document.getElementById: build() runs before the dock has attached
// the panel, so at wiring time these ids are not in the document yet.
let root = null;
const q = (id) => within(root, id);

function fillEps(scene, keep) {
  const ids = episodeIds(scene);
  q('ep').replaceChildren(...ids.map((i) => {
    const language = episodeLanguage(scene, i);
    return el('option', { value: i, text: i + (language ? ' / ' + language : '') });
  }));
  if (keep && ids.indexOf(keep) >= 0) q('ep').value = keep;
  updateTrajAll();
}

function updateTrajAll() {
  if (!trajAllBtn) return;
  const sc = root && q('scene') ? q('scene').value : '';
  const show = sc && CFG.scene_of[sc] === 'sage' && trajOf(curTraj());
  trajAllBtn.hidden = !show;
}

function fillScenes(key, keep) {
  const set = setOf(key);
  fillLanguages(key);
  if (q('row-capability')) q('row-capability').hidden = !hasCapabilityForSet(key);
  const sel = q('scene');
  const scenes = ((set && set.scenes) || []).filter(([id]) => episodeIds(id).length);
  sel.replaceChildren(...scenes.map(([id, label]) => {
    const n = episodeIds(id).length;
    const text = activeDimension() || activeLanguage() ? label + ' / ' + n : label;
    return el('option', { value: id, text });
  }));
  if (keep && scenes.some(([id]) => id === keep)) sel.value = keep;
  q('asset-desc').textContent = (set && set.desc) || '';
  fillEps(sel.value, sel.value === curScene ? curEp : null);
}

function epPicked() {
  epDirty = q('scene').value !== curScene || q('ep').value !== curEp;
  markPending(epDirty);
}

function markPending(on) {
  q('load-hint').hidden = !on;
  q('load').className = on ? 'b-load pend' : 'b-load';
}

function applyDriver() {
  const v = q('driver').value;
  if (headPicker) headPicker.show(v === 'action_head');
  const isReact = v === 'react' || v.startsWith('action_head')
    || ['embodied_point', 'unilavira_bbox'].includes(v);
  const isFm = v === 'fm';
  const isLightnav = v === 'lightnav';
  const noTemplate = isReact || isFm || isLightnav;
  q('driver-desc').textContent = descOf(CFG.drivers)[v] || '';
  q('row-tpl').className = noTemplate ? 'row off' : 'row';
  dim(q('nact'), q('lab-nact'), !noTemplate);
  dim(q('maxtools'), q('lab-maxtools'), isReact);   // only ReAct shows the tool budget
  emit('driver', { react: isReact, fm: isFm, lightnav: isLightnav });
  emit('cfg');
}

function dim(input, label, live) {
  input.disabled = !live;
  input.style.opacity = label.style.opacity = live ? 1 : 0.4;
}

function isFmModel(key) {
  const m = CFG.models.find((x) => x.key === key);
  return !!(m && m.fm);
}

function isLightnavModel(key) {
  const m = CFG.models.find((x) => x.key === key);
  return !!(m && m.lightnav);
}

function applyModel() {
  const mk = q('model').value;
  q('model-desc').textContent = descOf(CFG.models)[mk] || '';
  const drv = q('driver');
  if (isFmModel(mk)) {
    // VLN-FM 模型：驱动锁死 FM·流匹配，禁用驱动下拉（后端 _enforce_model_driver_lock 同款硬约束）
    if (drv.value !== 'fm') drv.value = 'fm';
    drv.disabled = true;
    drv.style.opacity = 0.4;
    applyDriver();
  } else if (isLightnavModel(mk)) {
    // LightNav 是独立策略模型，不经过普通 chat-model endpoint；选择模型即
    // 锁定它自己的 waypoint 驱动，避免“页面显示 Qwen、实际跑 LightNav”。
    if (drv.value !== 'lightnav') drv.value = 'lightnav';
    drv.disabled = true;
    drv.style.opacity = 0.4;
    applyDriver();
  } else {
    // 普通模型：启用驱动下拉；独立策略驱动均退回默认 ReAct。
    drv.disabled = false;
    drv.style.opacity = 1;
    if (drv.value === 'fm' || drv.value === 'lightnav') {
      drv.value = 'react'; applyDriver();
    }
  }
}

export function build() {
  const c = CFG.cur;
  capabilityItems = (CFG.capability_dimensions || []).map((d) => ({
    key: d.key, label: d.label + '（' + d.count + '）',
  }));
  const box = el('div.dk-pane.dk-pad');
  root = box;
  headPicker = checkpointPicker();
  const assetItems = CFG.assets.map((a) => ({
    key: a.key, label: assetDisplayLabel(a) + '（' + a.scenes.length + ' 场景）',
  }));

  box.append(
    labelRow('资产集', select('assets', assetItems, c.set)),
    el('div.hint', { id: 'asset-desc' }),
    el('div', { id: 'row-language', hidden: true }, [
      labelRow('Language', select('language', [], '')),
    ]),
    el('div', { id: 'row-capability', hidden: !capabilityItems.length }, [
      labelRow(CFG.dimension_label || 'Capability', select('capability', capabilityItems,
                                  c.capability || (capabilityItems.length
                                    ? capabilityItems[0].key : ''))),
    ]),
    el('div.row', {}, [
      el('label', { text: '场景' }),
      select('scene', [], ''),
      el('label.narrow', { text: '回合' }),
      el('select', { id: 'ep' }),
      el('button.b-load', { id: 'load', text: '⟳ 加载' }),
      el('button.b-plain.b-sm', { id: 'traj-all', type: 'button', hidden: true,
                                  text: '📋 当前类用例',
                                  title: '当前轨迹在所选能力维度下的指令用例' }),
    ]),
    el('div.hint.warn', { id: 'load-hint', hidden: true,
                          text: '已选好，点「加载」才会切换' }),
    labelRow('大脑/策略模型', select('model', CFG.models, c.model)),
    el('div.hint', { id: 'model-desc' }),
    labelRow('驱动', select('driver', CFG.drivers, c.driver)),
    el('div.hint', { id: 'driver-desc' }),
    headPicker.root,
    labelRow('模式', select('mode', CFG.modes, c.mode)),
    el('div.hint', { id: 'mode-desc' }),
    el('div.row', { id: 'row-tpl' }, [
      el('label', { text: '提示词' }), select('tpl', CFG.templates, c.tpl),
    ]),
    el('div.hint', { id: 'tpl-desc' }),
    el('div.row', {}, [
      el('label', { id: 'lab-nact', text: '动作数' }),
      el('input', { type: 'number', id: 'nact', min: 1, max: 8, value: c.nact }),
      el('label.narrow', {
        id: 'lab-maxtools', text: '工具预算',
        title: 'ReAct 一个回合最多允许多少次工具调用；模型每批工具返回里都能看到还剩几次',
      }),
      el('input', { type: 'number', id: 'maxtools', min: CFG.maxtools.min,
                    max: CFG.maxtools.max, value: c.maxtools }),
      el('label.narrow', { text: '温度' }),
      el('input', { type: 'text', id: 'temp', value: 'auto',
                    title: 'auto = 跟随模式预设（思考 1.0 / 非思考 0.7）' }),
      el('label.narrow', {
        id: 'lab-golden-step', text: '专家步长',
        title: '走专家轨迹的弧长 resample 步长（米，0.02~1.0）；越小步越密、模型输出框帧越多',
      }),
      el('input', { type: 'number', id: 'golden_step', min: 0.02, max: 1.0,
                    step: 0.01, value: 0.1 }),
      el('label.narrow', { text: '专家回放' }),
      select('golden_stride', [
        { key: '1', label: '精细 · 每步显示' },
        { key: '2', label: '较快 · 每2步显示' },
        { key: '4', label: '快速 · 每4步显示' },
        { key: '8', label: '极速 · 每8步显示' },
      ], '4'),
      el('label.narrow', {
        id: 'lab-pg-lookahead', text: '前瞻距离',
        title: 'Point-Goal 前瞻距离（米，0.5~10）：专家 path 上前方 waypoint 选点弧长，sage 室内默认 3m',
      }),
      el('input', { type: 'number', id: 'pg_lookahead', min: 0.5, max: 10.0,
                    step: 0.5, value: 3 }),
      el('label.chk', {}, [
        el('input', { type: 'checkbox', id: 'frompose', checked: true }), ' 从当前位置',
      ]),
    ]),
  );

  // ---- wiring ------------------------------------------------------------
  q('assets').addEventListener('change', () => {
    fillScenes(q('assets').value, curScene);
    epPicked();
  });
  q('language').addEventListener('change', () => {
    fillScenes(q('assets').value, q('scene').value);
    epPicked();
  });
  if (capabilityItems.length) q('capability').addEventListener('change', () => {
    fillScenes(q('assets').value, q('scene').value);
    epPicked();
  });
  q('scene').addEventListener('change', () => { fillEps(q('scene').value); epPicked() });
  q('ep').addEventListener('change', epPicked);
  q('load').addEventListener('click', () => {
    cmd('switch', { scene: q('scene').value, episode_id: q('ep').value });
  });

  q('tpl').addEventListener('change', () => {
    q('tpl-desc').textContent = descOf(CFG.templates)[q('tpl').value] || '';
    cmd('set_tpl', { tpl: q('tpl').value });
    emit('cfg');
  });
  q('mode').addEventListener('change', () => {
    q('mode-desc').textContent = descOf(CFG.modes)[q('mode').value] || '';
    cmd('set_mode', { mode: q('mode').value });
  });
  q('driver').addEventListener('change', () => {
    cmd('set_driver', { driver: q('driver').value });
    applyDriver();
  });
  q('model').addEventListener('change', () => {
    cmd('set_model', { model: q('model').value });
    applyModel();
  });
  q('nact').addEventListener('change', () => {
    cmd('set_nact', { n: q('nact').value }); emit('cfg');
  });
  q('maxtools').addEventListener('change', () => {
    cmd('set_maxtools', { n: q('maxtools').value }); emit('cfg');
  });
  q('temp').addEventListener('change', () => cmd('set_temp', { t: q('temp').value }));
  q('golden_step').addEventListener('change', () => {
    cmd('set_golden_step', { m: q('golden_step').value });
  });
  q('golden_stride').addEventListener('change', () => {
    cmd('set_golden_stride', { n: q('golden_stride').value });
  });
  q('pg_lookahead').addEventListener('change', () => {
    cmd('set_pointgoal_lookahead', { m: q('pg_lookahead').value });
  });
  q('frompose').addEventListener('change', () => {
    cmd('set_frompose', { on: q('frompose').checked ? 1 : 0 });
  });

  // ---- 该轨迹全部用例（sage）-----------------------------------------------
  const trajCaseList = el('div.traj-cases-list', { id: 'traj-cases-list' });
  const trajCases = el('div.bbox-picker', { id: 'traj-cases', hidden: true }, [
    el('div.bbox-picker-card', {}, [
      el('div.bbox-picker-head', {}, [
        el('b', { id: 'traj-cases-title', text: '轨迹用例' }),
        el('span.hint.mono', { id: 'traj-cases-n' }),
        el('button.b-plain', { id: 'traj-cases-close', type: 'button', text: '关闭' }),
      ]),
      trajCaseList,
    ]),
  ]);
  box.append(trajCases);

  trajAllBtn = q('traj-all');
  const trajCaseTitle = q('traj-cases-title');
  const trajCaseCount = q('traj-cases-n');
  const closeTrajCases = () => { trajCases.hidden = true; };
  q('traj-cases-close').addEventListener('click', () => closeTrajCases());
  trajCases.addEventListener('click', (event) => {
    if (event.target === trajCases) closeTrajCases();   // backdrop click
  });
  trajAllBtn.addEventListener('click', async () => {
    const scene = q('scene').value;
    const traj = trajOf(curTraj());
    if (!scene || !traj) return;
    const dimension = activeDimension();
    trajCaseTitle.textContent = '轨迹 ' + traj + (dimension ? ' · ' + dimension : '') + ' 用例';
    trajCaseCount.textContent = '';
    trajCaseList.replaceChildren(el('div.hint', { text: '加载中…' }));
    trajCases.hidden = false;
    let r;
    try {
      r = await getJSON('/traj_cases?scene=' + encodeURIComponent(scene)
                        + '&traj=' + encodeURIComponent(traj)
                        + (dimension ? '&dimension=' + encodeURIComponent(dimension) : ''));
    } catch (e) {
      r = { ok: false, err: String(e) };
    }
    if (!r || !r.ok) {
      trajCaseList.replaceChildren(el('div.hint.warn',
        { text: (r && r.err) || '获取失败' }));
      return;
    }
    const cases = r.cases || [];
    if (!cases.length) {
      trajCaseList.replaceChildren(el('div.hint', { text: '该轨迹没有可用用例' }));
      return;
    }
    trajCaseCount.textContent = cases.length + ' 条';
    trajCaseList.replaceChildren(...cases.map((c) => {
      const head = '[' + (c.instruction_type || '—') + '] '
                   + (c.start_label || '?') + ' → ' + (c.goal_label || '?');
      const text = String(c.instruction || '');
      return el('button.traj-case', {
        type: 'button', title: head + '　' + text,
        onclick: () => {
          closeTrajCases();
          const eps = q('ep');
          if (!Array.from(eps.options).some((o) => o.textContent === c.task_id)) {
            eps.append(el('option', { text: c.task_id }));
          }
          eps.value = c.task_id;
          cmd('switch', { scene, episode_id: c.task_id });
        },
      }, [
        el('span.traj-case-head', { text: head }),
        el('span.traj-case-text',
           { text: text.length > 150 ? text.slice(0, 150) + '…' : text }),
      ]);
    }));
  });

  fillScenes(c.set, c.scene);
  q('tpl-desc').textContent = descOf(CFG.templates)[c.tpl] || '';
  q('mode-desc').textContent = descOf(CFG.modes)[c.mode] || '';
  applyDriver();
  applyModel();

  onStatus(sync);
  return box;
}

function sync(s) {
  headPicker.sync(s);
  curScene = s.scene;
  curEp = s.episode_id;
  if (!epDirty) {
    // The set dropdown follows the loaded scene, or the scene the server
    // reports would not be in the list the scene dropdown is showing.
    const set = CFG.scene_of[s.scene];
    const dimension = episodeDimension(s.scene, s.episode_id);
    if (dimension && q('capability') && q('capability').value !== dimension) {
      q('capability').value = dimension;
      fillScenes(set || q('assets').value, s.scene);
    }
    if (set && q('assets').value !== set) { q('assets').value = set; fillScenes(set, s.scene) }
    // A load from another client may fall outside this browser's language view.
    if (activeLanguage() && episodeLanguage(s.scene, s.episode_id) !== activeLanguage()) {
      q('language').value = '';
      fillScenes(set || q('assets').value, s.scene);
    }
    if (q('scene').value !== s.scene) { q('scene').value = s.scene; fillEps(s.scene, s.episode_id) }
    if (q('ep').value !== s.episode_id) q('ep').value = s.episode_id;
    markPending(false);
  } else if (q('scene').value === s.scene && q('ep').value === s.episode_id) {
    epDirty = false;                       // the pick landed
    markPending(false);
  }
  if (s.model && q('model').value !== s.model) { q('model').value = s.model; applyModel() }
  if (s.driver && q('driver').value !== s.driver) { q('driver').value = s.driver; applyDriver() }
  if (s.golden_step != null && q('golden_step')
      && Number(q('golden_step').value) !== Number(s.golden_step)) {
    q('golden_step').value = s.golden_step;
  }
  if (s.golden_stride != null && q('golden_stride')) {
    q('golden_stride').value = s.golden_stride;
  }
  if (s.pointgoal_lookahead != null && q('pg_lookahead')
      && Number(q('pg_lookahead').value) !== Number(s.pointgoal_lookahead)) {
    q('pg_lookahead').value = s.pointgoal_lookahead;
  }
  updateTrajAll();
}

/** The model's label rather than its key, for the header pill. */
export function modelLabel() {
  const m = CFG.models.find((x) => x.key === (q('model') ? q('model').value : ''));
  return (m && m.label) || S.model || '-';
}

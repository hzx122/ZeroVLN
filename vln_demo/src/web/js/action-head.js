// Directory choice is staged; only Confirm invokes the small-brain loader.
import { el, within } from './util.js';
import { cmd, getJSON } from './api.js';

export function checkpointPicker() {
  let listing = null;
  let request = 0;
  let loading = false;
  let submitting = false;
  let running = false;
  let pathDirty = false;
  let catalog = [];
  let catalogRequest = 0;
  const root = el('div', { id: 'action-head-picker', hidden: true }, [
    el('div.hint', { text: '先确认实验、列出 checkpoint，再手动选一个加载。' }),
    el('div', { id: 'ah-picker-controls' }, [
    el('div.row', {}, [
      el('input', { id: 'ah-search', type: 'search', placeholder: '搜索实验 / v3 / v4.3 / d012 / bootstrap',
                    'aria-label': '搜索小脑权重' }),
      el('button', { id: 'ah-refresh', type: 'button', text: '刷新权重列表' }),
    ]),
    el('div.hint', { id: 'ah-catalog-status' }),
    el('label', { for: 'ah-experiment', text: '实验（按最近保存排序）' }),
    el('select', { id: 'ah-experiment', size: 5, style: 'width:100%' }),
    el('button', { id: 'ah-confirm', type: 'button', text: '确认实验，列出 checkpoint', disabled: true }),
    ]),
    el('div.row', { id: 'ah-picked', hidden: true }, [
      el('span', { id: 'ah-picked-label' }),
      el('button', { id: 'ah-change', type: 'button', text: '更换实验' }),
    ]),
    el('div', { id: 'ah-choices', hidden: true }, [
      el('label', { for: 'ah-step', text: '可用 checkpoint（请选择一个，不自动选择最新步）' }),
      el('select', { id: 'ah-step', size: 7, style: 'width:100%' }),
    ]),
    el('div.row', { id: 'ah-chosen', hidden: true }, [
      el('span', { id: 'ah-chosen-label' }),
      el('button', { id: 'ah-change-step', type: 'button', text: '更换 checkpoint' }),
    ]),
    el('details', { id: 'ah-advanced' }, [
    el('summary', { text: '高级：按路径浏览' }),
    el('div.row', {}, [
      el('label', { text: '小脑目录' }),
      el('input', { id: 'ah-path', type: 'text', placeholder: 'artifacts 内的 checkpoint 目录',
                    title: '可粘贴 global_step 目录或其 action_head 子目录的绝对路径' }),
      el('button', { id: 'ah-browse', type: 'button', text: '打开' }),
      el('button', { id: 'ah-up', type: 'button', text: '上一级', disabled: true }),
    ]),
    el('select', { id: 'ah-directories', size: 6, style: 'width:100%',
                   'aria-label': 'checkpoint 子目录' }),
    el('button', { id: 'ah-enter', type: 'button', text: '进入所选目录', disabled: true }),
    ]),
    el('div.hint', { id: 'ah-selection' }),
    el('button.b-load', { id: 'ah-load', type: 'button', text: '加载选中的 checkpoint', disabled: true }),
    el('div.hint', { id: 'ah-active', text: '小脑尚未加载' }),
    el('div.hint', { id: 'ah-loading' }),
    el('div.hint.warn', { id: 'ah-error' }),
  ]);
  const q = (id) => within(root, id);
  function buttons() {
    q('ah-confirm').disabled = !q('ah-experiment').value;
    q('ah-load').disabled = !listing?.checkpoint || pathDirty || loading || submitting || running;
    q('ah-enter').disabled = !q('ah-directories').value;
  }
  function stage(checkpoint) {
    ++request; // A pending directory response must not replace this choice.
    listing = checkpoint ? { checkpoint } : null;
    pathDirty = false;
    q('ah-path').value = checkpoint?.path || '';
    q('ah-up').disabled = true;
    q('ah-selection').textContent = checkpoint
      ? `待加载：${checkpoint.experiment} · step ${checkpoint.step} · harness ${checkpoint.version}`
        + (checkpoint.condition_mode === 'v5_native' ? ' · v5 原生编码器' : '')
      : '请明确选择一个 checkpoint，再点击“加载选中的 checkpoint”';
    q('ah-selection').title = checkpoint?.path || '';
    buttons();
  }
  function selectExperiment() {
    const items = catalog.filter(item => item.group === q('ah-experiment').value)
      .sort((a, b) => Number(b.step) - Number(a.step));
    q('ah-choices').hidden = false;
    q('ah-chosen').hidden = true;
    q('ah-step').replaceChildren(el('option', { value: '', text: '— 请选择 checkpoint —' }), ...items.map((item, index) => el('option', {
      value: item.path, text: `step ${item.step}${index === 0 ? '（此实验最新）' : ''} · harness ${item.version}`,
    })));
    q('ah-step').value = '';
    stage(null);
    q('ah-picked-label').textContent = items[0]?.experiment || q('ah-experiment').value;
    q('ah-picker-controls').hidden = true;
    q('ah-picked').hidden = false;
    q('ah-advanced').hidden = true;
  }
  function filterCatalog() {
    const search = q('ah-search').value.trim().toLowerCase();
    const matches = catalog.filter(item => `${item.experiment} ${item.version} ${item.condition_mode}`
      .toLowerCase().includes(search));
    const groups = new Map();
    for (const item of matches) if (!groups.has(item.group)) groups.set(item.group, item);
    const previousGroup = q('ah-experiment').value;
    q('ah-experiment').replaceChildren(...[...groups.values()].map(item => el('option', {
      value: item.group, text: item.experiment, title: item.experiment,
    })));
    if (groups.has(previousGroup)) q('ah-experiment').value = previousGroup;
    q('ah-choices').hidden = true;
    q('ah-chosen').hidden = true;
    q('ah-step').replaceChildren();
    stage(null);
    q('ah-selection').textContent = '确认实验后会列出其中的 checkpoint，尚未加载';
    if (!groups.size) q('ah-selection').textContent = '没有匹配的完整权重，请调整搜索或刷新列表';
  }
  async function scan(refresh = false) {
    const seq = ++catalogRequest;
    q('ah-refresh').disabled = true;
    q('ah-catalog-status').textContent = '正在自动查找已保存的完整权重…';
    try {
      const result = await getJSON('/action-head/catalog' + (refresh ? '?refresh=1' : ''));
      if (seq !== catalogRequest) return;
      if (!result.ok) throw new Error(result.error || '扫描失败');
      catalog = result.checkpoints;
      q('ah-catalog-status').textContent = `找到 ${new Set(catalog.map(item => item.group)).size} 个实验、${catalog.length} 个 checkpoint`
        + (result.rejected.length ? `；已过滤 ${result.rejected.length} 个不完整或不兼容的目录` : '')
        + (result.warnings.length ? '；' + result.warnings.join('；') : '');
      filterCatalog();
    } catch (error) {
      q('ah-catalog-status').textContent = '读取权重列表失败：' + String(error.message || error);
    } finally {
      if (seq === catalogRequest) q('ah-refresh').disabled = false;
    }
  }
  async function browse(path = '') {
    const seq = ++request;
    listing = null;
    q('ah-error').textContent = '';
    q('ah-selection').textContent = '正在读取目录…';
    buttons();
    try {
      const result = await getJSON('/action-head/checkpoints?path=' + encodeURIComponent(path));
      if (seq !== request) return;
      if (!result.ok) throw new Error(result.error || '读取目录失败');
      listing = result;
      pathDirty = false;
      q('ah-path').value = result.path;
      q('ah-up').disabled = !result.parent;
      q('ah-directories').replaceChildren(...result.directories.map((item) =>
        el('option', { value: item.path, text: item.name })));
      q('ah-selection').textContent = result.checkpoint
        ? '待加载：' + result.checkpoint.path + ' · ' + result.checkpoint.version
        : result.error || '进入实验及 global_step 目录，选择要加载的 checkpoint';
    } catch (error) {
      if (seq !== request) return;
      q('ah-selection').textContent = '';
      q('ah-error').textContent = String(error.message || error);
    }
    buttons();
  }
  q('ah-browse').addEventListener('click', () => browse(q('ah-path').value));
  q('ah-refresh').addEventListener('click', () => scan(true));
  q('ah-search').addEventListener('input', filterCatalog);
  q('ah-experiment').addEventListener('change', () => {
    q('ah-choices').hidden = true;
    q('ah-chosen').hidden = true;
    q('ah-step').replaceChildren();
    stage(null);
  });
  q('ah-confirm').addEventListener('click', selectExperiment);
  q('ah-change').addEventListener('click', () => {
    q('ah-picked').hidden = true;
    q('ah-chosen').hidden = true;
    q('ah-picker-controls').hidden = false;
    q('ah-advanced').hidden = false;
    q('ah-choices').hidden = true;
    q('ah-step').replaceChildren();
    stage(null);
  });
  q('ah-step').addEventListener('change', () => {
    const checkpoint = catalog.find(item => item.path === q('ah-step').value);
    stage(checkpoint);
    if (checkpoint) {
      q('ah-chosen-label').textContent = `step ${checkpoint.step} · harness ${checkpoint.version}`;
      q('ah-choices').hidden = true;
      q('ah-chosen').hidden = false;
    }
  });
  q('ah-change-step').addEventListener('click', () => {
    q('ah-chosen').hidden = true;
    q('ah-choices').hidden = false;
    q('ah-step').value = '';
    stage(null);
  });
  q('ah-advanced').addEventListener('toggle', () => {
    if (q('ah-advanced').open) browse(q('ah-path').value);
  });
  q('ah-path').addEventListener('input', () => {
    pathDirty = true;
    q('ah-selection').textContent = '目录已修改，点“打开”核对后再确认加载';
    buttons();
  });
  q('ah-path').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') { event.preventDefault(); browse(q('ah-path').value); }
  });
  q('ah-up').addEventListener('click', () => { if (listing?.parent) browse(listing.parent); });
  q('ah-enter').addEventListener('click', () => {
    if (q('ah-directories').value) browse(q('ah-directories').value);
  });
  q('ah-directories').addEventListener('change', buttons);
  q('ah-load').addEventListener('click', async () => {
    if (!listing?.checkpoint || pathDirty || loading || submitting || running) return;
    submitting = true;
    buttons();
    q('ah-error').textContent = '';
    try {
      const response = await cmd('load_action_head', { path: listing.checkpoint.path });
      if (!response) throw new Error('加载请求未发送，请检查界面锁定状态或网络');
      const result = await response.text();
      if (result !== 'ok') throw new Error(result);
      q('ah-loading').textContent = '已提交小脑加载请求…';
    } catch (error) {
      q('ah-error').textContent = String(error.message || error);
    } finally {
      submitting = false;
      buttons();
    }
  });
  return {
    root,
    show(visible) {
      root.hidden = !visible;
      if (visible && catalogRequest === 0) scan();
    },
    sync(state) {
      const head = state.action_head || {};
      loading = head.state === 'loading';
      running = ['running', 'manual', 'loading'].includes(state.status);
      q('ah-active').textContent = head.active ? '当前小脑：' + head.active.path : '小脑尚未加载';
      q('ah-loading').textContent = loading
        ? `${head.phase || '正在加载'} · 已用 ${Math.round(head.elapsed_s || 0)} 秒`
        : head.state === 'loaded' ? `上次加载完成，用时 ${Number(head.elapsed_s || 0).toFixed(1)} 秒` : '';
      if (head.encoder_state === 'loading') {
        q('ah-loading').textContent = `${head.encoder_phase} · ${Math.round(head.encoder_elapsed_s || 0)} 秒`
          + (loading ? '；所选 checkpoint 等待编码器就绪' : '；尚未加载 checkpoint');
      } else if (head.encoder_state === 'ready' && !loading && head.state === 'unloaded') {
        q('ah-loading').textContent = '默认小脑编码器已就绪，等待你选择 checkpoint';
      }
      q('ah-loading').title = head.pending?.path || '';
      // Preserve immediate command errors; a successful load clears old errors.
      if (head.error) q('ah-error').textContent = '加载失败：' + head.error;
      else if (loading) q('ah-error').textContent = '';
      buttons();
    },
  };
}

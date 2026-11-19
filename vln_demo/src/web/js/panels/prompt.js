// 提示词 -- what actually gets sent to the model, and an editor for the ReAct
// system prompt.
//
// The preview is live: the action-history block is baked into the prompt and
// changes as the robot drives, so it refreshes while a run is in progress. Two
// consequences the code has to respect:
//
//  * Never write into the editor from the refresh path. It fires on the poll and
//    would eat an edit in progress.
//  * Only refresh while the panel is actually on screen. It is a tab, so most of
//    the time it is not, and /prompt rebuilds the whole prompt each call.
//
// Editing is a ReAct-only affair: the classic driver has no system message, its
// whole prompt is the selected 提示词 template.

import { el } from '../util.js';
import { onStatus, S } from '../store.js';
import { prompt as fetchPrompt, postSysPrompt } from '../api.js';
import { on } from '../bus.js';
import { params } from '../params.js';

export function build() {
  const meta = el('span.pv-meta');
  const cust = el('span');
  const editBtn = el('button.b-sm', { text: '✎ 编辑' });
  const text = el('pre.pv-text.mono', { text: '加载中…' });
  const box = el('textarea.mono', { id: 'pv-sys', spellcheck: 'false' });
  const msg = el('span.hint.flat');
  const note = el('div.hint');

  const edit = el('div.pv-edit', { hidden: true }, [
    box,
    el('div.row.btns', {}, [
      el('button.b-go', { text: '✓ 保存并生效', onclick: () => save(box.value) }),
      el('button.b-plain', { text: '↺ 恢复默认', onclick: () => save('', true) }),
      msg,
    ]),
    el('div.hint', {
      html: '改的是 system prompt（不含下面那条 user prompt）。保存后<b>正在跑的回合不受影响</b>，'
        + '下一次「开始」生效；重启控制台也还在。<br>末尾 <code>&lt;tools&gt;</code> 那段是工具说明，'
        + '<b>删掉模型就不会调用工具</b>了。参考图/红框说明由系统按附件自动追加，不写入编辑框。'
        + '想恢复点「恢复默认」。',
    }),
  ]);

  const pane = el('div.dk-pane.dk-pad', {}, [
    el('div.pv-head', {}, [
      el('div', {}, ['发送给模型的完整提示词 ', meta, ' ', cust]),
      editBtn,
    ]),
    text, edit, note,
  ]);

  let lastSys = '';
  let editing = false;
  let sig = null;         // what the preview was rendered for

  const load = () => {
    sig = signature();
    fetchPrompt(params()).then((p) => {
      text.textContent = p.text;
      meta.textContent = (params().driver === 'react'
                          || params().driver.startsWith('action_head')
                          || ['embodied_point', 'unilavira_bbox'].includes(params().driver))
        ? p.label + ' · 每轮最多 ' + p.nact + ' 个工具调用 · 预算 ' + p.maxtools + ' 次'
        : p.label + ' · ' + p.nact + ' 动作 · 历史帧 ' + p.nhist;
      // The full preview may include a runtime visual-context suffix. Keep that
      // out of the editor or saving once would accidentally make it permanent.
      lastSys = p.sys_edit_text || p.sys_text || '';
      cust.textContent = p.sys_custom
        ? (p.sys_tools ? '· 自定义 system prompt' : '· 自定义 system prompt（⚠ 无工具说明）')
        : '';
      cust.className = p.sys_custom && !p.sys_tools ? 'warn' : 'cust';
    }).catch(() => { text.textContent = '加载失败' });
  };

  const toggleEdit = () => {
    editing = !editing;
    // Prefilled with the prompt currently in force (default or saved custom) --
    // an empty box would invite writing a prompt with no tool schemas in it.
    if (editing) box.value = lastSys;
    edit.hidden = !editing;
    text.hidden = editing;
    editBtn.textContent = editing ? '✕ 取消编辑' : '✎ 编辑';
    msg.textContent = '';
  };
  editBtn.addEventListener('click', toggleEdit);

  const save = (body, refill) => postSysPrompt(body).then((j) => {
    msg.textContent = j.custom
      ? '已生效（' + j.len + ' 字）' + (j.saved ? '' : '　⚠ 没写进磁盘：' + j.err)
        + (j.tools ? '' : '　⚠ 没有工具说明')
      : '已恢复默认';
    load();
    // 恢复默认 should leave the editor showing what is in force, not the
    // discarded text -- but the server has to answer first for lastSys to be new.
    if (refill && editing) setTimeout(() => { if (editing) box.value = lastSys }, 300);
  }).catch(() => { msg.textContent = '保存失败' });

  on('driver', (d) => {
    const react = d && d.react;
    const fm = d && d.fm;
    const lightnav = d && d.lightnav;
    if (fm) {
      note.innerHTML = 'FM 驱动在模型内部按 SocialNav 格式构造 prompt（当前画面 + 零 '
        + 'input_waypoint + 指令），不走 vLLM，无 system prompt 可改，这里只读预览。';
      editBtn.disabled = true;
      editBtn.title = 'FM 驱动没有可编辑的 system prompt';
      if (editing) toggleEdit();
      return;
    }
    if (lightnav) {
      note.innerHTML = 'LightNav-0 在独立 WebSocket 服务中按 checkpoint 配置构造输入并维护 '
        + 'SlowFast 历史；控制台只发送当前干净 RGB 与指令。这里是只读协议预览。';
      editBtn.disabled = true;
      editBtn.title = 'LightNav-0 driver 没有控制台侧可编辑的 system prompt';
      if (editing) toggleEdit();
      return;
    }
    note.innerHTML = react
      ? '第一条 user 之后，每批工具执行完的新画面会作为工具返回追加进去。'
      : '图像另外附在提示词前面（历史帧由早到近，最后一张是当前画面）。'
        + '「已执行动作」会随驾驶实时变化，这里显示的是当前值。';
    // Leave the button visible but dead, with the reason on hover, rather than
    // making it vanish.
    editBtn.disabled = !react;
    editBtn.title = react
      ? '改 ReAct 驱动的 system prompt。改完点保存立即生效，下一次「开始」就用新的'
      : '经典·多动作 没有 system prompt（只有一条 user 消息），改用「提示词」模板下拉框';
    if (!react && editing) toggleEdit();
  });
  on('cfg', () => { if (visible(pane)) load() });

  onStatus(() => {
    if (!visible(pane)) return;
    const now = signature();
    if (now !== sig) load();
  });

  return pane;
}

/** A tab that is not on top is display:none, and so has no layout boxes. */
const visible = (n) => n.isConnected && !n.hidden && n.offsetParent !== null;

/** Everything outside this panel that changes the rendered prompt. */
function signature() {
  const p = params();
  return [p.driver, p.tpl, p.nact, p.maxtools, p.instruction,
          p.reference_kind, S.action_head_as_tool_version,
          (S.actions || []).length, (S.motion || {}).steps].join('|');
}

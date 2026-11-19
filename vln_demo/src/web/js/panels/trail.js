// 动作轨迹 -- what has actually been executed.
//
// Two views of the same thing, because the two drivers move differently. The
// classic driver emits discrete 0.25 m / 30° actions, so a chip per action is
// exact. ReAct moves can be any length, and a manual hold is however long you
// held it, so there the chips carry the measured distance and the summary carries
// the totals. Manual driving reads the measured view whichever driver is
// selected.

import { el } from '../util.js';
import { onStatus } from '../store.js';

const ACTZH = { 0: '停止', 1: '前进', 2: '左转', 3: '右转' };

export function build() {
  const sum = el('div.trail-sum', { text: '尚未执行动作' });
  const chips = el('div.chips');
  const pane = el('div.dk-pane.dk-pad', {}, [sum, chips]);

  onStatus((s) => {
    if (s.driver === 'react' || s.driver.startsWith('action_head')
        || ['embodied_point', 'unilavira_bbox', 'lightnav'].includes(s.driver) || s.manual) {
      measured(sum, chips, s);
    }
    else discrete(sum, chips, s);
  });
  return pane;
}

function measured(sum, chips, s) {
  const mv = s.moves || [];
  const m = s.motion || { dist: 0, yaw: 0, steps: 0 };
  if (!mv.length && !m.steps) return empty(sum, chips);
  sum.innerHTML = '累计前进 <b>' + m.dist.toFixed(2) + '</b> m · 净转向 <b>'
    + m.yaw.toFixed(0) + '</b>° · 仿真步 <b>' + m.steps + '</b> · '
    + (s.manual ? '操作' : (s.driver === 'lightnav' ? '策略' : '工具'))
    + ' <b>' + mv.length + '</b> 次';
  chips.innerHTML = mv.map((x) => x.kind === 'fwd'
    ? '<span class="chip ' + (x.blocked ? 'c0' : 'c1') + '">前进 ' + x.m.toFixed(2) + 'm</span>'
    : x.kind === 'back'
    ? '<span class="chip ' + (x.blocked ? 'c0' : 'c4') + '">后退 ' + x.m.toFixed(2) + 'm</span>'
    : x.kind === 'point' || x.kind === 'bbox'
    ? '<span class="chip ' + (x.blocked ? 'c0' : 'c1') + '">'
      + (x.kind === 'point' ? '点导航 ' : '框导航 ') + x.m.toFixed(2) + 'm</span>'
    : x.kind === 'lightnav'
    ? '<span class="chip ' + (x.blocked ? 'c0' : 'c1') + '">LightNav '
      + x.m.toFixed(2) + 'm / ' + Number(x.yaw_deg || 0).toFixed(0) + '°'
      + (Math.abs(Number(x.lateral_m || 0)) > 0.001
        ? ' · 丢侧移 ' + Number(x.lateral_m).toFixed(2) + 'm' : '') + '</span>'
    : '<span class="chip ' + (x.dir === 'left' ? 'c2' : 'c3') + '">'
      + (x.dir === 'left' ? '左转 ' : '右转 ') + x.deg.toFixed(0) + '°</span>').join(' ');
}

function discrete(sum, chips, s) {
  const a = s.actions || [];
  if (!a.length) return empty(sum, chips);
  sum.innerHTML = '共 <b>' + a.length + '</b> 步 · 前进 <b>' + s.st.fwd + '</b>'
    + ' · 左转 <b>' + s.st.left + '</b> · 右转 <b>' + s.st.right + '</b> · 累计前进 <b>'
    + s.st.dist.toFixed(2) + '</b> m · 净转向 <b>' + s.st.yaw.toFixed(0) + '</b>°';
  chips.innerHTML = a
    .map((x) => '<span class="chip c' + x + '">' + ACTZH[x] + '</span>').join(' ');
}

function empty(sum, chips) {
  sum.textContent = '尚未执行动作';
  chips.replaceChildren();
}

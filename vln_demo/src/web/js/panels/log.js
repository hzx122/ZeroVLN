// 模型日志 -- model inputs, streaming output, tools and observations.
//
// Events arrive as a delta keyed by `since` (store.js owns that counter), so this
// panel only ever appends. The live box at the bottom is the one exception: it is
// rewritten every poll while the model is streaming, then removed when the turn
// lands as a real event.

import { el, esc } from '../util.js';
import { onStatus, resetSince } from '../store.js';
import { cmd } from '../api.js';

const ACTZH = { 0: '停止', 1: '前进', 2: '左转', 3: '右转' };

const chips = (as) => (as || [])
  .map((a) => '<span class="chip c' + a + '">' + ACTZH[a] + '</span>').join(' ');

// SGLang exposes reasoning_content separately. Some vLLM templates instead
// leave an in-band <think> block in content. Normalize both for one UI.
function modelParts(text, explicitReasoning) {
  text = String(text || '');
  if (explicitReasoning) return { reasoning: explicitReasoning, text };
  const close = text.indexOf('</think>');
  if (close >= 0) {
    return {
      reasoning: text.slice(0, close).replace(/^\s*<think>\s*/, '').trim(),
      text: text.slice(close + 8).trim(),
    };
  }
  if (/^\s*<think>/.test(text)) {
    return { reasoning: text.replace(/^\s*<think>\s*/, ''), text: '' };
  }
  return { reasoning: '', text };
}

function modelBody(text, reasoning, live) {
  const parts = modelParts(text, reasoning);
  let html = '';
  if (parts.reasoning) {
    html += '<div class="model-section-label">思维链</div>'
      + '<div class="t reasoning">' + esc(parts.reasoning) + '</div>';
  }
  if (parts.text || !parts.reasoning) {
    html += '<div class="model-section-label">最终输出</div>'
      + '<div class="t final">' + esc(parts.text || '(空响应)')
      + (live ? '<span class="cursor"></span>' : '') + '</div>';
  } else if (live) {
    html += '<span class="cursor"></span>';
  }
  return html;
}

// "前进 0.8 m" reads better in the log than the raw JSON arguments, but fall
// back to JSON for anything unexpected rather than dropping the arguments.
function toolLabel(e) {
  const a = e.args || {};
  if (e.name === 'move_forward' && a.distance != null) return '前进 ' + a.distance + ' m';
  if (e.name === 'move_backward' && a.distance != null) return '后退 ' + a.distance + ' m';
  if (e.name === 'move_to_point' && a.point_2d) {
    return 'point_2d ' + JSON.stringify(a.point_2d);
  }
  if (e.name === 'move_to_bbox' && a.bbox_2d) {
    return 'bbox_2d ' + JSON.stringify(a.bbox_2d);
  }
  if (e.name === 'turn' && a.degrees != null) {
    const d = a.direction === 'right' ? '右转 '
            : a.direction === 'left' ? '左转 ' : a.direction + ' ';
    return d + a.degrees + '°';
  }
  if (e.name === 'stop') return '停止' + (a.reason ? '（' + a.reason + '）' : '');
  return JSON.stringify(a);
}

function navResultSummary(text) {
  let x;
  try { x = JSON.parse(String(text || '').split('\n', 1)[0]); } catch (_) { return null; }
  if (!x || !['move_to_point', 'move_to_bbox'].includes(x.tool)) return null;
  const fmt = (v) => Number.isFinite(Number(v)) ? Number(v).toFixed(3) : '?';
  const reached = x.reason === 'goal_radius';
  return el('div.nav-result ' + (reached ? 'reached' : 'not-reached'), {}, [
    el('div', { text: (reached ? '✓ 已到达底层目标半径' : '⚠ 未到达底层目标')
      + ' · 终点误差 ' + fmt(x.final_goal_error_m) + 'm'
      + ' / 半径 ' + fmt(x.goal_radius_m) + 'm' }),
    el('div', { text: '视觉投影→可走区域偏移 ' + fmt(x.snap_offset_m) + 'm'
      + ' · 动态重规划 ' + Number(x.replans || 0) + ' 次'
      + ' · ' + String(x.planner || '?') }),
  ]);
}

function fillRequest(body, request, onGrow) {
  body.replaceChildren();
  for (const message of request.messages || []) {
    const card = el('div.req-message');
    card.appendChild(el('div.req-role', {
      text: '#' + message.index + ' · ' + String(message.role || 'unknown'),
    }));
    for (const part of message.parts || []) {
      if (part.type === 'image') {
        const image = el('img.req-image', {
          onload: onGrow, onerror: onGrow,
          src: part.url, alt: '模型输入图片', loading: 'lazy',
        });
        card.appendChild(el('a.req-image-link', {
          href: part.url, target: '_blank', rel: 'noopener',
          title: '点击打开原尺寸调试快照',
        }, [image, el('span', { text: part.width + '×' + part.height })]));
      } else {
        card.appendChild(el('pre.req-text.mono', { text: part.text || '' }));
      }
    }
    body.appendChild(card);
  }
  onGrow();
}

function renderRequest(e, d, onGrow) {
  d.className = 'ev request';
  const golden = e.driver === 'golden';
  const body = el('div.req-body', { text: '加载中…' });
  const fetchBody = () => fetch('/debug-request/' + e.request_id)
    .then((response) => response.json())
    .then((reply) => {
      if (!reply || !reply.ok) throw new Error((reply && reply.err) || '加载失败');
      fillRequest(body, reply.request, onGrow);
    })
    .catch((error) => {
      body.textContent = '加载失败：' + error.message;
      onGrow();
    });

  // 远端主线约定普通模型输入直接展开、不可折叠，保证输入/输出按时间线连续
  // 阅读。专家轨迹每 0.1m 都产生一帧，才使用下面的按需折叠以免批量解码图。
  if (!golden) {
    d.appendChild(el('div.h.req-title', {
      text: '第 ' + e.turn + ' 轮模型输入 · ' + e.message_count + ' 条消息 · '
        + e.image_count + ' 张图片',
    }));
    d.appendChild(body);
    fetchBody();
    return;
  }

  const details = el('details.req-details', {}, [
    el('summary', {
      text: '专家轨迹第 ' + e.turn + ' 步 · ' + e.message_count + ' 条消息 · '
        + e.image_count + ' 张图片',
    }),
    body,
  ]);
  let loaded = false;
  const load = () => {
    if (!details.open || loaded) return;
    loaded = true;
    body.textContent = '加载中…';
    fetchBody();
  };
  details.addEventListener('toggle', load);
  // 专家帧默认折叠，用户展开哪一步才加载哪张图。
  details.open = false;
  d.appendChild(details);
  onGrow();
}

function render(e, onGrow) {
  const d = el('div');
  if (e.t === 'instr') {
    d.className = 'ev instr';
    d.innerHTML = '<div class="h">指令' + (e.frompose ? ' · 从当前位置' : ' · 从回合起点')
      + ' · 模板 ' + esc(e.tpl) + '</div><div class="t">' + esc(e.text) + '</div>';
  } else if (e.t === 'request') {
    renderRequest(e, d, onGrow);
  } else if (e.t === 'model') {
    d.className = 'ev model';
    d.innerHTML = '<div class="h">' + (e.turn ? '第 ' + e.turn + ' 轮模型输出' : '模型推理')
      + ' · ' + e.dt.toFixed(1) + 's'
      + (e.abort ? ' · 已中断' : '') + '</div>'
      + modelBody(e.text, e.reasoning, false)
      + '<div class="acts">' + chips(e.actions) + '</div>';
  } else if (e.t === 'tool') {
    d.className = 'ev tool';
    d.innerHTML = '<div class="h">工具调用</div><div class="t">🔧 ' + esc(e.name)
      + ' · ' + esc(toolLabel(e)) + '</div>';
    if (e.visual && e.visual.type === 'image') {
      const image = el('img.req-image', {
        onload: onGrow, onerror: onGrow,
        src: e.visual.url, alt: '工具输出位置可视化', loading: 'lazy',
      });
      d.appendChild(el('a.req-image-link', {
        href: e.visual.url, target: '_blank', rel: 'noopener',
        title: '点击打开原尺寸标注图',
      }, [image, el('span', {
        text: e.visual.width + '×' + e.visual.height
          + ' · 红色=模型输出，黄色=实际深度采样点；执行路径可能绕障',
      })]));
    }
  } else if (e.t === 'obs') {
    // first line is the compact JSON, anything after it is the plain-language
    // note -- worth pulling out in amber since that is what the model acts on
    const lines = (e.text || '').split('\n');
    const head = lines.shift();
    d.className = 'ev obs';
    d.innerHTML = '<div class="t mono">' + esc(head)
      + (lines.length ? '<span class="note">' + esc(lines.join(' ')) + '</span>' : '')
      + '</div>';
    const nav = navResultSummary(head);
    if (nav) d.appendChild(nav);
  } else if (e.t === 'steps') {
    d.className = 'ev';
    d.innerHTML = '<div class="steps">执行 ' + esc(e.text) + '</div>';
  } else if (e.t === 'done') {
    d.className = 'ev done';
    d.innerHTML = '<div class="h">结束</div><div class="t">' + esc(e.text) + '</div>';
  } else if (e.t === 'err') {
    d.className = 'ev err';
    d.innerHTML = '<div class="h">错误</div><div class="t">' + esc(e.text) + '</div>';
  } else {
    d.className = 'ev sys';
    d.innerHTML = '<div class="t">' + esc(e.text) + '</div>';
  }
  return d;
}

export function build() {
  const box = el('div.dk-pane.log-box');
  let live = null;
  let liveTurn = null;
  let followBottom = true;
  let scrollFrame = null;
  let lastEvent = -1;

  const atBottom = () => box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
  const pinBottom = () => {
    if (!followBottom || scrollFrame != null) return;
    scrollFrame = window.requestAnimationFrame(() => {
      scrollFrame = null;
      if (followBottom) box.scrollTop = box.scrollHeight;
    });
  };
  box.addEventListener('scroll', () => { followBottom = atBottom() });

  onStatus((s) => {
    // Keep following only while the reader stays at the bottom. Async model
    // input bodies and images call pinBottom again after they increase height.
    if (s.reset_log) {
      box.replaceChildren(); live = null; liveTurn = null; followBottom = true; lastEvent = -1;
    }
    for (const e of s.events || []) {
      if (Number.isInteger(e.i)) {
        if (e.i <= lastEvent) continue;
        lastEvent = e.i;
      }
      box.appendChild(render(e, pinBottom));
      if (e.t === 'request') liveTurn = e.turn;
    }
    if (live) box.appendChild(live);        // a move: the live box stays last

    if (s.stream && s.stream.active) {
      if (!live) { live = el('div.ev.model.live'); box.appendChild(live) }
      live.innerHTML = '<div class="h">' + (liveTurn ? '第 ' + liveTurn + ' 轮模型输出中' : '模型推理中')
        + ' · ' + s.stream.dt.toFixed(1) + 's · '
        + s.stream.n + ' chunk</div>'
        + modelBody(s.stream.text, s.stream.reasoning, true);
    } else if (live) { live.remove(); live = null }

    pinBottom();
  });

  const clear = el('button', { text: '清空', title: '清掉日志（不影响仿真）' });
  clear.addEventListener('click', () => {
    cmd('clearlog').then(() => {
      box.replaceChildren(); resetSince(); live = null; followBottom = true; pinBottom();
    });
  });

  return { el: box, tools: [clear] };
}

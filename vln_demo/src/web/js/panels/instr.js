// 导航指令 -- instruction text, optional visual reference, and run buttons.
//
// A reference belongs to the instruction that is about to run, so it lives in
// this composer instead of a separate dock panel. Images are pasted directly
// from the clipboard. A bbox can be typed as four 0-to-1000 normalized coordinates, or drawn on
// a snapshot of the current first-person frame. The selected reference is sent
// immediately before /cmd?do=start, so the model sees the matching pair.

import { el, clamp } from '../util.js';
import { onStatus, S } from '../store.js';
import { cmd, getFrameSnapshot, postReactReference } from '../api.js';
import { params } from '../params.js';
import { emit } from '../bus.js';
import { pictureRect } from './map.js';

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const BBOX_MAX = 1000;

// The console is served over HTTP on an internal IP, where the modern
// Clipboard API is often unavailable. Keep the user-gesture copy fallback.
export async function copyInstruction(box, text) {
  if (globalThis.navigator?.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return true; } catch (_) {}
  }
  const value = box.value;
  const start = box.selectionStart, end = box.selectionEnd;
  box.value = text;
  box.focus();
  box.select();
  try { return Boolean(document.execCommand && document.execCommand('copy')); }
  catch (_) { return false; }
  finally {
    box.value = value || text;
    if (value && box.setSelectionRange) box.setSelectionRange(start, end);
  }
}

function clipboardImage(data) {
  if (!data) return null;
  const files = Array.from(data.files || []);
  const direct = files.find((f) => !f.type || f.type.startsWith('image/'));
  if (direct) return direct;
  for (const item of Array.from(data.items || [])) {
    if (item.kind === 'file' && (!item.type || item.type.startsWith('image/'))) {
      const file = item.getAsFile && item.getAsFile();
      if (file) return file;
    }
  }
  return null;
}

function bboxValue(inputs) {
  const raw = inputs.map((input) => String(input.value || '').trim());
  if (raw.every((v) => !v)) return null;
  if (raw.some((v) => !v)) throw new Error('bbox 需要填写四个坐标');
  const box = raw.map(Number);
  if (!box.every((v) => Number.isInteger(v) && v >= 0 && v <= BBOX_MAX)) {
    throw new Error('bbox 必须是四个 0–1000 的归一化整数坐标');
  }
  if (box[0] >= box[2] || box[1] >= box[3]) {
    throw new Error('bbox 必须满足 x1<x2、y1<y2');
  }
  return box;
}

export function build() {
  const box = el('textarea', {
    id: 'instr', spellcheck: 'false',
    placeholder: '输入导航指令；可在这里直接 Ctrl+V / ⌘V 粘贴参考图片',
  });
  const thumb = el('img', {
    id: 'react-ref-thumb', class: 'instr-ref-thumb', alt: '参考图', hidden: true,
  });
  const state = el('span.hint.flat.instr-ref-state', {
    id: 'react-ref-state', text: '可选：在指令框粘贴图片，或填写 bbox',
  });
  const clear = el('button.b-plain.b-sm', {
    id: 'react-ref-clear', type: 'button', text: '清除附件',
  });
  const names = ['x1', 'y1', 'x2', 'y2'];
  const bboxInputs = names.map((name) => el('input.mono.bbox-coordinate', {
    id: 'react-ref-' + name, type: 'number', min: 0, max: BBOX_MAX,
    step: 1, placeholder: name, title: name + '（0–1000；x 向右，y 向下）',
  }));
  const draw = el('button.b-plain.b-sm', {
    id: 'react-ref-draw', type: 'button', text: '在当前画面框选',
  });

  const pickerImage = el('img', {
    id: 'bbox-picker-image', alt: '当前第一视角画面', draggable: 'false',
  });
  const drawnBox = el('span.bbox-drawn', { id: 'bbox-drawn', hidden: true });
  const pickerStage = el('div.bbox-picker-stage', { id: 'bbox-picker-stage' }, [
    pickerImage, drawnBox,
  ]);
  const pickerCoords = el('span.mono', {
    id: 'bbox-picker-coords', text: '在画面中按住并拖动鼠标',
  });
  const closePicker = el('button.b-plain', {
    id: 'bbox-picker-close', type: 'button', text: '完成',
  });
  const picker = el('div.bbox-picker', { id: 'bbox-picker', hidden: true }, [
    el('div.bbox-picker-card', {}, [
      el('div.bbox-picker-head', {}, [
        el('b', { text: '框选当前画面' }), pickerCoords, closePicker,
      ]),
      pickerStage,
    ]),
  ]);

  const composer = el('div.instr-compose', { id: 'instr-compose' }, [
    box,
    el('div.instr-ref-line', {}, [thumb, state, clear]),
    el('div.instr-bbox-line', {}, [
      el('span.hint.flat', {
        text: 'bbox（0–1000；左上原点，x→右，y→下）',
      }), ...bboxInputs, draw,
    ]),
  ]);
  const start = el('button.b-go', { id: 'instr-start', text: '▶ 开始' });
  const copy = el('button.b-plain', { id: 'instr-copy', text: '复制指令', type: 'button' });
  copy.addEventListener('click', async () => {
    const text = box.value.trim() ? box.value : (S.instruction || '');
    if (!text) { copy.textContent = '暂无指令'; return; }
    const copied = await copyInstruction(box, text);
    copy.textContent = copied ? '已复制' : '请选中文字后按 Ctrl+C / ⌘C';
    setTimeout(() => { copy.textContent = '复制指令'; }, 2500);
  });
  const go = (action) => () => cmd(action, params());
  const walkGolden = () => {
    if (S.backend !== 'sage') {
      window.alert('走专家轨迹功能只支持 SAGE');
      return;
    }
    return cmd('walk_golden', params());
  };
  const pane = el('div.dk-pane.dk-pad', {}, [
    composer,
    el('div.row.btns', {}, [
      start, copy,
      el('button.b-stop', { text: '■ 停止', onclick: go('stop') }),
      el('button.b-man', { text: '✋ 手动', onclick: go('manual_start') }),
      el('button.b-plain', {
        id: 'walk-golden', text: '🧭 走专家轨迹', onclick: walkGolden,
        title: '仅 SAGE 后端支持',
      }),
      el('button.b-plain', { text: '↺ 回到起点', onclick: go('reset') }),
    ]),
    picker,
  ]);

  let upload = null;
  let reading = false;
  let localChoice = null; // null follows server; upload/bbox/clear is a UI choice
  let bboxSource = null;  // atomic /frame-snapshot token for a drawn bbox
  let react = true;
  let starting = false;

  composer.dataset.referenceKind = 'none';
  const setReferenceKind = (kind) => {
    const next = ['upload', 'point', 'bbox'].includes(kind) ? kind : 'none';
    if (composer.dataset.referenceKind === next) return;
    composer.dataset.referenceKind = next;
    emit('cfg');
  };

  const clearUpload = () => {
    upload = null;
    thumb.hidden = true;
    thumb.removeAttribute('src');
  };
  const clearBBox = () => {
    for (const input of bboxInputs) input.value = '';
    bboxSource = null;
  };
  const setMessage = (text, bad = false) => {
    state.textContent = text;
    state.classList.toggle('bad', bad);
  };

  function selectImage(picked) {
    if (!picked) return;
    if (picked.type && !picked.type.startsWith('image/')) {
      setMessage('只能添加图片文件', true);
      return;
    }
    if (!picked.size || picked.size > MAX_IMAGE_BYTES) {
      setMessage('图片必须大于 0 且不超过 8 MB', true);
      return;
    }
    reading = true;
    setMessage('正在读取图片…');
    const reader = new FileReader();
    reader.onload = () => {
      reading = false;
      upload = {
        kind: 'upload', name: picked.name || '粘贴图片',
        image_data: String(reader.result || ''),
      };
      clearBBox();
      localChoice = 'upload';
      setReferenceKind('upload');
      thumb.src = upload.image_data;
      thumb.hidden = false;
      setMessage('图片：' + upload.name + '（点开始时一起发送）');
    };
    reader.onerror = () => {
      reading = false;
      setMessage('读取图片失败', true);
    };
    reader.readAsDataURL(picked);
  }

  box.addEventListener('paste', (event) => {
    const image = clipboardImage(event.clipboardData);
    if (image) {
      event.preventDefault();
      selectImage(image);
    }
  });
  box.addEventListener('input', () => emit('cfg'));

  const bboxEdited = () => {
    bboxSource = null;
    if (bboxInputs.every((input) => !String(input.value || '').trim())) {
      if (localChoice === 'bbox') localChoice = 'clear';
      setReferenceKind(null);
      setMessage('无视觉附件');
      return;
    }
    clearUpload();
    localChoice = 'bbox';
    setReferenceKind('bbox');
    try {
      setMessage('bbox ' + JSON.stringify(bboxValue(bboxInputs)) + '（点开始时一起发送）');
    } catch (error) {
      setMessage(error.message, true);
    }
  };
  for (const input of bboxInputs) input.addEventListener('input', bboxEdited);

  clear.addEventListener('click', () => {
    clearUpload();
    clearBBox();
    localChoice = 'clear';
    setReferenceKind(null);
    setMessage('无视觉附件');
    postReactReference({ kind: 'clear' }).catch(() => {});
  });

  function pickerPoint(event) {
    const picture = pictureRect(pickerImage);
    if (!picture.width || !picture.height) return null;
    const x = clamp(event.clientX, picture.left, picture.left + picture.width);
    const y = clamp(event.clientY, picture.top, picture.top + picture.height);
    return {
      x, y,
      qx: Math.round((x - picture.left) * BBOX_MAX / picture.width),
      qy: Math.round((y - picture.top) * BBOX_MAX / picture.height),
    };
  }

  let dragStart = null;
  const paintBox = (a, b) => {
    const stage = pickerStage.getBoundingClientRect();
    drawnBox.style.left = Math.min(a.x, b.x) - stage.left + 'px';
    drawnBox.style.top = Math.min(a.y, b.y) - stage.top + 'px';
    drawnBox.style.width = Math.abs(a.x - b.x) + 'px';
    drawnBox.style.height = Math.abs(a.y - b.y) + 'px';
    drawnBox.hidden = false;
    pickerCoords.textContent = JSON.stringify([
      Math.min(a.qx, b.qx), Math.min(a.qy, b.qy),
      Math.max(a.qx, b.qx), Math.max(a.qy, b.qy),
    ]);
  };

  pickerStage.addEventListener('pointerdown', (event) => {
    if (event.button != null && event.button !== 0) return;
    dragStart = pickerPoint(event);
    if (!dragStart) return;
    if (pickerStage.setPointerCapture && event.pointerId != null) {
      pickerStage.setPointerCapture(event.pointerId);
    }
    paintBox(dragStart, dragStart);
    if (event.preventDefault) event.preventDefault();
  });
  pickerStage.addEventListener('pointermove', (event) => {
    if (!dragStart) return;
    const point = pickerPoint(event);
    if (point) paintBox(dragStart, point);
  });
  pickerStage.addEventListener('pointerup', (event) => {
    if (!dragStart) return;
    const end = pickerPoint(event);
    const begin = dragStart;
    dragStart = null;
    if (!end) return;
    paintBox(begin, end);
    const values = [Math.min(begin.qx, end.qx), Math.min(begin.qy, end.qy),
                    Math.max(begin.qx, end.qx), Math.max(begin.qy, end.qy)];
    if (values[0] === values[2] || values[1] === values[3]) {
      pickerCoords.textContent = '框太小，请重新拖动';
      return;
    }
    values.forEach((value, i) => { bboxInputs[i].value = value; });
    clearUpload();
    localChoice = 'bbox';
    setReferenceKind('bbox');
    setMessage('bbox ' + JSON.stringify(values) + '（点开始时一起发送）');
  });

  draw.addEventListener('click', async () => {
    try {
      const snapshot = await getFrameSnapshot();
      if (!snapshot || !snapshot.ok || !snapshot.image_data || !snapshot.source) {
        throw new Error((snapshot && snapshot.err) || '无法读取当前第一视角画面');
      }
      bboxSource = snapshot.source;
      pickerImage.src = snapshot.image_data;
      drawnBox.hidden = true;
      pickerCoords.textContent = '在画面中按住并拖动鼠标';
      picker.hidden = false;
    } catch (error) {
      bboxSource = null;
      setMessage(error.message, true);
    }
  });
  closePicker.addEventListener('click', () => { picker.hidden = true; });
  picker.addEventListener('click', (event) => {
    if (event.target === picker) picker.hidden = true;
  });

  async function syncReference() {
    if (reading) throw new Error('图片仍在读取，请稍等');
    let payload = null;
    if (localChoice === 'upload' && upload) payload = upload;
    else if (localChoice === 'bbox') {
      payload = { kind: 'bbox', bbox: bboxValue(bboxInputs) };
      if (bboxSource) payload.source = bboxSource;
    }
    else if (localChoice === 'clear') payload = { kind: 'clear' };
    if (!payload) return;
    const reply = await postReactReference(payload);
    if (!reply || !reply.ok) throw new Error((reply && reply.err) || '视觉附件发送失败');
  }

  start.addEventListener('click', async () => {
    if (starting) return;
    starting = true;
    start.disabled = true;
    try {
      if (react) {
        setMessage('正在发送视觉附件…');
        await syncReference();
      }
      await cmd('start', params());
      if (localChoice === 'upload' && upload) setMessage('已发送图片：' + upload.name);
      else if (localChoice === 'bbox') setMessage('已发送 bbox ' + JSON.stringify(bboxValue(bboxInputs)));
      else setMessage('已开始');
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      starting = false;
      start.disabled = false;
    }
  });

  let lastKey = '';
  let lastInstruction = '';
  let instructionEdited = false;
  box.addEventListener('input', () => { instructionEdited = true; });
  onStatus((status) => {
    react = status.driver === 'react' || status.driver.startsWith('action_head')
      || ['embodied_point', 'unilavira_bbox'].includes(status.driver);
    for (const control of [clear, draw, ...bboxInputs]) control.disabled = !react;
    composer.classList.toggle('off', !react);
    box.placeholder = '这个场景没有自带描述，自己写一条指令（也可以直接手动驾驶）';
    const key = status.scene + '/' + status.episode_id;
    const instruction = status.instruction || '';
    if (key !== lastKey) {
      lastKey = key;
      if (document.activeElement !== box || !instructionEdited) {
        box.value = instruction;
        instructionEdited = false;
      }
    } else if (!instructionEdited && instruction !== lastInstruction) {
      box.value = instruction;
    }
    lastInstruction = instruction;
    if (!react) {
      setMessage('视觉附件仅 ReAct 驱动可用');
    } else if (localChoice == null) {
      const ref = status.react_reference || {};
      if (ref.active) {
        setReferenceKind(ref.kind);
        thumb.src = '/react-reference.jpg?v=' + (ref.version || Date.now());
        thumb.hidden = false;
        setMessage(ref.kind === 'point'
          ? '当前落脚 point ' + JSON.stringify(ref.point)
          : ref.kind === 'bbox'
            ? '当前 bbox ' + JSON.stringify(ref.bbox)
            : '当前图片：' + (ref.name || '参考图'));
      } else {
        setReferenceKind(null);
        thumb.hidden = true;
        setMessage('可选：在指令框粘贴图片，或填写 bbox');
      }
    }
  });

  return pane;
}

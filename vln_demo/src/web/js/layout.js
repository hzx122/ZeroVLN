// Which panels exist, and where they start out.
//
// The default is the layout the console is designed around: the two views side by
// side on top, the model's output underneath them, and the controls in a column
// down the right. Everything after that is the user's business -- the dock
// remembers what they do to it.

import * as view from './panels/view.js';
import * as map from './panels/map.js';
import * as log from './panels/log.js';
import * as instr from './panels/instr.js';
import * as manual from './panels/manual.js';
import * as trail from './panels/trail.js';
import * as promptPanel from './panels/prompt.js';
import * as terminal from './panels/terminal.js';
import * as controls from './controls.js';

const SPEC = [
  ['view', '第一视角', view],
  ['map', '俯视地图', map],
  ['log', '模型日志', log],
  ['prompt', '提示词', promptPanel],
  ['run', '任务设置', controls],
  ['instr', '导航指令', instr],
  ['terminal', '终端', terminal],
  ['manual', '手动控制', manual],
  ['trail', '动作轨迹', trail],
];

/** Build every panel once. A panel returns either its element or {el, tools}. */
export function buildPanels() {
  return SPEC.map(([id, title, mod]) => {
    const out = mod.build();
    const node = out && out.el ? out.el : out;
    return { id, title, el: node, tools: (out && out.tools) || [] };
  });
}

export const DEFAULT = {
  t: 'split', dir: 'row', sizes: [70, 30],
  kids: [
    { t: 'split', dir: 'col', sizes: [58, 42],
      kids: [
        { t: 'split', dir: 'row', sizes: [52, 48],
          kids: [
            { t: 'tabs', ids: ['view'], active: 0 },
            { t: 'tabs', ids: ['map'], active: 0 },
          ] },
        { t: 'tabs', ids: ['log', 'prompt'], active: 0 },
      ] },
    // Down the right, sized by what each panel actually needs rather than
    // evenly: 任务设置 is a dozen rows of dropdowns, 导航指令 is a box you type
    // several lines into (18% used to leave it about two lines tall once the
    // four run buttons had taken their share), 手动控制 is a fixed 2x44 pad plus
    // its hint, and 动作轨迹 is a scrolling chip list that reads fine short.
    { t: 'split', dir: 'col', sizes: [40, 27, 20, 13],
      kids: [
        { t: 'tabs', ids: ['run'], active: 0 },
        { t: 'tabs', ids: ['terminal', 'instr'], active: 0 },
        { t: 'tabs', ids: ['manual'], active: 0 },
        { t: 'tabs', ids: ['trail'], active: 0 },
      ] },
  ],
};

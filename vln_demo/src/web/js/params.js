// What /cmd and /prompt need from the form, read straight off the DOM.
//
// Read by id rather than passed around because the controls and the buttons that
// use them now live in different panels, and either can be dragged anywhere in
// the dock. The ids are the contract between them.

import { $ } from './util.js';

const val = (id, d) => ($(id) ? $(id).value : d);

export function params() {
  return {
    instruction: val('instr', ''),
    model: val('model', ''),
    driver: val('driver', ''),
    tpl: val('tpl', ''),
    mode: val('mode', ''),
    nact: val('nact', 4),
    maxtools: val('maxtools', 10),
    temp: val('temp', 'auto'),
    frompose: $('frompose') && $('frompose').checked ? 1 : 0,
    reference_kind: (($('instr-compose') || {}).dataset || {}).referenceKind || 'none',
  };
}

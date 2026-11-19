// Light or dark, remembered.
//
// Light is the default because that is what a browser with no stored preference
// gets from tokens.css; nothing here runs on the first paint (index.html has a
// three-line inline script for that) so this module only handles the toggle.

import { ls } from './util.js';

const KEY = 'vln.theme';

export function current() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

export function apply(name) {
  const t = name === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = t;
  ls.set(KEY, t);
  return t;
}

export function initTheme(btn) {
  const paint = () => {
    const dark = current() === 'dark';
    // The button shows what you would switch *to*, which is the convention
    // every OS toggle uses.
    btn.textContent = dark ? '☀' : '🌙';
    btn.title = dark ? '切换到浅色' : '切换到深色';
  };
  btn.addEventListener('click', () => { apply(current() === 'dark' ? 'light' : 'dark'); paint() });
  paint();
}

// A private event bus, so panels can react to each other without importing each
// other. Two events, both fired by controls.js:
//
//   cfg    -- a control that changes the prompt moved (driver, template, nact,
//             tool budget). The prompt panel reloads its preview.
//   driver -- the driver changed; panels that only make sense for one driver
//             grey themselves out.
//
// A DOM node rather than a Map of handler lists: CustomEvent already does the
// dispatch, and a detached node keeps these off document where a stray listener
// elsewhere could see them.

const hub = document.createElement('div');

export function emit(name, detail) {
  hub.dispatchEvent(new CustomEvent(name, { detail }));
}

export function on(name, fn) {
  hub.addEventListener(name, (e) => fn(e.detail));
}

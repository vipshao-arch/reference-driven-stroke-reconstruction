/** Tiny always-on diagnostic module used to detect module execution ordering. */

window.__strokeLab = { ready: false, stage: 'module-start' };

export function setStage(stage) {
  window.__strokeLab.stage = stage;
}

export function setSnapshot(snapshot) {
  window.__strokeLab = { ...snapshot, ready: true };
}

export function setFatal(message) {
  window.__strokeLab = { ...window.__strokeLab, ready: false, fatal: message };
}

import { vec2, vec3 } from 'gl-matrix';
import {
  move,
  pressKey,
  releaseAllKeys,
  releaseKey,
  rotateCamera,
  store,
} from './store';

let released = true;
let releasedLock = true;
const canvas = document.getElementById('canvas');
const releaseLock = () => {
  document.exitPointerLock();
  releasedLock = true;
  releaseAllKeys();
};
const skipEvents = () => releasedLock || released;

canvas.addEventListener('contextmenu', (e) => {
  if (!releasedLock) {
    e.preventDefault();
    e.stopPropagation();
    releaseLock();
    return false;
  }
});

canvas.addEventListener('pointerdown', (e) => {
  if (e.pointerType !== 'mouse') {
    return;
  }
  e.preventDefault();
  e.stopPropagation();

  // left click to capture mouse
  if (e.button === 0) {
    canvas.requestPointerLock({ unadjustedMovement: true });
    releasedLock = false;
    released = false;
  }

  // right click to release mouse
  if (e.button === 2) {
    document.exitPointerLock();
    released = true;
  }
});

canvas.addEventListener('pointermove', (e) => {
  if (skipEvents()) return;

  const d = vec2.fromValues(e.movementX, e.movementY);
  rotateCamera(d);
});

window.addEventListener('keydown', (e) => {
  if (skipEvents()) return;
  e.preventDefault();

  pressKey(e.code);
});

window.addEventListener('keyup', (e) => {
  if (skipEvents()) return;
  releaseKey(e.code);
});

window.addEventListener('blur', () => {
  releaseLock();
});

export const handleControls = () => {
  if (releasedLock) return;
  const d = vec3.fromValues(0, 0, 0);
  const keys = store.keyboard;

  if (keys.includes('ArrowUp') || keys.includes('KeyW')) {
    d[2] += 1;
  }
  if (keys.includes('ArrowDown') || keys.includes('KeyS')) {
    d[2] -= 1;
  }
  if (keys.includes('ArrowLeft') || keys.includes('KeyA')) {
    d[0] -= 1;
  }
  if (keys.includes('ArrowRight') || keys.includes('KeyD')) {
    d[0] += 1;
  }
  if (keys.includes('Space')) {
    d[1] -= 1;
  }
  if (keys.includes('ControlLeft')) {
    d[1] += 1;
  }
  vec3.normalize(d, d);
  if (keys.includes('ShiftLeft')) {
    vec3.scale(d, d, store.runSpeed);
  }

  move(d);
};

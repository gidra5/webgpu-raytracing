export const init = () => {
  let released = true;
  const canvas = document.getElementById('canvas');

  canvas.addEventListener('contextmenu', (e) => {
    if (!released) {
      e.preventDefault();
      e.stopPropagation();
      document.exitPointerLock();
      released = true;
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
      released = false;
    }

    // right click to release mouse
    if (e.button === 2) {
      document.exitPointerLock();
    }
  });
};

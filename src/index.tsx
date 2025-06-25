import 'uno.css';
import { render } from 'solid-js/web';

import UI from './UI';
import { init as initControls } from './controls';
import { init as initRender } from './render';
import { setTime } from './store';

const controlsLoop = initControls();
const renderLoop = await initRender();

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?'
  );
}

render(() => <UI />, root);

function update() {
  const now = performance.now();
  setTime(now);

  renderLoop(now);
  controlsLoop();
  requestAnimationFrame(update);
}

requestAnimationFrame(update);

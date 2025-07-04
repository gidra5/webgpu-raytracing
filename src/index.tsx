import 'uno.css';
import { render } from 'solid-js/web';

import UI from './UI';
import { handleControls } from './controls';
import { renderFrame } from './render';
import { setTime } from './store';

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?'
  );
}

render(() => <UI />, root);

async function update() {
  const now = performance.now();
  setTime(now);

  await renderFrame(now);
  handleControls();
  requestAnimationFrame(update);
}

requestAnimationFrame(update);

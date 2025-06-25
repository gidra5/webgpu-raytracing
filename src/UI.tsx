import { Accessor, createEffect, createSignal, type Component } from 'solid-js';
import { setDebugBVH, store } from './store';
import { lerp } from './utils';

type Box<T> = { value: T };
const usePrev = <T,>(value: Accessor<T>): Accessor<T | null> => {
  let [prev, setPrev] = createSignal<T | null>(null);
  createEffect<T>((_prev) => {
    setPrev(() => _prev);
    return value();
  });
  return prev;
};
const useDiff = (value: Accessor<number>, init = 0): Accessor<number> => {
  const prevValue = usePrev(value);
  return () => value() - (prevValue() ?? 0);
};
const useSmoothedValue = (
  value: Accessor<number>,
  smooth: number
): Accessor<number> => {
  const [_value, setValue] = createSignal(value());
  createEffect<number>((prev) => {
    const next = lerp(prev, value(), smooth);
    setValue(next);
    return next;
  }, value());
  return _value;
};

const App: Component = () => {
  const smoothing = 0.02;
  const renderTime = useSmoothedValue(() => store.timings.dt * 1000, smoothing);
  const gpuTime = useSmoothedValue(
    () => store.timings.render.gpu / 1000,
    smoothing
  );
  const jsTime = useSmoothedValue(() => store.timings.render.js, smoothing);

  return (
    <div class="flex flex-col bg-black/30 p-4 gap-2 min-w-[200px] m-1">
      <div class="text-white text-sm">
        Update-time: {renderTime().toFixed(2)} ms
      </div>

      <div class="text-white text-sm">GPU-time: {gpuTime().toFixed(2)} µs</div>

      <div class="text-white text-sm">JS-time: {jsTime().toFixed(2)} ms</div>
      <label class="flex gap-1 text-white text-sm items-baseline">
        <input
          class="m-0"
          name="debugBVH"
          type="checkbox"
          checked={store.debugBVH}
          onChange={(e) => setDebugBVH(e.target.checked)}
        />
        Debug BVH
      </label>
    </div>
  );
};

export default App;

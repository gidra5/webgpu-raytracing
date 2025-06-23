import { Accessor, createEffect, createSignal, type Component } from 'solid-js';
import { store } from './store';
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
  const renderTime = useSmoothedValue(
    useDiff(() => store.renderTime),
    smoothing
  );
  const gpuTime = useSmoothedValue(() => store.gpuTime / 1000, smoothing);
  const jsTime = useSmoothedValue(() => store.jsTime, smoothing);

  return (
    <div class="flex flex-col bg-black/30 p-4 gap-2 min-w-[200px] m-1">
      <div class="text-white text-sm">
        Frame-time: {renderTime().toFixed(2)} ms
      </div>

      <div class="text-white text-sm">GPU-time: {gpuTime().toFixed(2)} µs</div>

      <div class="text-white text-sm">JS-time: {jsTime().toFixed(2)} ms</div>
    </div>
  );
};

export default App;

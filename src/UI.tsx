import { Accessor, createEffect, createSignal, type Component } from 'solid-js';
import {
  ProjectionType,
  setDebugBVH,
  setDebugNormals,
  setFov,
  setProjectionType,
  setShadingType,
  ShadingType,
  store,
} from './store';
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
  const smoothing = 0.9;
  const renderTime = useSmoothedValue(() => store.timings.dt * 1000, smoothing);
  const gpuTime = useSmoothedValue(
    () => store.timings.render.gpu / 1000,
    smoothing
  );
  const jsTime = useSmoothedValue(() => store.timings.render.js, smoothing);

  return (
    <div class="flex flex-col bg-black/30 p-4 gap-2 min-w-[200px] max-w-[400px] m-1">
      <div class="text-white text-sm">
        Update-time: {renderTime().toFixed(2)} ms
      </div>

      <div class="text-white text-sm">GPU-time: {gpuTime().toFixed(2)} µs</div>

      <div class="text-white text-sm">JS-time: {jsTime().toFixed(2)} ms</div>
      <label class="flex gap-1 text-white text-sm items-baseline">
        <input
          class="m-0"
          type="checkbox"
          checked={store.debugBVH}
          onChange={(e) => setDebugBVH(e.target.checked)}
        />
        Debug BVH
      </label>
      <label class="flex gap-1 text-white text-sm items-baseline">
        <input
          class="m-0"
          type="checkbox"
          checked={store.debugNormals}
          onChange={(e) => setDebugNormals(e.target.checked)}
        />
        Debug Normals
      </label>
      <label class="flex gap-2 text-white text-sm items-baseline">
        Shading type
        <select
          value={store.shadingType}
          onChange={(e) => setShadingType(Number(e.target.value))}
        >
          <option value={ShadingType.Flat}>Flat</option>
          <option value={ShadingType.Phong}>Phong</option>
        </select>
      </label>
      <label class="flex gap-2 text-white text-sm items-baseline">
        Projection type
        <select
          value={store.projectionType}
          onChange={(e) => setProjectionType(Number(e.target.value))}
        >
          <option value={ProjectionType.Panini}>Panini</option>
          <option value={ProjectionType.Perspective}>Perspective</option>
          <option value={ProjectionType.Orthographic}>Orthographic</option>
        </select>
      </label>

      <label class="flex gap-2 text-white text-sm items-baseline">
        Field of view
        <input
          class="m-0"
          type="number"
          value={Math.round((store.fov * 180) / Math.PI)}
          onChange={(e) => setFov((Number(e.target.value) * Math.PI) / 180)}
        />
      </label>

      <div class="text-white">
        Left click to lock mouse, right click to unlock.
      </div>
      <div class="text-white">
        Then you can move with WASD, arrow keys, Space and left Ctrl.
      </div>
      <div class="text-white">
        You can also move camera while mouse is locked.
      </div>
    </div>
  );
};

export default App;

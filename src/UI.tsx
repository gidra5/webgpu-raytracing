import { type Component } from 'solid-js';
import {
  FovOrientation,
  LensShape,
  ProjectionType,
  setBlitView,
  setCircleOfConfusion,
  setDebugBVH,
  setDebugReprojection,
  setFocusDistance,
  setFov,
  setFovOrientation,
  setLensShape,
  setJitterStrength,
  setProjectionType,
  setReprojectionRate,
  setScale,
  setShadingType,
  ShadingType,
  store,
  setResolutionScale,
} from './store';
import { useSmoothedValue } from './utils';

const App: Component = () => {
  const smoothing = 0.1;
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
          checked={store.debugReprojection}
          onChange={(e) => setDebugReprojection(e.target.checked)}
        />
        Debug Reprojection
      </label>
      <label class="flex gap-1 text-white text-sm items-baseline">
        Reprojection rate
        <input
          class="m-0"
          type="number"
          value={store.reprojectionRate}
          onChange={(e) => setReprojectionRate(Number(e.target.value))}
        />
      </label>
      <label class="flex gap-1 text-white text-sm items-baseline">
        Blit view
        <select
          value={store.blitView}
          onChange={(e) => setBlitView(e.target.value as any)}
        >
          <option value="image">Image</option>
          <option value="prevImage">Prev Image</option>
          <option value="depth">Depth</option>
          <option value="prevDepth">Prev Depth</option>
          <option value="depthDelta">Depth Delta</option>
          <option value="normals">Normals</option>
        </select>
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
          <option value={ProjectionType.Fisheye}>Fisheye</option>
          <option value={ProjectionType.Panini}>Panini</option>
          <option value={ProjectionType.Perspective}>Perspective</option>
          <option value={ProjectionType.Orthographic}>Orthographic</option>
        </select>
      </label>
      <label class="flex gap-2 text-white text-sm items-baseline">
        FoV orientation
        <select
          value={store.fovOrientation}
          onChange={(e) => setFovOrientation(Number(e.target.value))}
        >
          <option value={FovOrientation.Horizontal}>Horizontal</option>
          <option value={FovOrientation.Vertical}>Vertical</option>
          <option value={FovOrientation.Diagonal}>Diagonal</option>
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

      <label class="flex gap-2 text-white text-sm items-baseline">
        Jitter strength
        <input
          class="m-0"
          type="number"
          value={store.jitterStrength}
          onChange={(e) => setJitterStrength(Number(e.target.value))}
        />
      </label>

      <label class="flex gap-2 text-white text-sm items-baseline">
        Circle of confusion
        <input
          class="m-0"
          type="number"
          value={store.circleOfConfusion}
          onChange={(e) => setCircleOfConfusion(Number(e.target.value))}
        />
      </label>

      <label class="flex gap-2 text-white text-sm items-baseline">
        Focus distance
        <input
          class="m-0"
          type="number"
          value={store.focusDistance}
          onChange={(e) => setFocusDistance(Number(e.target.value))}
        />
      </label>

      <label class="flex gap-2 text-white text-sm items-baseline">
        Lens shape
        <select
          value={store.lensShape}
          onChange={(e) => setLensShape(Number(e.target.value))}
        >
          <option value={LensShape.Circle}>Circle</option>
          <option value={LensShape.Square}>Square</option>
        </select>
      </label>

      <label class="flex gap-2 text-white text-sm items-baseline">
        Scale
        <input
          class="m-0"
          type="number"
          value={store.scale}
          onChange={(e) => setScale(Number(e.target.value))}
        />
      </label>

      <label class="flex gap-2 text-white text-sm items-baseline">
        Resolution scale
        <input
          class="m-0"
          type="number"
          value={store.resolutionScale}
          onChange={(e) => setResolutionScale(Number(e.target.value))}
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

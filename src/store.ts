import { quat, vec3 } from 'gl-matrix';
import { createStore } from 'solid-js/store';

const [store, setStore] = createStore({
  position: vec3.create(),
  orientation: quat.create(),
  sampleCount: 0,
  bouncesCount: 0,
  fov: 45,
  focusDistance: 10,
  circleOfConfusion: 0.01,

  exposure: 1,
  resample: false,

  debugBVH: false,

  renderTime: 0,
  jsTime: 0,
  gpuTime: 0,
});

export { store };

export const setDebugBVH = (debugBVH: boolean) => {
  setStore('debugBVH', debugBVH);
};

export const setRenderTime = (time: number) => {
  setStore('renderTime', time);
};

export const setGPUTime = (time: number) => {
  setStore('gpuTime', time);
};

export const setJSTime = (time: number) => {
  setStore('jsTime', time);
};

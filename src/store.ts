import { mat3, quat, vec2, vec3 } from 'gl-matrix';
import { createStore } from 'solid-js/store';

const [store, setStore] = createStore({
  position: vec3.create(),
  orientation: quat.create(),
  view: vec2.create(),

  sampleCount: 0,
  bouncesCount: 0,

  fov: Math.PI / 3,
  focusDistance: 10,
  circleOfConfusion: 0.01,
  paniniDistance: 1,
  exposure: 1,
  ambience: 0.1,

  scale: 1,
  sensitivity: 0.03,
  speed: 2,
  runSpeed: 5,

  resample: false,
  debugBVH: false,

  timings: {
    time: 0, // ms
    dt: 0, // sec
    render: {
      js: 0,
      gpu: 0,
    },
  },

  keyboard: [],
});

export { store };

export const setView = (view: vec2) => {
  setStore('view', view);
};

export const setDebugBVH = (debugBVH: boolean) => {
  setStore('debugBVH', debugBVH);
};

export const setTime = (time: number) => {
  setStore('timings', 'dt', (time - store.timings.time) / 1000);
  setStore('timings', 'time', time);
};

export const setRenderGPUTime = (time: number) => {
  setStore('timings', 'render', 'gpu', time);
};

export const setRenderJSTime = (time: number) => {
  setStore('timings', 'render', 'js', time);
};

export const rotateCamera = (d: vec2) => {
  const dt = store.timings.dt;
  vec2.scale(d, d, (dt * store.sensitivity) / store.scale);

  const orientation = quat.clone(store.orientation);
  const right = vec3.fromValues(-1, 0, 0);
  vec3.transformQuat(right, right, orientation);

  const mvRight = vec3.fromValues(right[0], 0, right[2]);
  const mvFront = vec3.fromValues(0, 0, 1);
  vec3.transformQuat(mvFront, mvFront, orientation);
  mvFront[1] = 0;

  const qX = quat.create();
  quat.setAxisAngle(qX, vec3.fromValues(0, 1, 0), d[0]);

  const qY = quat.create();
  quat.setAxisAngle(qY, right, d[1]);

  const qZ = quat.create();
  quat.rotationTo(qZ, right, mvRight);

  quat.mul(orientation, qX, orientation);
  quat.mul(orientation, qY, orientation);
  quat.mul(orientation, qZ, orientation);

  setStore('orientation', orientation);
};

export const move = (d: vec3) => {
  const mvUp = vec3.fromValues(0, 1, 0);

  const mvRight = vec3.fromValues(1, 0, 0);
  vec3.transformQuat(mvRight, mvRight, store.orientation);
  mvRight[1] = 0;

  const mvFront = vec3.fromValues(0, 0, 1);
  vec3.transformQuat(mvFront, mvFront, store.orientation);
  mvFront[1] = 0;

  const dt = store.timings.dt;
  const position = vec3.clone(store.position);
  vec3.scale(d, d, dt * store.speed);

  // @ts-ignore
  vec3.transformMat3(d, d, mat3.fromValues(...mvRight, ...mvUp, ...mvFront));
  vec3.add(position, position, d);

  setStore('position', position);
};

export const pressKey = (key: string) => {
  setStore('keyboard', [...store.keyboard, key]);
};

export const releaseKey = (key: string) => {
  setStore(
    'keyboard',
    store.keyboard.filter((k) => k !== key)
  );
};

export const releaseAllKeys = () => {
  setStore('keyboard', []);
};

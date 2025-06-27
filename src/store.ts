import { mat3, mat4, quat, vec2, vec3 } from 'gl-matrix';
import { createStore } from 'solid-js/store';
import { front, right, up } from './camera';

export enum ShadingType {
  Flat,
  Phong,
}

export enum ProjectionType {
  Panini,
  Perspective,
  Orthographic,
}

const [store, setStore] = createStore({
  loadingTitle: '' as string | null,

  position: vec3.fromValues(0, 0, 0),
  orientation: quat.create(),
  view: vec2.create(),

  counter: 0,
  sampleCount: 1,
  bouncesCount: 1,

  fov: Math.PI / 2,
  focusDistance: 10,
  circleOfConfusion: 0,
  paniniDistance: 1,
  exposure: 1,
  ambience: 0.1,
  shadingType: ShadingType.Phong,
  projectionType: ProjectionType.Perspective,

  resolutionScale: 0.3,
  scale: 1,
  sensitivity: 0.03,
  speed: 2,
  runSpeed: 5,

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

export const view = () => {
  const pos = vec3.clone(store.position);
  vec3.scale(pos, pos, -1);
  const viewMatrix = mat4.fromRotationTranslation(
    mat4.create(),
    store.orientation,
    pos
  );
  return viewMatrix;
};

export const viewProjectionMatrix = () => {
  const m = mat4.create();

  const viewMatrix = mat4.create();
  mat4.invert(viewMatrix, view());

  const projectionMatrix = mat4.create();
  const r = store.view[0] / store.view[1];
  const d = Math.tan(store.fov / 2);
  mat4.perspectiveZO(projectionMatrix, 2 * Math.atan(d / r), r, 0.1, 1000);
  mat4.multiply(m, projectionMatrix, viewMatrix);
  return m;
};

export { store };

export const setLoadingTitle = (title: string) => {
  setStore('loadingTitle', title);
};

export const loadFinished = () => {
  setStore('loadingTitle', null);
};

export const resetCounter = () => {
  setStore('counter', 0);
};

export const incrementCounter = () => {
  setStore('counter', store.counter + 1);
};

export const setFov = (fov: number) => {
  setStore('fov', fov);
  resetCounter();
};

export const setProjectionType = (projectionType: ProjectionType) => {
  setStore('projectionType', projectionType);
  resetCounter();
};

export const setShadingType = (shadingType: ShadingType) => {
  setStore('shadingType', shadingType);
  resetCounter();
};

export const setView = (view: vec2) => {
  setStore('view', view);
  resetCounter();
};

export const setDebugBVH = (debugBVH: boolean) => {
  setStore('debugBVH', debugBVH);
  resetCounter();
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
  const orientation = quat.clone(store.orientation);
  const _right = vec3.clone(right);
  vec3.transformQuat(_right, _right, orientation);

  const mvRight = vec3.fromValues(_right[0], 0, _right[2]);
  const mvFront = vec3.clone(front);
  vec3.transformQuat(mvFront, mvFront, orientation);
  mvFront[1] = 0;

  const qX = quat.create();
  quat.setAxisAngle(qX, up, d[0]);

  const qY = quat.create();
  quat.setAxisAngle(qY, _right, d[1]);

  const qZ = quat.create();
  quat.rotationTo(qZ, _right, mvRight);

  quat.mul(orientation, qX, orientation);
  quat.mul(orientation, qY, orientation);
  quat.mul(orientation, qZ, orientation);

  setStore('orientation', orientation);
  resetCounter();
};

export const move = (d: vec3) => {
  const mvUp = vec3.clone(up);

  const mvRight = vec3.clone(right);
  vec3.transformQuat(mvRight, mvRight, store.orientation);
  mvRight[1] = 0;

  const mvFront = vec3.clone(front);
  vec3.transformQuat(mvFront, mvFront, store.orientation);
  mvFront[1] = 0;

  const position = vec3.clone(store.position);

  // @ts-ignore
  vec3.transformMat3(d, d, mat3.fromValues(...mvRight, ...mvUp, ...mvFront));
  vec3.add(position, position, d);

  setStore('position', position);
  resetCounter();
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

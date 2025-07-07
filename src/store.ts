import { mat3, mat4, quat, vec2, vec3, vec4 } from 'gl-matrix';
import { createStore } from 'solid-js/store';
import { front, right, up } from './camera';
import { Accessor, createMemo } from 'solid-js';
import { Iterator } from 'iterator-js';

export enum ShadingType {
  Flat,
  Phong,
}

export enum ProjectionType {
  Panini,
  Perspective,
  Orthographic,
}

export enum LensType {
  Circle,
  Square,
}

type BlitView =
  | 'image'
  | 'prevImage'
  | 'depth'
  | 'prevDepth'
  | 'depthDelta'
  | 'normals';

const [store, setStore] = createStore({
  loadingTitle: '' as string | null,

  position: vec3.fromValues(0, 0, 0),
  orientation: quat.create(),
  view: vec2.create(),

  counter: 0,
  sampleCount: 1,
  bouncesCount: 1,

  fov: Math.PI / 2,
  focusDistance: 4,
  circleOfConfusion: 0.05,
  paniniDistance: 1,
  exposure: 1,
  ambience: 0.1,
  shadingType: ShadingType.Phong,
  projectionType: ProjectionType.Perspective,
  lensType: LensType.Circle,

  reprojectionRate: 0,

  resolutionScale: 1,
  geometryBufferScale: 1,
  scale: 1,
  sensitivity: 0.03,
  speed: 2,
  runSpeed: 5,
  bvh: {
    maxDepth: 16,
    leafSoftMaxSize: 2,
  },

  debugBVH: false,
  debugReprojection: false,
  pixelJitter: true,
  bilateralFilter: false,
  blitView: 'image' as BlitView,

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

export const viewMatrix = createMemo(() => {
  const pos = vec3.clone(store.position);
  vec3.scale(pos, pos, -1);
  const viewMatrix = mat4.fromRotationTranslation(
    mat4.create(),
    store.orientation,
    pos
  );
  return viewMatrix;
});

export const viewProjectionMatrix = createMemo(() => {
  const m = mat4.create();

  const _viewMatrix = mat4.create();
  mat4.invert(_viewMatrix, viewMatrix());

  const projectionMatrix = mat4.create();
  const r = store.view[0] / store.view[1];
  const d = Math.tan(store.fov / 2);
  mat4.perspectiveZO(projectionMatrix, 2 * Math.atan(d / r), r, 0.1, 1000);
  mat4.multiply(m, projectionMatrix, _viewMatrix);
  return m;
});

export const reprojectionFrustrum = (prevView: Accessor<mat4 | undefined>) =>
  createMemo(() => {
    const view = prevView();
    if (!view) {
      return Iterator.repeat(0).take(12).toArray();
    }

    const aspectRatio = store.view[1] / store.view[0];
    const hfov = store.fov / 2; // horizontal field of view
    const tanHFov = Math.tan(hfov);
    const vfov = Math.atan(tanHFov / aspectRatio); // vertical field of view
    const w = view[15];
    const rayZ = -w / tanHFov;
    // view[2].xyz is forward direction
    const forward = vec3.fromValues(
      view[2 * 4 + 0],
      view[2 * 4 + 1],
      view[2 * 4 + 2]
    );

    const cornerRay = (x: number, y: number) => {
      const dir = vec3.fromValues(x, y * aspectRatio, rayZ);
      vec3.normalize(dir, dir);
      const dir4 = vec4.fromValues(dir[0], dir[1], dir[2], 0);
      vec4.transformMat4(dir4, dir4, view);
      vec3.set(dir, dir4[0], dir4[1], dir4[2]);
      return dir;
    };
    // frustrum side plane normals
    const frustrum = Iterator.zip(
      [cornerRay(-1, -1), cornerRay(1, -1)],
      [cornerRay(-1, 1), cornerRay(-1, -1)]
    )
      .map(([a, b]) => vec3.cross(vec3.create(), a, b))
      .map((a) => vec3.normalize(vec3.create(), a))
      .toArray();

    const [left, top] = frustrum;
    const c = vec3.scale(vec3.create(), forward, -2 * Math.cos(hfov));
    const d = vec3.scale(vec3.create(), forward, -2 * Math.cos(vfov));

    vec3.scale(left, left, store.view[0]);
    vec3.scale(top, top, store.view[1]);

    // for reprojection we need to compute d1 / (d1 + d2)
    // where d1 = dot(n1, p-p0), d2 = dot(n2, p-p0), p0 - view origin,
    // n1 - left side plane normal, n2 - right side plane normal
    // taken from https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/
    // we can collect normals into a 4x3 matrix
    return Iterator.zip(left, top, c, d).flat().toArray();
  });

export const prevViewInv = (prevView: Accessor<mat4 | undefined>) =>
  createMemo(() => {
    const _prevView = prevView();
    if (_prevView) {
      return mat4.invert(mat4.create(), _prevView);
    }
    return mat4.create();
  });

export { store };

export const setDebugReprojection = (debug: boolean) => {
  setStore('debugReprojection', debug);
  resetCounter();
};

export const setPixelJitter = (jitter: boolean) => {
  setStore('pixelJitter', jitter);
  resetCounter();
};

export const setCircleOfConfusion = (circleOfConfusion: number) => {
  setStore('circleOfConfusion', circleOfConfusion);
  resetCounter();
};

export const setFocusDistance = (focusDistance: number) => {
  setStore('focusDistance', focusDistance);
  resetCounter();
};

export const setLensType = (lensType: LensType) => {
  setStore('lensType', lensType);
  resetCounter();
};

export const setReprojectionRate = (rate: number) => {
  setStore('reprojectionRate', rate);
  resetCounter();
};

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

export const setBlitView = (blitView: BlitView) => {
  setStore('blitView', blitView);
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

  if (quat.exactEquals(orientation, store.orientation)) return;
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

  if (vec3.exactEquals(position, store.position)) return;

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

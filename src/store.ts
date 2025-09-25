import { mat3, mat4, quat, vec2, vec3 } from 'gl-matrix';
import { createStore } from 'solid-js/store';
import { front, right, up } from './camera';
import { Accessor, createMemo } from 'solid-js';
import { Iterator } from 'iterator-js';

export enum NormalsType {
  Flat,
  Interpolated,
}

export enum ProjectionType {
  Fisheye,
  Panini,
  Perspective,
  Orthographic,
}

export enum FovOrientation {
  Horizontal,
  Vertical,
  Diagonal,
}

export enum LensShape {
  Circle,
  Square,
}

export enum Tonemapping {
  Reinhard,
  Filmic,
  Aces,
  Lottes,
  None,
}

export enum SkyboxType {
  None,
  Plain,
  Exr,
}

export enum ReprojectionFiltering {
  Average,
  Bilateral,
  ExponentialAverage,
  ErrorWeighted,
}

export enum BlitView {
  Image,
  Image2,
  Image3,
  PrevImage,
  PrevImage2,
  PrevImage3,
  Depth,
  PrevDepth,
  DepthDelta,
  Normals,
  Reproject,
}

export const [store, setStore] = createStore({
  loadingTitle: '' as string | null,

  position: vec3.fromValues(0, 0, 0),
  orientation: quat.create(),
  view: vec2.create(),

  counter: 0,
  sampleCount: 1,
  bouncesDepth: 2,
  samplesPerPoint: 1,
  samplesPerBounce: 1,

  fov: (Math.PI * 1) / 2,
  far: 1000,
  near: 1,
  fovOrientation: FovOrientation.Horizontal,
  focusDistance: 4,
  circleOfConfusion: 0,
  paniniDistance: 1,
  verticalCompression: 0,
  exposure: 5e-28,
  // exposure: 1,
  gamma: 1,
  ambience: 0.1,
  normalsType: NormalsType.Interpolated,
  projectionType: ProjectionType.Perspective,
  lensShape: LensShape.Circle,
  tonemapping: Tonemapping.Aces,
  skybox: SkyboxType.Plain,

  reprojection: {
    rate: 1,
    pointAccuracy: 1e-4,
    objectClamping: false,
    faceClamping: false,
    distanceClamping: false,
    doubleAccuracy: true,
    doubleReprojectErrorCorrection: true,
    gradientErrorCorrection: true,
    identityErrorCorrection: true,
    filtering: ReprojectionFiltering.Bilateral,
    filteringRate: 0.9,
    debug: false,
  },

  gBuffer: {
    width: null,
    height: null,
    layers: 2, // depth peeling layers
    frames: 2, // how many frames to buffer
    frustrum: 0b100000, // bitfield of [+Z, -Z, +X, -X, +Y, -Y] for each view direction
    frustrumSides: 0b100000, // bitfield of [+Z, -Z, +X, -X, +Y, -Y] for clip space cube sides
  },

  depthLayer: 0,

  jitterStrength: 0,
  resolutionScale: 1,
  scale: 1,
  sensitivity: 0.03,
  speed: 2,
  runSpeed: 5,
  bvh: {
    maxDepth: 32,
    leafSoftMaxSize: 4,
  },

  debugBVH: false,
  blitView: BlitView.Image,

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

export const viewInvMatrix = createMemo(() => {
  return mat4.invert(mat4.create(), viewMatrix());
});

export const aspectRatio = createMemo(() => {
  return store.view[0] / store.view[1];
});

export const hFovX = createMemo(() => {
  return store.fov / 2;
});

export const hFovY = createMemo(() => {
  return Math.atan(Math.tan(hFovX()) / aspectRatio());
});

export const projectionMatrix = createMemo(() => {
  return mat4.perspectiveZO(
    mat4.create(),
    2 * hFovY(),
    aspectRatio(),
    store.near,
    store.far
  );
});

export const reprojectionFrustrum = createMemo(() => {
  // TODO: use hFovX and hFovY
  const hfov = hFovX(); // horizontal field of view
  const vfov = Math.atan(Math.tan(hfov) * aspectRatio()); // vertical field of view

  // TODO: simplify the trigs
  // (x * cos - z * sin) / (-2 * z * cos) =
  // 0.5 * (tan - x / z)
  // (y * sin - z * cos) / (-2 * z * cos) =
  // 0.5 * (1 - tan * y / z)
  // 0.5 * [tan, -1] * [1, x / z]
  //       [1, -tan] * [1, y / z]
  const left = vec3.fromValues(Math.cos(hfov), 0, -Math.sin(hfov));
  const top = vec3.fromValues(0, Math.sin(vfov), -Math.cos(vfov));
  const c = vec3.fromValues(0, 0, -2 * Math.cos(hfov));
  const d = vec3.fromValues(0, 0, -2 * Math.cos(vfov));

  vec3.scale(left, left, store.view[0]);
  vec3.scale(top, top, store.view[1]);

  // for reprojection we need to compute d1 / (d1 + d2)
  // where d1 = dot(n1, p-p0), d2 = dot(n2, p-p0), p0 - view origin,
  // n1 - left side plane normal, n2 - right side plane normal
  // taken from https://jacco.ompf2.com/2024/01/18/reprojection-in-a-ray-tracer/
  // we can collect normals into a 4x3 matrix
  return Iterator.zip(left, top, c, d).flat().toArray();
});

export const viewProjectionMatrix = createMemo(() => {
  const _viewMatrix = mat4.invert(mat4.create(), viewMatrix());

  return mat4.multiply(mat4.create(), projectionMatrix(), _viewMatrix);
});

export const invMat = (mat: Accessor<mat4 | undefined>) =>
  createMemo(() => {
    const m = mat();
    if (m) {
      return mat4.invert(mat4.create(), m);
    }
    return mat4.create();
  });

export const setScale = (scale: number) => {
  setStore('scale', scale);
  resetCounter();
};

export const setReprojectionFiltering = (
  reprojectionFiltering: ReprojectionFiltering
) => {
  setStore('reprojection', 'filtering', reprojectionFiltering);
  resetCounter();
};

export const setResolutionScale = (scale: number) => {
  setStore('resolutionScale', scale);
  resetCounter();
};

export const setDebugReprojection = (debug: boolean) => {
  setStore('reprojection', 'debug', debug);
  resetCounter();
};

export const setJitterStrength = (jitter: number) => {
  setStore('jitterStrength', jitter);
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

export const setLensShape = (shape: LensShape) => {
  setStore('lensShape', shape);
  resetCounter();
};

export const setReprojectionRate = (rate: number) => {
  setStore('reprojection', 'rate', rate);
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

export const setFovOrientation = (fovOrientation: FovOrientation) => {
  setStore('fovOrientation', fovOrientation);
  resetCounter();
};

export const setShadingType = (shadingType: NormalsType) => {
  setStore('normalsType', shadingType);
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

export const setDepthLayer = (layer: number) => {
  const maxLayer = Math.max(0, store.gBuffer.layers - 1);
  const clamped = Math.min(Math.max(0, Math.round(layer)), maxLayer);
  setStore('depthLayer', clamped);
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
  vec3.normalize(mvRight, mvRight);

  const mvFront = vec3.clone(front);
  vec3.transformQuat(mvFront, mvFront, store.orientation);
  mvFront[1] = 0;
  vec3.normalize(mvFront, mvFront);

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

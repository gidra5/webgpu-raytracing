import { mat4, vec2, vec3 } from 'gl-matrix';
import {
  computePass,
  createStorageBuffer,
  createUniformBuffer,
  getDevice,
  getTimestampHandler,
  reactiveComputePipeline,
  reactiveUniformBuffer,
  renderBundlePass,
  renderPass,
  renderPipeline,
  writeUint32Buffer,
  writeVec2fBuffer,
} from './gpu';
import {
  FovOrientation,
  incrementCounter,
  LensShape,
  prevViewInv,
  ProjectionType,
  reprojectionFrustrum,
  setRenderGPUTime,
  setRenderJSTime,
  setView,
  ShadingType,
  store,
  Tonemapping,
  viewMatrix,
  viewProjectionMatrix,
} from './store';
import { createEffect, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';
import {
  loadMaterialsToBuffers,
  loadModels,
  loadModelsToBuffers,
  loadSkybox,
} from './scene';
import { wait } from './utils';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu');
const device = await getDevice(context as GPUCanvasContext);
const [imageBuffer, setImageBuffer] = createSignal<GPUBuffer>();
const [prevImageBuffer, setPrevImageBuffer] = createSignal<GPUBuffer>();
const [geometryBuffer, setGeometryBuffer] = createSignal<GPUBuffer>();
const [prevGeometryBuffer, setPrevGeometryBuffer] = createSignal<GPUBuffer>();
const [blitRenderBundle, setBlitRenderBundle] = createSignal<GPURenderBundle>();
const [debugBVHRenderBundle, setDebugBVHRenderBundle] =
  createSignal<GPURenderBundle>();
const [prevView, setPrevView] = createSignal<mat4>(undefined, {
  equals: (a, b) => a && mat4.exactEquals(a, b),
});
const _prevViewInv = prevViewInv(prevView);
const prevViewInvBuffer = reactiveUniformBuffer(16, _prevViewInv);
const _reprojectionFrustrum = reprojectionFrustrum(prevView);
const prevViewBuffer = reactiveUniformBuffer(16, prevView);
const viewBuffer = reactiveUniformBuffer(
  16,
  viewMatrix,
  GPUBufferUsage.COPY_SRC
);
const viewProjBuffer = reactiveUniformBuffer(16, viewProjectionMatrix);
const reprojectionFrustrumBuffer = reactiveUniformBuffer(
  12,
  _reprojectionFrustrum,
  GPUBufferUsage.COPY_SRC
);
const jitterBuffer = createUniformBuffer(
  16,
  'Jitter Buffer',
  GPUBufferUsage.COPY_SRC
);
const prevJitterBuffer = createUniformBuffer(
  16,
  'Prev Jitter Buffer',
  GPUBufferUsage.COPY_DST
);
const prevViewBuffer2 = createUniformBuffer(
  16 * 4,
  'Prev View Buffer 2',
  GPUBufferUsage.COPY_DST
);

const { models, materials } = await loadModels();
const { materialsBuffer } = await loadMaterialsToBuffers(materials);
const { facesBuffer, bvhBuffer, bvhCount, modelsBuffer } =
  // await loadModelsToBuffers(models);
  await loadModelsToBuffers([
    models[2],
    models[10],
    models[6],
    models[11],
    models[8],
    models[5],
    models[3],
    models[4],
  ]);
const skybox = await loadSkybox();
const skyboxSampler = device.createSampler();

const seedUniformBuffer = createUniformBuffer(4);
const counterUniformBuffer = createUniformBuffer(4);
const updatePrevUniformBuffer = createUniformBuffer(4);

const resize = () => {
  const scale = devicePixelRatio * store.resolutionScale;
  canvas.width = canvas.clientWidth * scale;
  canvas.height = canvas.clientHeight * scale;
  setView(vec2.fromValues(canvas.width, canvas.height));
};

createEffect<() => void>((destroy) => {
  destroy?.();
  resize();
  window.addEventListener('resize', resize);
  return () => window.removeEventListener('resize', resize);
});

createEffect<GPUBuffer[]>((prevBuffers) => {
  if (prevBuffers) prevBuffers.forEach((b) => b.destroy());
  const width = store.view[0] + 1;
  const height = store.view[1];

  // color + accumulated samples count
  const imageSize = Float32Array.BYTES_PER_ELEMENT * 4 * width * height;
  const current = createStorageBuffer(
    imageSize,
    'Raytraced Image Buffer',
    GPUBufferUsage.COPY_SRC
  );
  setImageBuffer(current);
  const prev = createStorageBuffer(
    imageSize,
    'Prev Raytraced Image Buffer',
    GPUBufferUsage.COPY_DST
  );
  setPrevImageBuffer(prev);

  const geometryBufferItemSize = Float32Array.BYTES_PER_ELEMENT * 32;
  const geometryBufferSize =
    store.geometryBufferScale * geometryBufferItemSize * width * height;
  const currentGeometry = createStorageBuffer(
    geometryBufferSize,
    'Geometry Buffer',
    GPUBufferUsage.COPY_SRC
  );
  setGeometryBuffer(currentGeometry);
  const prevGeometry = createStorageBuffer(
    geometryBufferSize,
    'Prev Geometry Buffer',
    GPUBufferUsage.COPY_DST
  );
  setPrevGeometryBuffer(prevGeometry);

  return [current, prev, currentGeometry, prevGeometry];
});

createEffect(() => {
  const { pipeline, bindGroups } = renderPipeline({
    vertexShader: () => /* wgsl */ `
      // xy pos + uv
      const FULLSCREEN_QUAD = array<vec4<f32>, 3>(
        vec4(-1, 3, 0, 2),
        vec4(-1, -1, 0, 0),
        vec4(3, -1, 2, 0),
      );

      struct VertexOutput {
        @builtin(position) Position: vec4f,
        @location(0) uv: vec2f,
      }
      
      @vertex
      fn main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
        var output: VertexOutput;
        output.Position = vec4<f32>(FULLSCREEN_QUAD[VertexIndex].xy, 0.0, 1.0);
        output.uv = FULLSCREEN_QUAD[VertexIndex].zw;
        return output;
      }
    `,
    fragmentShader: (x) => /* wgsl */ `
      ${tonemapping}

      ${x.bindVarBuffer('read-only-storage', 'imageBuffer: array<vec4f>', imageBuffer())}
      ${x.bindVarBuffer('read-only-storage', 'prevImageBuffer: array<vec4f>', prevImageBuffer())}

      const viewport = vec2u(${store.view[0]}, ${store.view[1]});
      const viewportf = vec2f(viewport);
      const exposure = ${store.exposure};

      fn getColor(idx: u32, pos: vec2f) -> vec3f {
        if ${store.blitView == 'image'} {
          let value = imageBuffer[idx];
          return value.rgb / value.w * exposure; 
        } else if ${store.blitView == 'prevImage'} {
          let value = prevImageBuffer[idx];
          return value.rgb / value.w; 
        } else if ${store.blitView == 'normals'} {
          let value = imageBuffer[idx];
          return value.rgb; 
        } else if ${store.blitView == 'depth'} {
          // return vec3f(depthBuffer[idx][0]) / 10;
          let value = imageBuffer[idx];
          return value.rgb / value.w; 
        } else if ${store.blitView == 'prevDepth'} {
          // return vec3f(depthBuffer[idx][1]) / 10;
          let value = imageBuffer[idx];
          return value.rgb / value.w; 
        } else if ${store.blitView == 'depthDelta'} {
          // return vec3f(depthBuffer[idx][0] - depthBuffer[idx][1]);
          let value = imageBuffer[idx];
          return value.rgb / value.w; 
        }
        return vec3f(0);
      }

      fn tonemap(c: vec3f) -> vec3f {
        if ${store.tonemapping} == ${Tonemapping.Reinhard} {
          return reinhard(c);
        } else if ${store.tonemapping} == ${Tonemapping.Filmic} {
          return filmic(c);
        } else if ${store.tonemapping} == ${Tonemapping.Aces} {
          return aces(c);
        } else if ${store.tonemapping} == ${Tonemapping.Lottes} {
          return lottes(c);
        } else {
          return c;
        }
      }

      @fragment
      fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
        let pos = uv * viewportf;
        let upos = vec2u(pos);
        let idx = upos.y * viewport.x + upos.x;
        let color = getColor(idx, pos);
        let gammaCorrected = gamma(color, 1 / ${store.gamma});
        let tonemapped = tonemap(gammaCorrected);
        return vec4(tonemapped, 1.0);
      }
    `,
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
  });

  const bundle = renderBundlePass({}, (renderPass) => {
    renderPass.setPipeline(pipeline);
    bindGroups.forEach((bindGroup, i) => renderPass.setBindGroup(i, bindGroup));
    renderPass.draw(3);
  });

  setBlitRenderBundle(bundle);
});

const COMPUTE_WORKGROUP_SIZE_X = 16;
const COMPUTE_WORKGROUP_SIZE_Y = 16;

const structs = /* wgsl */ `
  struct Geometry {
    position: vec3f,
    faceIdx: u32,
    objectIdx: u32,
  }

  struct Ray {
    pos: vec3f, // Origin
    dir: vec3f, // Direction (normalized)
  };

  struct BoundRay {
    pos: vec3f, // Origin
    dir: vec3f, // Direction (normalized)
    maxDist: f32, 
  };

  struct BoundingVolume {
    min: vec3f,
    rightIdx: i32,
    max: vec3f,
    faces: array<i32, 2>,
  }
  
  struct Offset {
    offset: u32,
    count: u32,
  }

  struct Model {
    faces: Offset,
    bvh: Offset,
  }

  struct FacePoint {
    pos: vec3f,
    normal: vec3f,
  }

  struct Face {
    faceNormal: vec3f,
    materialIdx: u32,
    points: array<FacePoint, 3>
  }

  struct Material {
    color: vec3f,
    emission: vec3f
  };
`;

const intervals = /* wgsl */ `
  struct Interval {
    min: f32,
    max: f32,
  };
  
  @must_use
  fn intervalOverlap(interval1: Interval, interval2: Interval) -> bool {
    return interval1.min <= interval2.max || interval2.min <= interval1.max;
  }
  
  @must_use
  fn intervalContains(interval: Interval, x: f32) -> bool {
    return interval.min <= x && x <= interval.max;
  }

  @must_use
  fn intervalSurrounds(interval: Interval, x: f32) -> bool {
    return interval.min < x && x < interval.max;
  }

  @must_use
  fn intervalClamp(interval: Interval, x: f32) -> f32 {
    return min(max(x, interval.min), interval.max);
  }

  const emptyInterval = Interval(f32max, f32min);
  const universeInterval = Interval(f32min, f32max);
  const positiveUniverseInterval = Interval(EPSILON, f32max);
`;

const rayIntersect = /* wgsl */ `
  struct FaceIntersectonResult {
    barycentric: vec3f,
    hit: bool,
  }

  struct Triangle {
    p0: vec3f,
    e1: vec3f,
    e2: vec3f,
  }

  @must_use
  fn rayIntersectFace(
    ray: Ray,
    face: Face,
    interval: Interval
  ) -> FaceIntersectonResult {
    var result: FaceIntersectonResult;
    result.hit = false;

    // Mäller-Trumbore algorithm
    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    
    let p0 = face.points[0].pos;
    let e1 = face.points[1].pos;
    let e2 = face.points[2].pos;

    let h = cross(ray.dir, e2);
    let det = dot(e1, h);
    
    // negative determinant will do backface culling
    // near zero determinant will detect parallel rays
    if det < EPSILON * EPSILON { 
      return result;
    }

    let s = ray.pos - p0;
    let u = dot(s, h);

    if u < 0.0f || u > det {
      return result;
    }

    let q = cross(s, e1);
    let v = dot(ray.dir, q);

    if v < 0.0f || u + v > det {
      return result;
    }

    let t = dot(e2, q);
    let pt = vec3f(t, u, v) / det;

    if !intervalSurrounds(interval, pt.x) {
      return result;
    }

    result.barycentric = pt;
    result.hit = true;
    
    return result;
  }
`;

const bvIntersect = /* wgsl */ `
  struct BVIntersectionResult {
    hit: bool,
    t: f32,
  }
  
  @must_use
  fn rayIntersectBV(ray: Ray, bv: BoundingVolume, interval: Interval) -> BVIntersectionResult {
    let t0 = (bv.min - ray.pos) / ray.dir;
    let t1 = (bv.max - ray.pos) / ray.dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let near = max(tmin.x, max(tmin.y, tmin.z));
    let far = min(tmax.x, min(tmax.y, tmax.z));
    if near < far && intervalOverlap(interval, Interval(near, far)) {
      return BVIntersectionResult(true, near);
    }
    return BVIntersectionResult(false, f32max);
  }
`;

const bvh = () => /* wgsl */ `
  struct BVHIntersectionResult {
    hit: bool,
    barycentric: vec3f,
    faceIdx: u32,
    objectIdx: u32,
  }

  struct BVHIntersectionStackEntry {
    idx: u32,
    t: f32,
  }
  const BV_MAX_STACK_DEPTH = 16;
  @must_use
  fn rayIntersectBVH(
    ray: Ray,
    maxDist: f32
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(maxDist, 0, 0);
    result.hit = false;
    result.faceIdx = 0;

    for (var objectIdx = 0u; objectIdx < arrayLength(&models); objectIdx++) {
      let hit = rayIntersectObjectBVH(ray, objectIdx, result.barycentric.x);
      if !hit.hit {
        continue;
      }
      result = hit;
    }

    return result;
  }

  @must_use
  fn rayIntersectBVHAnyHit(
    ray: Ray,
    maxDist: f32
  ) -> bool {
    for (var objectIdx = 0u; objectIdx < arrayLength(&models); objectIdx++) {
      let hit = rayIntersectObjectBVHAnyHit(ray, objectIdx, maxDist);
      if hit {
        return true;
      }
    }

    return false;
  }

  @must_use
  fn rayIntersectObjectBVHAnyHit(
    ray: Ray,
    objectIdx: u32,
    maxDist: f32
  ) -> bool {
    var stack: array<u32, BV_MAX_STACK_DEPTH>;
    var top: i32;

    let model = models[objectIdx];
    let bv = bvh[model.bvh.offset];
    let bvResult = rayIntersectBV(ray, bv, Interval(min_dist, maxDist));
    if (!bvResult.hit) {
      return false;
    }

    top = 0;
    stack[top] = 0;

    while (top > -1) {
      let idx = stack[top];
      top--;
      let bv = bvh[model.bvh.offset + idx];

      let isLeaf = bv.rightIdx == -1; // right will be -1 too
      if (isLeaf) {
        for (var i = 0u; i < 2; i = i + 1) {
          let offset = bv.faces[i];
          if offset == -1 {
            continue;
          }
          let faceIdx = model.faces.offset + u32(offset);
          let face = faces[faceIdx];
          let hit = rayIntersectFace(ray, face, Interval(min_dist, maxDist));
          if (!hit.hit) {
            continue;
          }
          return true;
        }
        continue;
      }

      let leftIdx = u32(idx + 1);
      let rightIdx = u32(bv.rightIdx);
      let left = bvh[model.bvh.offset + leftIdx];
      let right = bvh[model.bvh.offset + rightIdx];
      let resultLeft = rayIntersectBV(ray, left, Interval(min_dist, maxDist));
      let resultRight = rayIntersectBV(ray, right, Interval(min_dist, maxDist));
      if resultLeft.hit && resultRight.hit {
        if resultLeft.t < resultRight.t {
          top++;
          stack[top] = rightIdx;
          top++;
          stack[top] = leftIdx;
        } else {
          top++;
          stack[top] = leftIdx;
          top++;
          stack[top] = rightIdx;
        }
      } else if resultLeft.hit {
        top++;
        stack[top] = leftIdx;
      } else if resultRight.hit {
        top++;
        stack[top] = rightIdx;
      }
    }

    return false;
  }

  @must_use
  fn rayIntersectObjectBVH(
    ray: Ray,
    objectIdx: u32,
    maxDist: f32
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(maxDist, 0, 0);
    result.hit = false;
    result.faceIdx = 0;
    
    var stack: array<BVHIntersectionStackEntry, BV_MAX_STACK_DEPTH>;
    var top: i32;

    let model = models[objectIdx];
    let bv = bvh[model.bvh.offset];
    let bvResult = rayIntersectBV(ray, bv, Interval(min_dist, result.barycentric.x));
    if (!bvResult.hit) {
      return result;
    }

    top = 0;
    stack[top] = BVHIntersectionStackEntry(0, bvResult.t);

    while (top > -1) {
      let stackEntry = stack[top];
      top--;
      if stackEntry.t > result.barycentric.x {
        continue;
      }

      let idx = stackEntry.idx;
      let bv = bvh[model.bvh.offset + idx];

      let isLeaf = bv.rightIdx == -1; // right will be -1 too
      if (isLeaf) {
        for (var i = 0u; i < 2; i = i + 1) {
          let offset = bv.faces[i];
          if offset == -1 {
            continue;
          }
          let faceIdx = model.faces.offset + u32(offset);
          let face = faces[faceIdx];
          let hit = rayIntersectFace(ray, face, Interval(min_dist, result.barycentric.x));
          if (!hit.hit) {
            continue;
          }
          result.barycentric = hit.barycentric;
          result.hit = true;
          result.faceIdx = faceIdx;
          result.objectIdx = objectIdx;
        }
        continue;
      }

      let leftIdx = u32(idx + 1);
      let rightIdx = u32(bv.rightIdx);
      let left = bvh[model.bvh.offset + leftIdx];
      let right = bvh[model.bvh.offset + rightIdx];
      let resultLeft = rayIntersectBV(ray, left, Interval(min_dist, result.barycentric.x));
      let resultRight = rayIntersectBV(ray, right, Interval(min_dist, result.barycentric.x));
      if resultLeft.hit && resultRight.hit {
        let leftEntry = BVHIntersectionStackEntry(leftIdx, resultLeft.t);
        let rightEntry = BVHIntersectionStackEntry(rightIdx, resultRight.t);
        if resultLeft.t < resultRight.t {
          top++;
          stack[top] = rightEntry;
          top++;
          stack[top] = leftEntry;
        } else {
          top++;
          stack[top] = leftEntry;
          top++;
          stack[top] = rightEntry;
        }
      } else if resultLeft.hit {
        top++;
        stack[top] = BVHIntersectionStackEntry(leftIdx, resultLeft.t);
      } else if resultRight.hit {
        top++;
        stack[top] = BVHIntersectionStackEntry(rightIdx, resultRight.t);
      }
    }

    return result;
  }
`;

const raygen = () => /* wgsl */ `
  const cameraFovAngle = ${store.fov};
  const cameraFovDistance = ${(store.fov / Math.PI) * 4};
  const cameraRayZ = -1/tan(cameraFovAngle / 2);
  const paniniDistance = ${store.paniniDistance};
  const lensFocusDistance = ${store.focusDistance};
  const circleOfConfusionRadius = ${store.circleOfConfusion};
  const projectionType = ${store.projectionType};
  const verticalCompression = ${store.verticalCompression};
  const fovOrientation = ${store.fovOrientation};

  fn pinholeRayDirection(pixel: vec2f) -> vec3f {
    return normalize(vec3(pixel, cameraRayZ));
  }

  fn paniniRayDirection(pixel: vec2f) -> vec3f {
    let half_fov = cameraFovAngle / 2.0;
    let hv = pixel * half_fov;
    let half_panini_fov = atan2(sin(half_fov), cos(half_fov) + paniniDistance);
    let hv_pan = hv * half_panini_fov; 

    let M = sqrt(1.0 - square(sin(hv_pan.x) * paniniDistance)) + paniniDistance * cos(hv_pan.x);
    let x = sin(hv_pan.x) * M;
    let z = (cos(hv_pan.x) * M) - paniniDistance;
    
    let y = tan(hv_pan.y) * (z + paniniDistance * (1.0 - verticalCompression));

    return normalize(vec3<f32>(x, y, -z));
  }

  fn square(x: f32) -> f32 {
    return x * x;
  }

  fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
  }

  fn fisheyeRayDirection(pixel: vec2f) -> vec3f {
    let clampedHalfFOV = cameraFovAngle / 2;
    let angle = pixel * clampedHalfFOV;

    return normalize(vec3(
      -sin(angle.x), 
      -sin(angle.y) * cos(angle.x), 
      cos(angle.y) * cos(angle.x)
    ));
  }

  fn orthographicRayDirection(uv: vec2f) -> vec3f {
    return vec3(0, 0, -1);
  }

  fn thinLensRay(dir: vec3f, uv: vec2f) -> Ray {
    let pos = vec3(uv * circleOfConfusionRadius, 0.f);
    let focusPoint = -dir * lensFocusDistance / dir.z;
    return Ray(
      pos,
      normalize(focusPoint - pos)
    );
  }

  fn cameraRayDirection(uv: vec2f) -> vec3f {
    switch (projectionType) {
      case ${ProjectionType.Panini}: {
        return paniniRayDirection(uv);
      }
      case ${ProjectionType.Perspective}: {
        return pinholeRayDirection(uv);
      }
      case ${ProjectionType.Orthographic}: {
        return orthographicRayDirection(uv);
      }
      case ${ProjectionType.Fisheye}: {
        return fisheyeRayDirection(uv);
      }
      default: {
        return vec3(0);
      }
    }
  }

  fn cameraRayPosition(uv: vec2f) -> vec3f {
    if projectionType == ${ProjectionType.Orthographic} {
      return vec3(uv * cameraFovDistance, 0);
    }
    return vec3(0);
  }

  fn ray_transform(_ray: Ray, view: mat4x4f) -> Ray {
    var ray = _ray;
    let ray_pos = view * vec4(ray.pos, 1.);
    ray.pos = ray_pos.xyz;
    ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
    ray.dir = (view * vec4(ray.dir, 0.)).xyz;
    return ray;
  }

  fn sampleLens() -> vec2f {
    if ${store.lensShape} == ${LensShape.Circle} {
      return sample_incircle(random_2());
    } else if ${store.lensShape} == ${LensShape.Square} {
      return sample_insquare(random_2());
    }
    return vec2f(0);
  }

  fn cameraRay(pos: vec2f, view: mat4x4f) -> Ray {
    var uv = (2. * pos - viewportf);

    if ${store.fovOrientation} == ${FovOrientation.Vertical} {
      uv /= viewportf.y;
    } else if ${store.fovOrientation} == ${FovOrientation.Horizontal} {
      uv /= viewportf.x;
    } else if ${store.fovOrientation} == ${FovOrientation.Diagonal} {
      uv /= length(viewportf);
    }

    let rayDirection = cameraRayDirection(uv);
    
    var ray = thinLensRay(rayDirection, sampleLens());
    ray.pos += cameraRayPosition(uv);
    return ray_transform(ray, view);
  }
`;

const scene = () => /* wgsl */ `
  const flatShading = ${store.shadingType};
    
  const ambience = ${store.ambience};
  const sun_color = vec3f(1);
  const sun_dir = normalize(vec3f(1, 1, 1));
  const sphere_center = vec3f(0, 0, 4);

  fn sceneAnyHit(ray: Ray, maxDist: f32) -> bool {
    return rayIntersectBVHAnyHit(ray, maxDist);
  }

  fn scene(ray: Ray, maxDist: f32) -> BVHIntersectionResult {
    return rayIntersectBVH(ray, maxDist);
  }

  fn objectFaceHit(faceIdx: u32, objectIdx: u32, ray: Ray, maxDist: f32) -> BVHIntersectionResult {
    var hit: BVHIntersectionResult;
    hit.hit = false;
    hit.barycentric = vec3f(maxDist, 0, 0);
    hit.faceIdx = faceIdx;
    hit.objectIdx = objectIdx;
    
    {
      let model = models[objectIdx];
      let face = faces[model.faces.offset + faceIdx];
      let _hit = rayIntersectFace(ray, face, Interval(min_dist, maxDist));
      if _hit.hit {
        hit.hit = true;
        hit.barycentric = _hit.barycentric;
      }
    }

    {
      let model = models[objectIdx];
      let _hit = rayIntersectObjectBVH(ray, objectIdx, hit.barycentric.x + EPSILON);
      if _hit.hit {
        hit = _hit;
      }
    }

    return hit;
  }

  fn objectFaceAnyHit(faceIdx: u32, objectIdx: u32, ray: Ray, maxDist: f32) -> bool {
    {
      let model = models[objectIdx];
      let face = faces[model.faces.offset + faceIdx];
      let _hit = rayIntersectFace(ray, face, Interval(min_dist, maxDist));
      if _hit.hit {
        return true;
      }
    }

    {
      let model = models[objectIdx];
      let _hit = rayIntersectObjectBVH(ray, objectIdx, maxDist);
      if _hit.hit {
        return true;
      }
    }

    return false;
  }

  struct SceneSample {
    p: f32, // 1/pdf
    point: vec3f,
    normal: vec3f,
    uv: vec2f,
    materialIdx: u32,
  }

  fn sampleScene() -> SceneSample {
    let randomModelIdx = random_1u() % arrayLength(&models); 
    let model = models[randomModelIdx];
    var sample = sampleModel(model);
    sample.p *= f32(arrayLength(&models));
    return sample;
  }

  fn sampleLights() -> SceneSample {
    var sample = sampleModel(models[0]);
    return sample;
  }

  fn sampleModel(model: Model) -> SceneSample {
    let randomFaceIdx = random_1u() % model.faces.count;
    let face = faces[model.faces.offset + randomFaceIdx];
    var sample = sampleFace(face);
    sample.p *= f32(model.faces.count);
    return sample;
  }

  fn sampleFace(face: Face) -> SceneSample {
    let uv = sample_intriangle(random_2());
    let point = facePointOffset(face, uv);
    let normal = faceNormal(face, uv);
    let texUV = faceTexCoords(face, uv);
    let p = cross(face.points[1].pos, face.points[2].pos); // TODO: precompute area
    return SceneSample(length(p)/2, point, normal, texUV, face.materialIdx);
  }

  // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf
  // p. 83
  // computing from uv and offsetting the point 
  // makes sure there are no self-intersections 
  // due to floating point errors
  fn facePoint(face: Face, uv: vec2f) -> vec3f {
    let p1 = face.points[0].pos;
    let e1 = face.points[1].pos;
    let e2 = face.points[2].pos;
    let p = p1 + mat2x3f(e1, e2) * uv;
    return p;
  }
  fn facePointOffset(face: Face, uv: vec2f) -> vec3f {
    let p1 = face.points[0].pos;
    let e1 = face.points[1].pos;
    let e2 = face.points[2].pos;
    let p = p1 + mat2x3f(e1, e2) * uv;
    return offsetRay(p, face.faceNormal);
  }

  fn faceNormal(face: Face, uv: vec2f) -> vec3f {
    if (flatShading == ${ShadingType.Phong}) {
      let n1 = face.points[0].normal;
      let n2 = face.points[1].normal;
      let n3 = face.points[2].normal;
      return mat3x3f(n1, n2, n3) * uv2toUV3(uv);
    } else {
      return face.faceNormal;
    }
  }

  const origin = 1.0 / 32.0;
  const floatScale = 1.0 / 65536.0;
  const intScale = 256.0;
  fn offsetRay(p: vec3f, n: vec3f) -> vec3f {
    let ofI = vec3i(intScale * n);
    let pI = vec3f(
      bitcast<f32>(bitcast<i32>(p.x) + select(-ofI.x, ofI.x, p.x < 0)),
      bitcast<f32>(bitcast<i32>(p.y) + select(-ofI.y, ofI.y, p.y < 0)),
      bitcast<f32>(bitcast<i32>(p.z) + select(-ofI.z, ofI.z, p.z < 0))
    );
    return vec3f(
      select(p.x + floatScale * n.x, pI.x, abs(p.x) < origin),
      select(p.y + floatScale * n.y, pI.y, abs(p.y) < origin),
      select(p.z + floatScale * n.z, pI.z, abs(p.z) < origin)
    );
  }

  // TODO: correct uv texture coords
  fn faceTexCoords(face: Face, uv: vec2f) -> vec2f {
    // let p1 = face.points[0].uv;
    // let e1 = face.points[1].uv;
    // let e2 = face.points[2].uv;
    // return mat3x2f(p1, e1, e2) * uv2toUV3(uv);
    return uv;
  }

  fn uv2toUV3(uv: vec2f) -> vec3f {
    return vec3f(1-uv.x-uv.y, uv.x, uv.y);
  }

  // Function to sample the skybox
  fn sampleSkybox(dir: vec3f) -> vec3f {
    let u = (atan2(dir.z, dir.x) * INV_PI + 1) * 0.5; 
    let v = 1 - acos(dir.y) * INV_PI;
    let uv = vec2<f32>(u, v);

    let color = textureSampleLevel(skyboxTexture, skyboxSampler, uv, 0);
    return color.xyz;
  }
`;

const derivatives = () => /* wgsl */ `
  fn dFdx1(p: f32) -> f32 {
    var dx = p - quadSwapX(p);
    if quadIdx == 0 || quadIdx == 2 {
      dx = -dx;
    }
    return dx;
  }

  fn dFdy1(p: f32) -> f32 {
    var dy = p - quadSwapY(p);
    if quadIdx == 0 || quadIdx == 1 {
      dy = -dy;
    }
    return dy;
  }

  fn dFdx2(p: vec2f) -> vec2f {
    var dx = p - quadSwapX(p);
    if quadIdx == 0 || quadIdx == 2 {
      dx = -dx;
    }
    return dx;
  }

  fn dFdy2(p: vec2f) -> vec2f {
    var dy = p - quadSwapY(p);
    if quadIdx == 0 || quadIdx == 1 {
      dy = -dy;
    }
    return dy;
  }

  fn dFdx3(p: vec3f) -> vec3f {
    var dx = p - quadSwapX(p);
    if quadIdx == 0 || quadIdx == 2 {
      dx = -dx;
    }
    return dx;
  }

  fn dFdy3(p: vec3f) -> vec3f {
    var dy = p - quadSwapY(p);
    if quadIdx == 0 || quadIdx == 1 {
      dy = -dy;
    }
    return dy;
  }

  fn dFdx4(p: vec4f) -> vec4f {
    var dx = p - quadSwapX(p);
    if quadIdx == 0 || quadIdx == 2 {
      dx = -dx;
    }
    return dx;
  }

  fn dFdy4(p: vec4f) -> vec4f {
    var dy = p - quadSwapY(p);
    if quadIdx == 0 || quadIdx == 1 {
      dy = -dy;
    }
    return dy;
  }
`;

const reproject = () => /* wgsl */ `
  struct ReprojectionResult {
    color: vec4f, // color + accumulated samples count
  }

  fn roundUpTo2(x: vec2f, n: f32) -> vec2f {
    return round(x / n) * n;
  }

  fn roundUpTo(x: f32, n: f32) -> f32 {
    return round(x / n) * n;
  }

  fn reprojectPoint(p: vec3f, view: mat4x4f) -> vec2f {
    let duv = reprojectionFrustrum * (p - view[3].xyz);
    return duv.xy / duv.zw;
  }

  const bilateralFilterRadius = 2;
  const bilateralFilterSigmaPos = 0.01;
  const bilateralFilterSigmaColor = 0.01;
  const bilateralFilterStep = 0.1;
  fn bilateralFilter(uv: vec2f, p: vec3f, c: vec3f) -> vec4f {
    var color = vec4f(0);
    var weight = 0.;
    for (var i = -bilateralFilterRadius; i <= bilateralFilterRadius; i = i + 1) {
      for (var j = -bilateralFilterRadius; j <= bilateralFilterRadius; j = j + 1) {
        let _uv = uv + vec2f(f32(i), f32(j)) * bilateralFilterStep;
        let _color = sampleImage4(_uv, &prevImageBuffer);
        if _color.w <= 0 {
          continue;
        }

        let _pos = sampleGeometryAll(_uv, &prevGeometryBuffer).position;
        let dp = p - _pos;
        let dc = c - _color.xyz/_color.w;
        let _weight = exp(
          -dot(dp, dp) / bilateralFilterSigmaPos -
           dot(dc, dc) / bilateralFilterSigmaColor
        );
        color += _color * _weight;
        weight += _weight;
      }
    }

    if weight == 0. {
      return vec4f(0);
    }

    return color / weight;
  }


  const threshold = 0.00000001;

  fn reproject(p: vec3f, c: vec3f) -> ReprojectionResult {
    let view = prevViewMatrix2;
    let uv = reprojectPoint(p, view);
    if any(uv < vec2(0)) || any(uv > vec2(viewportf)) { // outside viewport
      if ${store.debugReprojection} {
        return ReprojectionResult(vec4f(0, 1, 0, 1));
      } else {
        return ReprojectionResult(vec4f(0));
      }
    }

    var min_uv = uv;
    var dp = sampleGeometryAll(min_uv, &prevGeometryBuffer).position - p;
    var d = dot(dp, dp);

    if !(d < threshold) {
      var step = 0.1;
      for (var i = 0u; i < 128u && d >= threshold; i = i + 1u) {
        if i % 16 == 0 {
          step -= 0.005;
        }
        let next_uv = min_uv - sample_insquare(random_2()) * step;
        let next_dp = sampleGeometryAll(next_uv, &prevGeometryBuffer).position - p; 
        let next_d = dot(next_dp, next_dp);
        if next_d < d {
          dp = next_dp;
          d = next_d;
          min_uv = next_uv;
        }
      }
    }

    if !(d < threshold) { // didn't converge fast enough, rejecting
      if ${store.debugReprojection} {
        return ReprojectionResult(vec4f(d, 0, 0, 1));
      } else {
        return ReprojectionResult(vec4f(0));
      }
    }

    if ${store.debugReprojection} {
      return ReprojectionResult(vec4f(fract(min_uv/4), 1, 1));
    } else if ${store.bilateralFilter} {
      let color = bilateralFilter(min_uv, p, c);
      if color.w == 0. {
        let color = sampleImage4(min_uv, &prevImageBuffer);
        return ReprojectionResult(color);
      }
      return ReprojectionResult(color);
    } else {
      let color = sampleImage4(min_uv, &prevImageBuffer);
      return ReprojectionResult(color);
    }
  }
`;

const computeColor = () => /* wgsl */ `
  fn pixelHitDist(idx: u32, ray: Ray) -> f32 {
    var hit: BVHIntersectionResult;
    hit.hit = false;
    hit.barycentric = vec3f(f32max, 0, 0);

    for (var i = 0u; i < 4; i = i + 1u) {
      let quadIdx = quad[i];
      let prevObjectIdx = prevGeometryBuffer[quadIdx].objectIdx;
      let prevFaceIdx = prevGeometryBuffer[quadIdx].faceIdx;
      if prevFaceIdx == hit.faceIdx {
        continue;
      }
      
      let _hit = objectFaceHit(prevFaceIdx, prevObjectIdx, ray, hit.barycentric.x + EPSILON);
      if _hit.hit {
        hit = _hit;
      }
    }
    
    return hit.barycentric.x + EPSILON;
  }

  fn pointColor(point: vec3f, normal: vec3f) -> vec3f {
    var color = vec3f(0);

    for (var i = 0u; i < ${store.samplesPerPoint}; i = i + 1u) {
      let s = sampleLights();
      let sMaterial = materials[s.materialIdx];
      let ds = s.point - point;
      let d_sq = dot(ds, ds);
      let d = ds * inverseSqrt(d_sq);
      let r = Ray(point, d);
      color += in_shadow(r, d_sq) * attenuation(d, normal) * sMaterial.emission / d_sq * s.p;
    }

    return color / ${store.samplesPerPoint};
  }

  struct BounceStackEntry {
    ray: Ray,
    maxDist: f32,

    color: vec4f,
    throughput: vec3f,
  }
  const maxBounces = ${store.bouncesDepth};
  fn pixelColor(_hit: ptr<function, BVHIntersectionResult>, _ray: Ray, maxDist: f32) -> vec3f {
    var stack: array<BounceStackEntry, maxBounces>;
    var top: u32;

    top = 0;
    stack[top] = BounceStackEntry(_ray, maxDist, vec4f(0), vec3f(1));

    while (top < maxBounces - 1) {
      // shoot a ray out into the world
      let entry = stack[top];
      let hit = scene(entry.ray, entry.maxDist);
      var color = entry.color.rgb;
      var throughput = entry.throughput;
      if top == 0 {
        *_hit = hit;
      }
      if !hit.hit {
        stack[top].color += vec4f(sampleSkybox(entry.ray.dir) * throughput, 1);
        break;
      }

      let face = faces[hit.faceIdx];
      let material = materials[face.materialIdx];
      color += material.emission * throughput;
      throughput *= material.color;

      let normal = faceNormal(face, hit.barycentric.yz);
      let ray = Ray(
        facePointOffset(face, hit.barycentric.yz),
        sample_cosine_weighted_hemisphere(random_2(), 1, normal)
      );
      top++;
      stack[top] = BounceStackEntry(ray, f32max, vec4f(color, 1), throughput);

      // russian roulette
      {
        let p = max(throughput.x, max(throughput.y, throughput.z));
        if random_1() > p {
          break;
        }
        stack[top].throughput /= p;
      }
    }

    return stack[top].color.rgb;
  }
  
  fn in_shadow(ray: Ray, mag_sq: f32) -> f32 {
    return select(1., 0., sceneAnyHit(ray, sqrt(mag_sq)));
  }

  fn light_vis(pos: vec3f, dir: vec3f) -> f32 {
    return in_shadow(Ray(pos, dir), f32max);
  }

  fn attenuation(dir: vec3f, norm: vec3f) -> f32 {
    return max(dot(dir, norm), 0.);
  }
`;

const bilinearInterpolation = /* wgsl */ `
  fn bilinearInterpolation(uv: vec2f, p: vec4f) -> f32 {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }

  fn bilinearInterpolation2(uv: vec2f, p: mat4x2f) -> vec2f {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }

  fn bilinearInterpolation3(uv: vec2f, p: mat4x3f) -> vec3f {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }

  fn bilinearInterpolation4(uv: vec2f, p: mat4x4f) -> vec4f {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }
`;

const imageSampler = /* wgsl */ `
  fn imageIdx(uv: vec2u) -> u32 {
    return uv.x + uv.y * viewport.x;
  }

  // fn sampleImage(uv: vec2f, _image: ptr<function, array<vec4f>>) -> f32 {
  //   let uv_u = floor(uv);
  //   let uv_f = fract(uv);
  //   let m = vec4f(
  //     image[imageIdx(uv_u)],
  //     image[imageIdx(uv_u + vec2u(1, 0))],
  //     image[imageIdx(uv_u + vec2u(0, 1))],
  //     image[imageIdx(uv_u + vec2u(1, 1))],
  //   );
  //   let value = bilinearInterpolation4(uv_f, m);
  //   return value;
  // }

  // fn sampleImage2(uv: vec2f, _image: ptr<function, array<vec4f>>) -> vec2f {
  //   let uv_u = floor(uv);
  //   let uv_f = fract(uv);
  //   let m = mat4x2f(
  //     image[imageIdx(uv_u)],
  //     image[imageIdx(uv_u + vec2u(1, 0))],
  //     image[imageIdx(uv_u + vec2u(0, 1))],
  //     image[imageIdx(uv_u + vec2u(1, 1))],
  //   );
  //   let value = bilinearInterpolation2(uv_f, m);
  //   return value;
  // }

  fn sampleImage3(uv: vec2f, _image: ptr<storage, array<vec3f>, read_write>) -> vec3f {
    let uv_u = vec2u(floor(uv));
    let uv_f = fract(uv);
    let m = mat4x3f(
      (*_image)[imageIdx(uv_u)],
      (*_image)[imageIdx(uv_u + vec2u(1, 0))],
      (*_image)[imageIdx(uv_u + vec2u(0, 1))],
      (*_image)[imageIdx(uv_u + vec2u(1, 1))],
    );
    let value = bilinearInterpolation3(uv_f, m);
    return value;
  }

  fn sampleImage4(uv: vec2f, _image: ptr<storage, array<vec4f>, read>) -> vec4f {
    let uv_u = vec2u(floor(uv));
    let uv_f = fract(uv);
    let m = mat4x4f(
      (*_image)[imageIdx(uv_u)],
      (*_image)[imageIdx(uv_u + vec2u(1, 0))],
      (*_image)[imageIdx(uv_u + vec2u(0, 1))],
      (*_image)[imageIdx(uv_u + vec2u(1, 1))],
    );
    let value = bilinearInterpolation4(uv_f, m);
    return value;
  }
`;

const geometrySampler = () => /* wgsl */ `
  fn geometryIdx(uv: vec2u) -> u32 {
    return uv.x + uv.y * viewport.x;
  }

  fn sampleGeometryAll(uv: vec2f, buffer: ptr<storage, array<Geometry>, read>) -> Geometry {
    let uv_u = vec2u(floor(uv));
    let uv_f = fract(uv);
    
    var result: Geometry;
    let positions = mat4x3f(
      (*buffer)[geometryIdx(uv_u)].position,
      (*buffer)[geometryIdx(uv_u + vec2u(1, 0))].position,
      (*buffer)[geometryIdx(uv_u + vec2u(0, 1))].position,
      (*buffer)[geometryIdx(uv_u + vec2u(1, 1))].position,
    );
    result.position = bilinearInterpolation3(uv_f, positions);
    return result;
  }
`;

const matInv = /* wgsl */ `
  fn inverse(m: mat4x4f) -> mat4x4f {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    return mat4x4f(
        a11 * b11 - a12 * b10 + a13 * b09,
        a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03,
        a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07,
        a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01,
        a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06,
        a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00,
        a21 * b02 - a20 * b04 - a23 * b00,
        a11 * b07 - a10 * b09 - a12 * b06,
        a00 * b09 - a01 * b07 + a02 * b06,
        a31 * b01 - a30 * b03 - a32 * b00,
        a20 * b03 - a21 * b01 + a22 * b00) * (1 / det);
  }
`;

const [computePipeline, computeBindGroups] = reactiveComputePipeline({
  shader: (x) => /* wgsl */ `
    enable subgroups;

    ${x.bindVarBuffer('storage', 'imageBuffer: array<vec4f>', imageBuffer())}
    ${x.bindVarBuffer('read-only-storage', 'prevImageBuffer: array<vec4f>', prevImageBuffer())}
    ${x.bindVarBuffer('storage', 'geometryBuffer: array<Geometry>', geometryBuffer())}
    ${x.bindVarBuffer('read-only-storage', 'prevGeometryBuffer: array<Geometry>', prevGeometryBuffer())}
    ${x.bindVarBuffer('uniform', 'viewMatrix: mat4x4f', viewBuffer)}
    ${x.bindVarBuffer('uniform', 'prevViewMatrix: mat4x4f', prevViewBuffer)}
    ${x.bindVarBuffer('uniform', 'prevViewMatrix2: mat4x4f', prevViewBuffer2)}
    ${x.bindVarBuffer('uniform', 'prevViewInvMatrix: mat4x4f', prevViewInvBuffer)}
    ${x.bindVarBuffer('uniform', 'reprojectionFrustrum: mat3x4f', reprojectionFrustrumBuffer)}

    ${x.bindVarBuffer('read-only-storage', 'faces: array<Face>', facesBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'materials: array<Material>', materialsBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'models: array<Model>', modelsBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'bvh: array<BoundingVolume>', bvhBuffer)}

    ${x.bindVarBuffer('uniform', 'seed: u32', seedUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'counter: u32', counterUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'updatePrev: u32', updatePrevUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'jitter: vec2f', jitterBuffer)}
    ${x.bindVarBuffer('uniform', 'prevJitter: vec2f', prevJitterBuffer)}

    ${x.bindTexture('skyboxTexture', 'unfilterable-float', skybox)}
    ${x.bindSampler('skyboxSampler', 'non-filtering', skyboxSampler)}

    const _reproject = ${store.reprojectionRate > 0};
    const viewport = vec2u(${store.view[0]}, ${store.view[1]});
    const viewportf = vec2f(viewport);
    const aspect = viewportf.y / viewportf.x;
    const viewportN = viewportf / viewportf.x; // viewport normalized

    ${rng}
    ${intervals}
    ${bvh()}
    ${structs}
    ${rayIntersect}
    ${bvIntersect}
    ${scene()}
    ${raygen()}
    ${reproject()}
    ${computeColor()}
    ${bilinearInterpolation}
    ${imageSampler}
    ${geometrySampler()}
    ${matInv}
    ${derivatives()}

    var<private> quadIdx: u32;
    var<private> quad: array<u32, 4>;
    var<private> quadNeighborXIdx: u32;
    var<private> quadNeighborYIdx: u32;

    @compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE_X}, ${COMPUTE_WORKGROUP_SIZE_Y})
    fn main(
      @builtin(global_invocation_id) globalInvocationId: vec3<u32>, 
      @builtin(local_invocation_index) localInvocationIndex: u32
    ) {
      let upos = globalInvocationId.xy;
      let idx = imageIdx(upos);
      quadIdx = localInvocationIndex % 4;
      quad[0] = quadBroadcast(idx, 0);
      quad[1] = quadBroadcast(idx, 1);
      quad[2] = quadBroadcast(idx, 2);
      quad[3] = quadBroadcast(idx, 3);
      quadNeighborXIdx = quadSwapX(idx);
      quadNeighborYIdx = quadSwapY(idx);
      if (any(globalInvocationId.xy >= viewport)) {
        return;
      }

      let pos = vec2f(upos) + jitter;

      rng_state = seed + idx;
      if counter == 0u && !_reproject {
        imageBuffer[idx] = vec4f(0);
        geometryBuffer[idx].position = vec3f(0);
        geometryBuffer[idx].faceIdx = 0;
        geometryBuffer[idx].objectIdx = 0;
      }

      var color = vec3f(0);
      var samples = 0u;

      let ray = cameraRay(pos, viewMatrix);
      let hitDist = pixelHitDist(idx, ray);
      var hit: BVHIntersectionResult;
      color += pixelColor(&hit, ray, hitDist);
      samples++;

      let face = faces[hit.faceIdx];
      let uv = hit.barycentric.yz;
      let point = facePointOffset(face, uv);
      geometryBuffer[idx].position = point;
      geometryBuffer[idx].faceIdx = hit.faceIdx;
      geometryBuffer[idx].objectIdx = hit.objectIdx;

      for (var i = 0u; i < ${store.sampleCount}; i = i + 1u) {
        let pos = pos + sample_insquare(random_2()) * 0.5;
        let ray = cameraRay(pos, viewMatrix);
        let hitDist = pixelHitDist(idx, ray);
        var hit: BVHIntersectionResult;
        color += pixelColor(&hit, ray, hitDist);
        samples++;

        if _reproject {
          let face = faces[hit.faceIdx];
          let uv = hit.barycentric.yz;
          let point = facePointOffset(face, uv);
          let result = reproject(point, color);
          if result.color.w > 0 {
            color += result.color.xyz / result.color.w;
            samples++;
          }
        }
      }

      if _reproject {
        let result = reproject(point, color);
        imageBuffer[idx] = result.color;
      }

      if !${store.debugReprojection} {
        if ${store.blitView == 'normals'} {
          imageBuffer[idx] = vec4f(color, 1);
        } else {
          imageBuffer[idx] += vec4f(color, f32(samples));
        }
      }
    }
  `,
});

const { canTimestamp, querySet, submit } = getTimestampHandler((times) => {
  setRenderGPUTime(Number(times[1] - times[0]));
});

createEffect(() => {
  const { pipeline: debugBVHPipeline, bindGroups: debugBVHBindGroup } =
    renderPipeline({
      vertexShader: (x) => /* wgsl */ `
        ${structs}
  
        ${x.bindVarBuffer('read-only-storage', 'bvh: array<BoundingVolume>', bvhBuffer)}
        ${x.bindVarBuffer('uniform', 'viewProjMatrix: mat4x4f', viewProjBuffer)}
  
        const EDGES_PER_CUBE = 12u;
  
        @vertex
        fn main(
          @builtin(instance_index) instanceIndex: u32,
          @builtin(vertex_index) vertexIndex: u32
        ) -> @builtin(position) vec4f {
          let lineInstanceIdx = instanceIndex % EDGES_PER_CUBE;
          let aabbInstanceIdx = instanceIndex / EDGES_PER_CUBE;
          let a = bvh[aabbInstanceIdx];
          let aMin = a.min;
          let aMax = a.max;
          // let aMin = vec3f(0, 0, 0);
          // let aMax = vec3f(1, 1, 1);
          var pos: vec3f;
          let fVertexIndex = f32(vertexIndex);
                        
            //      a7 _______________ a6
            //       / |             /|
            //      /  |            / |
            //  a4 /   |       a5  /  |
            //    /____|__________/   |
            //    |    |__________|___|
            //    |   / a3        |   / a2
            //    |  /            |  /
            //    | /             | /
            //    |/______________|/
            //    a0              a1
  
          let dx = aMax.x - aMin.x;
          let dy = aMax.y - aMin.y;
          let dz = aMax.z - aMin.z;
          
          let a0 = aMin;
          let a1 = vec3f(aMin.x + dx, aMin.y,      aMin.z     );
          let a2 = vec3f(aMin.x + dx, aMin.y,      aMin.z + dz);
          let a3 = vec3f(aMin.x,      aMin.y,      aMin.z + dz);
          let a4 = vec3f(aMin.x,      aMin.y + dy, aMin.z     );
          let a5 = vec3f(aMin.x + dx, aMin.y + dy, aMin.z     );
          let a6 = aMax;
          let a7 = vec3f(aMin.x,      aMin.y + dy, aMin.z + dz);
  
          if (lineInstanceIdx == 0) {
            pos = mix(a0, a1, fVertexIndex);
          } else if (lineInstanceIdx == 1) {
            pos = mix(a1, a2, fVertexIndex);
          } else if (lineInstanceIdx == 2) {
            pos = mix(a2, a3, fVertexIndex);
          } else if (lineInstanceIdx == 3) {
            pos = mix(a0, a3, fVertexIndex);
          } else if (lineInstanceIdx == 4) {
            pos = mix(a0, a4, fVertexIndex);
          } else if (lineInstanceIdx == 5) {
            pos = mix(a1, a5, fVertexIndex);
          } else if (lineInstanceIdx == 6) {
            pos = mix(a2, a6, fVertexIndex);
          } else if (lineInstanceIdx == 7) {
            pos = mix(a3, a7, fVertexIndex);
          } else if (lineInstanceIdx == 8) {
            pos = mix(a4, a5, fVertexIndex);
          } else if (lineInstanceIdx == 9) {
            pos = mix(a5, a6, fVertexIndex);
          } else if (lineInstanceIdx == 10) {
            pos = mix(a6, a7, fVertexIndex);
          } else if (lineInstanceIdx == 11) {
            pos = mix(a7, a4, fVertexIndex);
          }
          return viewProjMatrix * vec4(pos, 1);
        }
        `,
      fragmentShader: () => /* wgsl */ `
        @fragment
        fn main() -> @location(0) vec4f {
          return vec4f(0.01);
          // return vec4f(0.2); 
          // return vec4f(1);
        }
      `,
      fragmentPresentationFormatTarget: {
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
          },
        },
      },
      primitive: {
        topology: 'line-list',
      },
    });

  setDebugBVHRenderBundle(
    renderBundlePass({}, (renderPass) => {
      renderPass.setPipeline(debugBVHPipeline);
      debugBVHBindGroup.forEach((bindGroup, i) =>
        renderPass.setBindGroup(i, bindGroup)
      );
      renderPass.draw(2, bvhCount * 12);
    })
  );
});

const rpd: GPURenderPassDescriptor = {
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),
      clearValue: [0, 0, 0, 0], // Clear to transparent
      loadOp: 'clear',
      storeOp: 'store',
    },
  ],
  ...(canTimestamp && {
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  }),
};

let frameCounter = 0;
export async function renderFrame(now: number) {
  const rate = store.reprojectionRate;
  const updatePrev =
    rate === 0 ||
    frameCounter % rate === 0 ||
    (store.debugReprojection || true ? false : store.counter !== 0);
  frameCounter = (frameCounter + 1) % rate;
  writeUint32Buffer(seedUniformBuffer, Math.random() * 0xffffffff);
  writeUint32Buffer(counterUniformBuffer, store.counter);
  writeUint32Buffer(updatePrevUniformBuffer, updatePrev ? 1 : 0);
  if (updatePrev) {
    const jitter = vec2.fromValues(Math.random() - 0.5, Math.random() - 0.5);
    vec2.scale(jitter, jitter, store.jitterStrength);
    writeVec2fBuffer(jitterBuffer, jitter);
  }
  incrementCounter();
  const view = viewMatrix();

  const encoder = device.createCommandEncoder();
  rpd.colorAttachments[0].view = context.getCurrentTexture().createView();

  // raytrace
  computePass(encoder, {}, (computePass) => {
    computePass.setPipeline(computePipeline());
    computeBindGroups().forEach((bindGroup, i) =>
      computePass.setBindGroup(i, bindGroup)
    );
    computePass.dispatchWorkgroups(
      Math.ceil(canvas.width / COMPUTE_WORKGROUP_SIZE_X),
      Math.ceil(canvas.height / COMPUTE_WORKGROUP_SIZE_Y),
      1
    );
  });

  renderPass(encoder, rpd, (renderPass) => {
    renderPass.executeBundles([blitRenderBundle()]);

    // debug BVH
    if (store.debugBVH) {
      renderPass.executeBundles([debugBVHRenderBundle()]);
    }
  });

  if (updatePrev) {
    encoder.copyBufferToBuffer(jitterBuffer, prevJitterBuffer);
    encoder.copyBufferToBuffer(imageBuffer(), prevImageBuffer());
    encoder.copyBufferToBuffer(geometryBuffer(), prevGeometryBuffer());
    encoder.copyBufferToBuffer(viewBuffer, prevViewBuffer2);
  }

  await submit(encoder, () => {
    device.queue.submit([encoder.finish()]);
  });

  await device.queue.onSubmittedWorkDone();
  setRenderJSTime(performance.now() - now);
  if (updatePrev) {
    setPrevView(view);
  }
}

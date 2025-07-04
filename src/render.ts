import { mat4, vec2 } from 'gl-matrix';
import {
  computePass,
  createStorageBuffer,
  createUniformBuffer,
  getDevice,
  getTimestampHandler,
  PipelineBuilder,
  reactiveComputePipeline,
  reactiveUniformBuffer,
  renderBundlePass,
  renderPass,
  renderPipeline,
  writeUint32Buffer,
} from './gpu';
import {
  incrementCounter,
  ProjectionType,
  reprojectionFrustrum,
  setRenderGPUTime,
  setRenderJSTime,
  setView,
  ShadingType,
  store,
  viewMatrix,
  viewProjectionMatrix,
} from './store';
import { createEffect, createMemo, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';
import {
  loadMaterialsToBuffers,
  loadModels,
  loadModelsToBuffers,
} from './scene';
import { usePrevious } from './utils';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu');
const device = await getDevice(context as GPUCanvasContext);
const [imageBuffer, setImageBuffer] = createSignal<GPUBuffer>();
const [depthBuffer, setDepthBuffer] = createSignal<GPUBuffer>();
const [prevImageBuffer, setPrevImageBuffer] = createSignal<GPUBuffer>();
const [blitRenderBundle, setBlitRenderBundle] = createSignal<GPURenderBundle>();
const [debugBVHRenderBundle, setDebugBVHRenderBundle] =
  createSignal<GPURenderBundle>();
const [prevView, setPrevView] = createSignal<mat4>(undefined, {
  equals: (a, b) => a && mat4.exactEquals(a, b),
});
const prevViewInv = createMemo(() => {
  const _prevView = prevView();
  if (_prevView) {
    return mat4.invert(mat4.create(), _prevView);
  }
  return mat4.create();
});
const prevViewBuffer = reactiveUniformBuffer(16, prevView);
const prevViewInvBuffer = reactiveUniformBuffer(16, prevViewInv);
const viewBuffer = reactiveUniformBuffer(16, viewMatrix);
const viewProjBuffer = reactiveUniformBuffer(16, viewProjectionMatrix);
const reprojectionFrustrumBuffer = reactiveUniformBuffer(
  12,
  reprojectionFrustrum(prevView)
);

const { models, materials } = await loadModels();
const { materialsBuffer } = await loadMaterialsToBuffers(materials);
const { facesBuffer, bvhBuffer, bvhCount, modelsBuffer } =
  await loadModelsToBuffers([
    models[2],
    models[10],
    models[6],
    models[3],
    models[4],
  ]);

const seedUniformBuffer = createUniformBuffer(4);
const counterUniformBuffer = createUniformBuffer(4);

const resize = () => {
  const scale = devicePixelRatio * store.resolutionScale;
  canvas.width = canvas.clientWidth * scale;
  canvas.height = canvas.clientHeight * scale;
  setView(vec2.fromValues(canvas.width, canvas.height));
};

resize();
window.addEventListener('resize', resize);

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

  // depth + prev depth
  const depthImageSize = Float32Array.BYTES_PER_ELEMENT * 2 * width * height;
  const depth = createStorageBuffer(
    depthImageSize,
    'Depth Buffer',
    GPUBufferUsage.COPY_SRC
  );
  setDepthBuffer(depth);

  return [current, prev, depth];
});

createEffect(() => {
  const { pipeline, bindGroups } = renderPipeline({
    vertexShader: () => /* wgsl */ `
      // xy pos + uv
      const FULLSCREEN_QUAD = array<vec4<f32>, 6>(
        vec4(-1, 1, 0, 0),
        vec4(-1, -1, 0, 1),
        vec4(1, -1, 1, 1),
        vec4(-1, 1, 0, 0),
        vec4(1, -1, 1, 1),
        vec4(1, 1, 1, 0)
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
      ${x.bindVarBuffer('read-only-storage', 'depthBuffer: array<vec2f>', depthBuffer())}
      ${x.bindVarBuffer('read-only-storage', 'prevImageBuffer: array<vec4f>', prevImageBuffer())}

      const viewport = vec2u(${store.view[0]}, ${store.view[1]});
      const viewportf = vec2f(viewport);

      fn getColor(idx: u32, pos: vec2f) -> vec3f {
        if ${store.blitView == 'image'} {
          let value = imageBuffer[idx];
          return value.rgb / value.w; 
        } else if ${store.blitView == 'prevImage'} {
          let value = prevImageBuffer[idx];
          return value.rgb / value.w; 
        } else if ${store.blitView == 'normals'} {
          let value = imageBuffer[idx];
          return value.rgb; 
        } else if ${store.blitView == 'depth'} {
          return vec3f(depthBuffer[idx][0]) / 10;
        } else if ${store.blitView == 'prevDepth'} {
          return vec3f(depthBuffer[idx][1]) / 10;
        } else if ${store.blitView == 'depthDelta'} {
          return vec3f(depthBuffer[idx][0] - depthBuffer[idx][1]);
        }
        return vec3f(0);
      }

      @fragment
      fn main(@location(0) _uv: vec2f) -> @location(0) vec4f {
        let uv = vec2f(_uv.x, 1 - _uv.y); // flip y
        let pos = uv * viewportf;
        let upos = vec2u(pos);
        let idx = upos.y * viewport.x + upos.x;
        let color = getColor(idx, pos);
        let tonemapped = color;
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
    renderPass.draw(6);
  });

  setBlitRenderBundle(bundle);
});

const COMPUTE_WORKGROUP_SIZE_X = 16;
const COMPUTE_WORKGROUP_SIZE_Y = 16;

const structs = /* wgsl */ `
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

const bvh = (x: PipelineBuilder) => /* wgsl */ `
  ${x.bindVarBuffer('read-only-storage', 'bvh: array<BoundingVolume>', bvhBuffer)}

  struct BVHIntersectionResult {
    hit: bool,
    barycentric: vec3f,
    faceIdx: u32,
  }

  struct BVHIntersectionStackEntry {
    idx: u32,
    t: f32,
  }
  const BV_MAX_STACK_DEPTH = 16;
  @must_use
  fn rayIntersectBVH(
    ray: Ray,
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(f32max, 0, 0);
    result.hit = false;
    result.faceIdx = 0;
    
    var stack: array<BVHIntersectionStackEntry, BV_MAX_STACK_DEPTH>;
    var top: i32;

    for (var modelIdx = 0u; modelIdx < modelsCount; modelIdx++) {
      let model = models[modelIdx];
      let bv = bvh[model.bvh.offset];
      let bvResult = rayIntersectBV(ray, bv, Interval(min_dist, result.barycentric.x));
      if (!bvResult.hit) {
        continue;
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
    }

    return result;
  }
`;

const raygen = () => /* wgsl */ `
  const cameraFovAngle = ${store.fov};
  const cameraRayZ = -1/tan(cameraFovAngle / 2.f);
  const paniniDistance = ${store.paniniDistance};
  const lensFocusDistance = ${store.focusDistance};
  const circleOfConfusionRadius = ${store.circleOfConfusion};
  const projectionType = ${store.projectionType};

  fn pinholeRayDirection(pixel: vec2f) -> vec3f {
    return normalize(vec3(pixel, cameraRayZ));
  }

  fn paniniRayDirection(pixel: vec2f) -> vec3f {
    let halfFOV = cameraFovAngle / 2.f;
    let p = vec2(sin(halfFOV), cos(halfFOV) + paniniDistance);
    let M = sqrt(dot(p, p));
    let halfPaniniFOV = atan2(p.x, p.y);
    let hvPan = pixel * vec2(halfPaniniFOV, halfFOV);
    let x = sin(hvPan.x) * M;
    let z = cos(hvPan.x) * M - paniniDistance;
    // let y = tan(hvPan.y) * (z + verticalCompression);
    let y = tan(hvPan.y) * (z + pow(max(0., (3. * cameraFovAngle/PI - 1.) / 8.), 0.92));

    return normalize(vec3(x, y, z));
  }

  fn orthographicRayDirection(uv: vec2f) -> vec3f {
    return vec3(0, 0, 1);
  }

  fn thinLensRay(dir: vec3f, uv: vec2f) -> Ray {
    let pos = vec3(uv * circleOfConfusionRadius, 0.f);
    return Ray(
      pos,
      normalize(dir * lensFocusDistance - pos)
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
      default: {
        return vec3(0);
      }
    }
  }

  fn ray_transform(_ray: Ray, view: mat4x4f) -> Ray {
    var ray = _ray;
    let ray_pos = view * vec4(ray.pos, 1.);
    ray.pos = ray_pos.xyz;
    ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
    ray.dir = (view * vec4(ray.dir, 0.)).xyz;
    return ray;
  }

  fn cameraRay(uv: vec2f, view: mat4x4f) -> Ray {
    let rayDirection = cameraRayDirection(uv);
    
    let ray = thinLensRay(rayDirection, sample_incircle(random_2()));
    return ray_transform(ray, view);
  }
`;

const scene = () => /* wgsl */ `
  const flatShading = ${store.shadingType};
    
  const ambience = ${store.ambience};
  const sun_color = vec3f(1);
  const sun_dir = normalize(vec3f(1, 1, 1));
  const sphere_center = vec3f(0, 0, 4);

  struct Hit {
    hit: bool,
    dist: f32,
    point: vec3f,
    normal: vec3f,
    materialIdx: u32, 
    uv: vec2f
  };

  fn scene(ray: Ray) -> Hit {
    let hit = rayIntersectBVH(ray);
    if !hit.hit {
      let result = Hit(
        false,
        max_dist,
        ray.pos,
        vec3f(0),
        0,
        vec2f(0)
      );
      return result;
    }
    
    let face = faces[hit.faceIdx];
    let uv = hit.barycentric.yz;
    let result = Hit(
      true,
      hit.barycentric.x,
      facePoint(face, uv),
      faceNormal(face, uv),
      face.materialIdx,
      faceTexCoords(face, uv)
    );

    return result;
  }

  struct SceneSample {
    p: f32, // 1/pdf
    point: vec3f,
    normal: vec3f,
    uv: vec2f,
    materialIdx: u32,
  }

  fn sampleScene() -> SceneSample {
    let randomModelIdx = random_1u() % modelsCount; 
    let model = models[randomModelIdx];
    var sample = sampleModel(model);
    sample.p *= f32(modelsCount);
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
    let point = facePoint(face, uv);
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

  fn in_shadow(ray: Ray, mag_sq: f32) -> f32 {
    let hitObj = scene(ray);
    let hit = hitObj.hit;
    let ds = hitObj.dist;

    return select(1., 0., hit && ds * ds <= mag_sq);
  }

  fn light_vis(pos: vec3f, dir: vec3f) -> f32 {
    return in_shadow(Ray(pos, dir), f32max);
  }

  fn attenuation(dir: vec3f, norm: vec3f) -> f32 { 
    return max(dot(dir, norm), 0.);
  }

  fn sun(pos: vec3f, norm: vec3f) -> vec3f {
    return (light_vis(pos, sun_dir) * attenuation(sun_dir, norm) + ambience) * sun_color;
  }

  fn skyColor(dir: vec3f) -> vec3f {
    return ambience * sun_color;
  }
`;

const reproject = /* wgsl */ `
  struct ReprojectionResult {
    color: vec4f, // color + accumulated samples count
  }

  const m = mat4x4f(
    1, 0, 1, 0,
    0, 1, 0, 1,
    -1, 0, 1, 0,
    0, -1, 0, 1
  );
  fn reprojectPoint(p: vec3f, view: mat4x4f, viewInv: mat4x4f) -> vec2f {
    let pp = p - view[3].xyz;
    let duv = reprojectionFrustrum * pp;
    return duv.xy / duv.zw;  
  }
  fn reproject(p: vec3f, view: mat4x4f, viewInv: mat4x4f) -> ReprojectionResult {
    let _uv = reprojectPoint(p, view, viewInv);
    if any(_uv < -viewportN) || any(_uv > viewportN) { // outside viewport
      return ReprojectionResult(vec4f(0));
    }

    // return ReprojectionResult(vec4f(_uv, 0, 1));

    let uv = (_uv * viewportf.x + viewportf)/2;
    let uv2 = vec2u(round(uv - vec2f(0.5)));
    // let min_idx = uv2.y * viewport.x + uv2.x;
    let uv_a = array<vec2u, 9>(
      vec2u(uv2.x, uv2.y),
      vec2u(uv2.x + 1, uv2.y),
      vec2u(uv2.x, uv2.y + 1),
      vec2u(uv2.x + 1, uv2.y + 1),
      vec2u(uv2.x - 1, uv2.y),
      vec2u(uv2.x, uv2.y - 1),
      vec2u(uv2.x - 1, uv2.y - 1),
      vec2u(uv2.x + 1, uv2.y - 1),
      vec2u(uv2.x - 1, uv2.y + 1),
    );
    var min_d = f32max;
    var min_idx = 0u;
    for (var i = 0u; i < 9u; i = i + 1u) {
      let idx = uv_a[i].y * viewport.x + uv_a[i].x;
      let prevDepth = depthBuffer[idx][1];

      // let prevCameraPos = view[3].xyz;
      // let d2_len = length(p - prevCameraPos);
      // if !(abs(d2_len - prevDepth) < 0.001) {
      //   return ReprojectionResult(false, 0);
      // }

      let old_ray = cameraRay(_uv, view);
      let old_p = old_ray.pos + old_ray.dir * prevDepth;
      let dp = p - old_p;
      let d = dot(dp, dp);
      if d < min_d {
        min_d = d;
        min_idx = idx;
      }
    }

    if !(min_d < 0.001) { // depth is not the same
      return ReprojectionResult(vec4f(0));
    }

    return ReprojectionResult(prevImageBuffer[min_idx]);
  }
`;

const [computePipeline, computeBindGroups] = reactiveComputePipeline({
  shader: (x) => /* wgsl */ `
    ${x.bindVarBuffer('storage', 'prevImageBuffer: array<vec4f>', prevImageBuffer())}
    ${x.bindVarBuffer('storage', 'imageBuffer: array<vec4f>', imageBuffer())}
    ${x.bindVarBuffer('storage', 'depthBuffer: array<vec2f>', depthBuffer())}
    ${x.bindVarBuffer('uniform', 'viewMatrix: mat4x4f', viewBuffer)}
    ${x.bindVarBuffer('uniform', 'prevViewMatrix: mat4x4f', prevViewBuffer)}
    ${x.bindVarBuffer('uniform', 'prevViewInvMatrix: mat4x4f', prevViewInvBuffer)}
    ${x.bindVarBuffer('uniform', 'reprojectionFrustrum: mat3x4f', reprojectionFrustrumBuffer)}

    const modelsCount = ${models.length};
    ${x.bindVarBuffer('read-only-storage', 'faces: array<Face>', facesBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'materials: array<Material>', materialsBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'models: array<Model>', modelsBuffer)}


    ${x.bindVarBuffer('uniform', 'seed: u32', seedUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'counter: u32', counterUniformBuffer)}

    const viewport = vec2u(${store.view[0]}, ${store.view[1]});
    const viewportf = vec2f(viewport);
    const aspect = viewportf.y / viewportf.x;
    const viewportN = viewportf / viewportf.x; // viewport normalized

    ${rng}
    ${intervals}
    ${bvh(x)}
    ${structs}
    ${rayIntersect}
    ${bvIntersect}
    ${scene()}
    ${raygen()}
    ${reproject}


    fn computeColor(pos: vec2f, _hit: ptr<function, Hit>) -> vec3f {
      let uv = (2. * pos - viewportf) / viewportf.x;
      let ray = cameraRay(uv, viewMatrix);
      let hit = scene(ray);
      *_hit = hit;
      if !hit.hit {
        return skyColor(ray.dir);
      }

      if ${store.blitView === 'normals'} {
        return (hit.normal + 1) / 2;
      }

      let material = materials[hit.materialIdx];
      let p = hit.point;
      var color = vec3f(0);
      color += sun(p, hit.normal) * material.color;
      color += material.emission;

      let s = sampleLights();
      let sMaterial = materials[s.materialIdx];
      let ds = s.point - p;
      let d_sq = dot(ds, ds);
      let d = ds * inverseSqrt(d_sq);
      let r = Ray(p, d);
      color += in_shadow(r, d_sq) * attenuation(d, hit.normal) * sMaterial.emission * material.color / d_sq * s.p;

      return color;
    }

    @compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE_X}, ${COMPUTE_WORKGROUP_SIZE_Y})
    fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
      if (any(globalInvocationId.xy >= viewport)) {
        return;
      }

      let upos = globalInvocationId.xy;
      let idx = upos.x + upos.y * viewport.x;
      let pos = vec2f(upos);
      rng_state = seed + idx;
      if counter == 0u && !${store.reproject} {
        depthBuffer[idx] = vec2f(0);
        imageBuffer[idx] = vec4f(0);
        prevImageBuffer[idx] = vec4f(0);
      }

      let prevDepth = depthBuffer[idx][0];
      let prevColor = imageBuffer[idx];

      var color = vec3f(0);
      var hit: Hit;
      var samples = 0u;

      color += computeColor(pos, &hit);
      samples++;
      depthBuffer[idx][0] = hit.dist;

      for (var i = 0u; i < ${store.sampleCount}; i = i + 1u) {
        let jitter = sample_insquare(random_2());
        color += computeColor(pos + jitter, &hit);
        samples++;
      }

      if ${store.reproject} {
        let result = reproject(hit.point, prevViewMatrix, prevViewInvMatrix);
        imageBuffer[idx] = result.color;
      }

      prevImageBuffer[idx] = prevColor;
      depthBuffer[idx][1] = prevDepth;
      if ${store.blitView == 'normals'} {
        imageBuffer[idx] = vec4f(color, 1);
      } else {
        imageBuffer[idx] += vec4f(color, f32(samples));
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

export function renderFrame(now: number) {
  // if (store.counter !== 0) {
  setPrevView(viewMatrix());
  // }
  writeUint32Buffer(seedUniformBuffer, Math.random() * 0xffffffff);
  writeUint32Buffer(counterUniformBuffer, store.counter);
  incrementCounter();
  // if (prevView()) {
  //   // const p = mat4.multiply(mat4.create(), prevViewInv(), prevView());
  //   // console.log(mat4.exactEquals(p, mat4.create()));
  //   // console.log(mat4.exactEquals(prevView(), viewMatrix()));
  // }

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

  submit(encoder, () => {
    device.queue.submit([encoder.finish()]);
  });
  setRenderJSTime(performance.now() - now);
}

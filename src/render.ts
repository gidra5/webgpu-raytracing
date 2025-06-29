import { vec2 } from 'gl-matrix';
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
} from './gpu';
import {
  incrementCounter,
  ProjectionType,
  setRenderGPUTime,
  setRenderJSTime,
  setView,
  ShadingType,
  store,
  view,
  viewProjectionMatrix,
} from './store';
import { createEffect, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';
import {
  loadMaterialsToBuffers,
  loadModels,
  loadModelsToBuffers,
} from './scene';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu');
const device = await getDevice(context as GPUCanvasContext);
const [imageBuffer, setImageBuffer] = createSignal<GPUBuffer>();
const [blitRenderBundle, setBlitRenderBundle] = createSignal<GPURenderBundle>();
const [debugBVHRenderBundle, setDebugBVHRenderBundle] =
  createSignal<GPURenderBundle>();
const viewBuffer = reactiveUniformBuffer(16, view);
const viewProjBuffer = reactiveUniformBuffer(16, viewProjectionMatrix);

const { models, materials } = await loadModels();
const { materialsBuffer } = await loadMaterialsToBuffers(materials);
const { facesBuffer, bvhBuffer, bvhCount, modelsBuffer } =
  await loadModelsToBuffers(models);

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

createEffect<GPUBuffer>((prevBuffer) => {
  if (prevBuffer) prevBuffer.destroy();
  const width = store.view[0] + 1;
  const height = store.view[1];
  const size = Float32Array.BYTES_PER_ELEMENT * 4 * width * height;
  const buffer = createStorageBuffer(size, 'Raytraced Image Buffer');
  setImageBuffer(buffer);
  return buffer;
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

      ${x.bindVarBuffer('read-only-storage', 'imageBuffer: array<vec3f>', imageBuffer())}
      ${x.bindVarBuffer('uniform', 'counter: u32', counterUniformBuffer)}
      // @group(0) @binding(1) var<uniform> commonUniforms: CommonUniforms;

      const viewport = vec2u(${store.view[0]}, ${store.view[1]});

      @fragment
      fn main(@location(0) _uv: vec2<f32>) -> @location(0) vec4f {
        let uv = vec2f(_uv.x, 1. - _uv.y);
        let pos = vec2u(uv * vec2f(viewport));
        // let idx = fma(pos.y, viewport.x, pos.x);
        let idx = pos.y * viewport.x + pos.x; 
        let color = imageBuffer[idx] / f32(counter + ${store.sampleCount});
        // let color = imageBuffer[idx];
        let tonemapped = color;
        // let color = lottes(raytraceImageBuffer[idx] / f32(commonUniforms.frameCounter + 1));
        // return aces(raytraceImageBuffer[idx] / f32(commonUniforms.frameCounter + 1));
        // return vec4(aces(imageBuffer[idx]), 1.0);
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

const intersect = /* wgsl */ `
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

const bvh = /* wgsl */ `
  struct BVHIntersectionResult {
    hit: bool,
    barycentric: vec3f,
    faceIdx: u32,
  }
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

const raygen = /* wgsl */ `
  fn pinholeRayDirection(pixel: vec2f) -> vec3f {
    return normalize(vec3(pixel, -1/tan(cameraFovAngle / 2.f)));
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
`;

const scene = /* wgsl */ `
  struct Hit {
    hit: bool,
    dist: f32,
    point: vec3f,
    normal: vec3f,
    material: Material
  };

  fn scene(ray: Ray) -> Hit {
    let hit = rayIntersectBVH(ray);
    if !hit.hit {
      let result = Hit(
        false,
        max_dist,
        ray.pos,
        vec3f(0),
        Material(
          vec3(1.),
          vec3(0.),
        )
      );
      return result;
    }
    
    let face = faces[hit.faceIdx];
    var result = Hit(
      true,
      hit.barycentric.x,
      ray.pos + hit.barycentric.x * ray.dir,
      face.faceNormal,
      Material(
        vec3(1.),
        vec3(0.),
      )
    );

    if (flatShading == ${ShadingType.Phong}) {
      let n1 = face.points[0].normal;
      let n2 = face.points[1].normal;
      let n3 = face.points[2].normal;
      let _n = mat3x3f(n1, n2, n3);

      let pt = hit.barycentric;
      let b = vec3f(1f - pt.y - pt.z, pt.y, pt.z);
      let n = _n * b;
      result.normal = n;
    }

    return result;
  }

  fn in_shadow(ray: Ray, mag_sq: f32) -> f32 {
    let hitObj = scene(ray);
    let hit = hitObj.hit;
    let ds = hitObj.dist;

    return select(1., 0., !hit || ds * ds >= mag_sq);
  }

  fn light_vis(pos: vec3f, dir: vec3f) -> f32 {
    return in_shadow(Ray(pos, dir), 0.99 / (min_dist * min_dist));
  }

  fn sun_light_col(dir: vec3f, norm: vec3f) -> vec3f { 
    return max(dot(-dir, norm), 0.) * sun_color;
  }
  fn sun(pos: vec3f, norm: vec3f) -> vec3f {
    return light_vis(pos, sun_dir) * sun_light_col(sun_dir, norm) + ambience * sun_color;
  }

  fn ray_transform(_ray: Ray) -> Ray {
    var ray = _ray;
    let ray_pos = u_view * vec4(ray.pos, 1.);
    ray.pos = ray_pos.xyz;
    ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
    ray.dir = (u_view * vec4(ray.dir, 0.)).xyz;
    return ray;
  }

  fn skyColor(dir: vec3f) -> vec3f {
    return vec3f(0.1);
  }
`;

const [computePipeline, computeBindGroups] = reactiveComputePipeline({
  shader: (x) => /* wgsl */ `
    ${x.bindVarBuffer('storage', 'imageBuffer: array<vec3f>', imageBuffer())}
    ${x.bindVarBuffer('uniform', 'u_view: mat4x4f', viewBuffer)}

    ${rng}
    ${intervals}
    ${bvh}
    ${structs}

    const modelsCount = ${models.length};
    ${x.bindVarBuffer('read-only-storage', 'faces: array<Face>', facesBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'bvh: array<BoundingVolume>', bvhBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'materials: array<Material>', materialsBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'models: array<Model>', modelsBuffer)}


    ${x.bindVarBuffer('uniform', 'seed: u32', seedUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'counter: u32', counterUniformBuffer)}
    
    const cameraFovAngle = ${store.fov};
    const paniniDistance = ${store.paniniDistance};
    const lensFocusDistance = ${store.focusDistance};
    const circleOfConfusionRadius = ${store.circleOfConfusion};
    const flatShading = ${store.shadingType};
    const debugNormals = ${store.debugNormals};
    const projectionType = ${store.projectionType};
    
    const ambience = ${store.ambience};
    const sun_color = vec3f(1);
    const sun_dir = normalize(vec3f(-1, 1, 1));
    const sphere_center = vec3f(0, 0, 4);

    const viewport = vec2u(${store.view[0]}, ${store.view[1]});
    const viewportf = vec2f(viewport);


    ${intersect}
    ${scene}
    ${raygen}

    @compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE_X}, ${COMPUTE_WORKGROUP_SIZE_Y})
    fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
      if (any(globalInvocationId.xy >= viewport)) {
        return;
      }

      var color = vec3f(0);

      let upos = globalInvocationId.xy;
      let idx = upos.x + upos.y * viewport.x;
      let pos = vec2f(upos);
      rng_state = seed + idx;

      
      if (counter == 0u) {
        imageBuffer[idx] = vec3f(0);
      }

      for (var i = 0u; i < ${store.sampleCount}; i++) {
        let subpixel = random_2();
        let uv = (2. * (pos + subpixel) - viewportf) / viewportf.x;
        let rayDirection = cameraRayDirection(uv);
        
        var ray = thinLensRay(rayDirection, sample_incircle(random_2()));
        ray = ray_transform(ray);

        let hitObj = scene(ray);
        let t = hitObj.dist;
        if !hitObj.hit {
          imageBuffer[idx] = skyColor(ray.dir);
          continue; 
        }

        if debugNormals {
          color = (hitObj.normal+1)/2;
        } else {
          color += sun(ray.pos + t * ray.dir, hitObj.normal) * hitObj.material.color; 
        }

        imageBuffer[idx] += color;
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
  writeUint32Buffer(seedUniformBuffer, Math.random() * 0xffffffff);
  writeUint32Buffer(counterUniformBuffer, store.counter);
  incrementCounter();

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

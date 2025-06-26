import { mat4, vec2 } from 'gl-matrix';
import {
  computePass,
  computePipeline,
  createStorageBuffer,
  createUniformBuffer,
  getDevice,
  getTimestampHandler,
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
} from './store';
import { createEffect, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';
import { loadModels, loadModelFacesToBuffer } from './scene';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu');
const device = await getDevice(context as GPUCanvasContext);
const [imageBuffer, setImageBuffer] = createSignal<GPUBuffer>();
const [blitRenderBundle, setBlitRenderBundle] = createSignal<GPURenderBundle>();
const viewBuffer = reactiveUniformBuffer(16, () => {
  const m = mat4.fromRotationTranslation(
    mat4.create(),
    store.orientation,
    store.position
  );
  const m2 = mat4.create();
  m2[1 + 4 * 1] = -1;
  mat4.multiply(m, m, m2);
  return m;
});
const [_computePipeline, setComputePipeline] =
  createSignal<GPUComputePipeline>();
const [computeBindGroups, setComputeBindGroups] =
  createSignal<GPUBindGroup[]>();

const modelIds = await loadModels();
const [facesBuffer, facesOffsetBuffer, facesLength] =
  await loadModelFacesToBuffer();
const seedUniformBuffer = createUniformBuffer(4);
const counterUniformBuffer = createUniformBuffer(4);

const resize = () => {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  setView(vec2.fromValues(canvas.width, canvas.height));
};

resize();
window.addEventListener('resize', () => {
  resize();
});

createEffect<GPUBuffer>((prevBuffer) => {
  if (prevBuffer) prevBuffer.destroy();
  const size =
    Float32Array.BYTES_PER_ELEMENT * 4 * store.view[0] * store.view[1];
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
      fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4f {
        let pos = vec2u(uv * vec2f(viewport));
        // let idx = fma(pos.y, viewport.x, pos.x);
        let idx = pos.y * viewport.x + pos.x; 
        let color = imageBuffer[idx] / f32(counter + 1);
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

const intervals = /* wgsl */ `
  struct Interval {
    min: f32,
    max: f32,
  };
  
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
  struct IntersectonResult {
    hit: bool,
    barycentric: vec3f,
  }

  @must_use
  fn rayIntersectFace(
    ray: Ray,
    face: Face,
    interval: Interval
  ) -> IntersectonResult {
    var rec: IntersectonResult;
    rec.hit = false;

    // Mäller-Trumbore algorithm
    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    
    let pn0 = transpose(face.points[0].posNormalT);
    let pn1 = transpose(face.points[1].posNormalT);
    let pn2 = transpose(face.points[2].posNormalT);
    let p0 = pn0[0];
    let e1 = pn1[0];
    let e2 = pn2[0];

    let h = cross(ray.dir, e2);
    let det = dot(e1, h);
    
    if abs(det) < EPSILON * EPSILON {
      return rec;
    }

    let s = ray.pos - p0;
    let u = dot(s, h);

    if u < 0.0f || u > det {
      return rec;
    }

    let q = cross(s, e1);
    let v = dot(ray.dir, q);

    if v < 0.0f || u + v > det {
      return rec;
    }

    let t = dot(e2, q);
    let pt = vec3f(t, u, v) / det;

    if !intervalSurrounds(interval, pt.x) {
      return rec;
    }

    rec.barycentric = pt;
    rec.hit = true;
    
    return rec;
  }
`;

createEffect(() => {
  const { pipeline, bindGroups } = computePipeline({
    shader: (x) => /* wgsl */ `
      ${x.bindVarBuffer('storage', 'imageBuffer: array<vec3f>', imageBuffer())}
      ${x.bindVarBuffer('uniform', 'u_view: mat4x4f', viewBuffer)}

      ${rng}
      ${intervals}

      struct FacePoint {
        posNormalT: mat3x2f // pos and normal packed into transposed 2x3 matrix. That way we don't waste space on alignment
      }
      struct Face {
        faceNormal: vec3f,
        materialIdx: u32,
        points: array<FacePoint, 3>
      }

      ${x.bindVarBuffer('read-only-storage', 'faces: array<Face>', facesBuffer)}
      const facesLength = ${facesLength};

      ${x.bindVarBuffer('uniform', 'seed: u32', seedUniformBuffer)}
      ${x.bindVarBuffer('uniform', 'counter: u32', counterUniformBuffer)}
      
      const cameraFovAngle = ${store.fov};
      const paniniDistance = ${store.paniniDistance};
      const lensFocusDistance = ${store.focusDistance};
      const circleOfConfusionRadius = ${store.circleOfConfusion};
      const flatShading = ${store.shadingType};
      const projectionType = ${store.projectionType};
      
      const ambience = ${store.ambience};
      const sun_color = vec3f(1);
      const sun_dir = normalize(vec3f(-1, 1, 1));
      const sphere_center = vec3f(0, 0, 4);

      const viewport = vec2u(${store.view[0]}, ${store.view[0]});
      const viewportf = vec2f(viewport);

      struct Material {
        color: vec3f,
        emission: vec3f
      };

      struct Ray {
        pos: vec3f, // Origin
        dir: vec3f, // Direction (normalized)
      };

      struct Hit {
        normal: vec3f,
        point: vec3f,
        dist: f32,
        hit: bool,
        material: Material
      };

      ${intersect}

      fn scene(ray: Ray) -> Hit {
        var hitObj: Hit;
        hitObj.hit = false;
        hitObj.dist = max_dist;
        hitObj.point = ray.pos;
        hitObj.material.color = vec3(1.);
        hitObj.material.emission = vec3(0.);

        for (var i = 0u; i < facesLength; i = i + 1) {
          let face = faces[i];
          let hit = rayIntersectFace(ray, face, Interval(min_dist, hitObj.dist));
          if (hit.hit) {
            hitObj.hit = true;
            hitObj.dist = hit.barycentric.x;
            hitObj.point = ray.pos + hit.barycentric.x * ray.dir;
            if (flatShading == ${ShadingType.Flat}) {
              hitObj.normal = face.faceNormal;
            } else {
              let pn0 = transpose(face.points[0].posNormalT);
              let pn1 = transpose(face.points[1].posNormalT);
              let pn2 = transpose(face.points[2].posNormalT);
              let _n = mat3x3f(pn0[1], pn1[1], pn2[1]);

              let pt = hit.barycentric;
              let b = vec3f(1f - pt.y - pt.z, pt.y, pt.z);
              let n = _n * b;
              hitObj.normal = n;
            }
          }
        }

        return hitObj;
      }


      fn pinholeRay(pixel: vec2f) -> vec3f { 
        return normalize(vec3(pixel, -1/tan(cameraFovAngle / 2.f)));
      }

      fn paniniRay(pixel: vec2f) -> vec3f {
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

      fn orthographicRay(uv: vec2f) -> vec3f {
        return vec3(0, 0, 1);
      }

      fn thinLensRay(dir: vec3f, uv: vec2f) -> Ray {
        let pos = vec3(uv * circleOfConfusionRadius, 0.f);
        return Ray(
          pos,
          normalize(dir * lensFocusDistance - pos)
        );
      }

      fn cameraRay(uv: vec2f) -> vec3f {
        switch (projectionType) {
          case ${ProjectionType.Panini}: {
            return paniniRay(uv);
          }
          case ${ProjectionType.Perspective}: {
            return pinholeRay(uv);
          }
          case ${ProjectionType.Orthographic}: {
            return orthographicRay(uv);
          }
          default: {
            return vec3(0);
          }
        }
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

      @compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE_X}, ${COMPUTE_WORKGROUP_SIZE_Y})
      fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
        if (any(globalInvocationId.xy > viewport)) {
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

        let subpixel = random_2();
        let uv = (2. * (pos + subpixel) - viewportf) / viewportf.x;
        let rayDirection = cameraRay(uv);
        
        var ray = thinLensRay(rayDirection, sample_incircle(random_2()));
        ray = ray_transform(ray);

        let hitObj = scene(ray);
        let t = hitObj.dist;
        if (!hitObj.hit) { 
          // imageBuffer[idx] = vec3f(0);
          return; 
        }
        // color += sun(ray.pos + t * ray.dir, hitObj.normal) * hitObj.material.color; 
        color = (hitObj.normal+1)/2;

        imageBuffer[idx] += color;
        // imageBuffer[idx] = color;
      }
    `,
  });
  setComputeBindGroups(bindGroups);
  setComputePipeline(pipeline);
});

const { canTimestamp, querySet, submit } = getTimestampHandler((times) => {
  setRenderGPUTime(Number(times[1] - times[0]));
});

const debugBVHRenderBundle = renderBundlePass({}, (renderPass) => {
  //   renderPass.setPipeline(debugBVHPipeline);
  //   renderPass.setBindGroup(0, debugBVHBindGroup);
  //   renderPass.draw(2, Scene.AABBS_COUNT * 12);
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
    computePass.setPipeline(_computePipeline());
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
      renderPass.executeBundles([debugBVHRenderBundle]);
    }
  });

  submit(encoder, () => {
    device.queue.submit([encoder.finish()]);
  });

  setRenderJSTime(performance.now() - now);
}

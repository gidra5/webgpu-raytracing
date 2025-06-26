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
  writeBuffer,
} from './gpu';
import { setRenderGPUTime, setRenderJSTime, setView, store } from './store';
import { createEffect, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';
import { loadModel, loadModelToBuffer } from './scene';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu');
const device = await getDevice(context as GPUCanvasContext);
const [imageBuffer, setImageBuffer] = createSignal<GPUBuffer>();
const [blitRenderBundle, setBlitRenderBundle] = createSignal<GPURenderBundle>();
const viewBuffer = reactiveUniformBuffer(16, () =>
  mat4.fromRotationTranslation(mat4.create(), store.orientation, store.position)
);
const [_computePipeline, setComputePipeline] =
  createSignal<GPUComputePipeline>();
const [computeBindGroups, setComputeBindGroups] =
  createSignal<GPUBindGroup[]>();

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
      // @group(0) @binding(1) var<uniform> commonUniforms: CommonUniforms;

      const viewport = vec2u(${store.view[0]}, ${store.view[1]});

      @fragment
      fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4f {
        let pos = vec2u(uv * vec2f(viewport));
        // let idx = fma(pos.y, viewport.x, pos.x);
        let idx = pos.y * viewport.x + pos.x; 
        let color = imageBuffer[idx];
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

const models = await loadModel();
const model = models[10];
const facesBuffer = await loadModelToBuffer(model);
const facesLengthBuffer = createUniformBuffer(4);
writeBuffer(facesLengthBuffer, 0, new Uint32Array([model.faces.length]));

console.log(
  model,
  models.map((m) => m.name)
);

createEffect(() => {
  const { pipeline, bindGroups } = computePipeline({
    shader: (x) => /* wgsl */ `
      ${x.bindVarBuffer('storage', 'imageBuffer: array<vec3f>', imageBuffer())}
      ${x.bindVarBuffer('uniform', 'u_view: mat4x4f', viewBuffer)}

      ${rng}

      struct FacePoint {
        // pos: vec3f,
        // normal: vec3f
        posNormalT: mat3x2f // pos and normal packed into transposed 2x3 matrix. That way we don't waste space on alignment
      }
      struct Face {
        faceNormal: vec3f,
        materialIdx: u32,
        points: array<FacePoint, 3>
      }

      ${x.bindVarBuffer('read-only-storage', 'faces: array<Face>', facesBuffer)}
      ${x.bindVarBuffer('uniform', 'facesLength: u32', facesLengthBuffer)}

      const cameraFovAngle = ${store.fov};
      const paniniDistance = ${store.paniniDistance};
      const lensFocusDistance = ${store.focusDistance};
      const circleOfConfusionRadius = ${store.circleOfConfusion};
      
      const ambience = ${store.ambience};
      const sun_color = vec3f(1);
      const sun_dir = normalize(vec3f(-1, 1, 1));
      // const sun_dir = normalize(vec3f(0, -1, 0));
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
        throughput: vec3f
      };

      struct Hit {
        normal: vec3f,
        point: vec3f,
        dist: f32,
        hit: bool,
        material: Material
      };
        
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
        var out = x;
        if (x < interval.min) {
          out = interval.min;
        }
        if (x > interval.max) {
          out = interval.max;
        }
        return out;
      }

      const f32min = 0x1p-126f;
      const f32max = 0x1.fffffep+127;
      const emptyInterval = Interval(f32max, f32min);
      const universeInterval = Interval(f32min, f32max);
      const positiveUniverseInterval = Interval(EPSILON, f32max);

      @must_use
      fn rayIntersectFace(
        ray: Ray,
        face: Face,
        interval: Interval
      ) -> Hit {
        var rec: Hit;

        // Mäller-Trumbore algorithm
        // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm

        let fnDotRayDir = dot(face.faceNormal, ray.dir);
        if (abs(fnDotRayDir) < EPSILON) {
          rec.hit = false;
          return rec; // ray direction almost parallel
        }
        let pn0 = transpose(face.points[0].posNormalT);
        let pn1 = transpose(face.points[1].posNormalT);
        let pn2 = transpose(face.points[2].posNormalT);
        let p0 = pn0[0];
        let p1 = pn1[0];
        let p2 = pn2[0];
        let n0 = pn0[1];
        let n1 = pn1[1];
        let n2 = pn2[1];

        let e1 = p1 - p0;
        let e2 = p2 - p0;

        let h = cross(ray.dir, e2);
        let det = dot(e1, h);

        if det > -0.00001 && det < 0.00001 {
          rec.hit = false;
          return rec;
        }

        let invDet = 1.0f / det;
        let s = ray.pos - p0;
        let u = invDet * dot(s, h);

        if u < 0.0f || u > 1.0f {
          rec.hit = false;
          return rec;
        }

        let q = cross(s, e1);
        let v = invDet * dot(ray.dir, q);

        if v < 0.0f || u + v > 1.0f {
          rec.hit = false;
          return rec;
        }

        let t = invDet * dot(e2, q);

        if intervalSurrounds(interval, t) {
          // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
          
          let p = p0 + u * e1 + v * e2;

          rec.dist = t;
          rec.point = p;
          // rec.materialIdx = face.materialIdx;
          rec.material.color = vec3(1.);
          rec.material.emission = vec3(0.);
          // if (commonUniforms.flatShading == 1u) {
            // rec.normal = face.faceNormal;
          // } else {
            let b = vec3f(1f - u - v, u, v);
            let n = b[0] * n0 + b[1] * n1 + b[2] * n2;
            rec.normal = n;
          // }
          rec.hit = true;
        } else {
          rec.hit = false;
        }
        return rec;
      }

      fn scene(ray: Ray) -> Hit {
        var hitObj: Hit;
        hitObj.hit = false;
        hitObj.dist = max_dist;
        hitObj.point = ray.pos;
        hitObj.material.color = vec3(1.);
        hitObj.material.emission = vec3(0.);

        let ray2 = Ray(ray.pos -  4* sphere_center, ray.dir, vec3f(0.));

        for (var i = 0u; i < facesLength; i = i + 1) {
          let face = faces[i];
          // let hit = rayIntersectFace(ray2, face, Interval(min_dist, hitObj.dist));
          let hit = rayIntersectFace(ray, face, Interval(min_dist, hitObj.dist));
          if (hit.hit) {
            hitObj = hit;
          }
        }

        return hitObj;
      }


      fn pinholeRay(pixel: vec2f) -> vec3f { 
        return normalize(vec3(pixel, 1/tan(cameraFovAngle / 2.f)));
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

      fn thinLensRay(dir: vec3f, uv: vec2f) -> Ray {
        let pos = vec3(uv * circleOfConfusionRadius, 0.f);
        return Ray(
          pos,
          normalize(dir * lensFocusDistance - pos),
          vec3(1.)
        );
      }

      fn in_shadow(ray: Ray, mag_sq: f32) -> f32 {
        let hitObj = scene(ray);
        let hit = hitObj.hit;
        let ds = hitObj.dist;

        return select(1., 0., !hit || ds * ds >= mag_sq);
      }

      fn light_vis(pos: vec3f, dir: vec3f) -> f32 {
        return in_shadow(Ray(pos, dir, vec3(1.)), 0.99 / (min_dist * min_dist));
      }

      fn sun_light_col(dir: vec3f, norm: vec3f) -> vec3f { 
        return max(dot(-dir, norm), 0.) * sun_color;
      }
      fn sun(pos: vec3f, norm: vec3f) -> vec3f {
        return light_vis(pos, sun_dir) * sun_light_col(sun_dir, norm) + ambience * sun_color;
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
        imageBuffer[idx] = vec3f(0);

        let subpixel = random_2();
        let uv = (2. * (pos + subpixel) - viewportf) / viewportf.x;
        // let rayDirection = paniniRay(uv);
        let rayDirection = pinholeRay(uv);
        
        var ray = thinLensRay(rayDirection, sample_incircle(random_2()));

        let ray_pos = u_view * vec4(ray.pos, 1.);
        ray.pos = ray_pos.xyz;
        ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
        ray.dir = (u_view * vec4(ray.dir, 0.)).xyz;
        // why
        ray.dir = ray.dir * vec3(1,-1,-1);
        ray.throughput = vec3(1.);

        let hitObj = scene(ray);
        let t = hitObj.dist;
        if (!hitObj.hit) { return; }
        // color += sun(ray.pos + t * ray.dir, hitObj.normal) * hitObj.material.color; 
        color = (hitObj.normal+1)/2;

        imageBuffer[idx] = color;
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

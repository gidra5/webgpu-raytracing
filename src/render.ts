import { mat4, vec2 } from 'gl-matrix';
import {
  computePass,
  computePipeline,
  createStorageBuffer,
  getDevice,
  getTimestampHandler,
  reactiveUniformBuffer,
  renderBundlePass,
  renderPass,
  renderPipeline,
} from './gpu';
import { setRenderGPUTime, setRenderJSTime, setView, store } from './store';
import { createEffect, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';

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

createEffect(() => {
  const { pipeline, bindGroups } = computePipeline({
    shader: (x) => /* wgsl */ `
          ${x.bindVarBuffer('storage', 'imageBuffer: array<vec3f>', imageBuffer())}
          ${x.bindVarBuffer('uniform', 'u_view: mat4x4f', viewBuffer)}

          ${rng}

          const cameraFovAngle = ${store.fov};
          const paniniDistance = ${store.paniniDistance};
          const lensFocusDistance = ${store.focusDistance};
          const circleOfConfusionRadius = ${store.circleOfConfusion};
          
          const ambience = ${store.ambience};
          const sun_color = vec3f(1);
          const sun_dir = normalize(vec3f(-1, -1, 1));
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
            dist: f32,
            hit: bool,
            material: Material
          };

          fn scene(ray: Ray) -> Hit {
            var hitObj: Hit;
            hitObj.hit = false;
            hitObj.dist = max_dist;
            hitObj.material.color = vec3(1.);
            hitObj.material.emission = vec3(0.);

            var d2: f32;
            
            d2 = iSphere(ray.pos - sphere_center, ray.dir, vec2(min_dist, hitObj.dist), &hitObj.normal, 1.);
            if (d2 < hitObj.dist) {
              hitObj.hit = true;
              hitObj.dist = d2;
            }

            return hitObj;
          }

          // Sphere:          https://www.shadertoy.com/view/4d2XWV
          fn iSphere( ro: vec3f, rd: vec3f, distBound: vec2f, normal: ptr<function, vec3f>,   
                        sphereRadius: f32 ) -> f32{
              let b = dot(ro, rd);
              let c = dot(ro, ro) - sphereRadius*sphereRadius;
              var h = b*b - c;
              if (h < 0.) {
                  return max_dist;
              } else {
                h = sqrt(h);
                  let d1 = -b-h;
                  let d2 = -b+h;
                  if (d1 >= distBound.x && d1 <= distBound.y) {
                      *normal = normalize(ro + rd*d1);
                      return d1;
                  } else if (d2 >= distBound.x && d2 <= distBound.y) { 
                      *normal = normalize(ro + rd*d2);            
                      return d2;
                  } else {
                      return max_dist;
                  }
              }
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

            
            // for (int x = 0; x < int(samples); ++x) {
              let subpixel = random_2();
              let uv = (2. * (pos + subpixel) - viewportf) / viewportf.x;
              // let rayDirection = paniniRay(uv);
              let rayDirection = pinholeRay(uv);
              
              var ray = thinLensRay(rayDirection, sample_incircle(random_2()));

              let ray_pos = u_view * vec4(ray.pos, 1.);
              // let ray_pos = vec4(ray.pos, 1.);
              ray.pos = ray_pos.xyz;
              ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
              ray.dir = (u_view * vec4(ray.dir, 0.)).xyz;
              // ray.dir = (vec4(ray.dir, 0.)).xyz;
              ray.throughput = vec3(1.);

              // for (int i = 0; i < int(gi_reflection_depth) + 1; ++i) {
                // float seed = x + i;
                let hitObj = scene(ray);
                // let material = hitObj.material;
                let t = hitObj.dist;
                // if (!hitObj.hit) break;
                if (!hitObj.hit) { return; }
                // let cos_angle = -dot(hitObj.normal, ray.dir);
                // let incoming = int(sign(cos_angle));
                // let pos = ray.pos + t * ray.dir;
                // let dir: vec3f;
                // let col: vec3f;
                // color += sun(pos, hitObj.normal); 
                color += sun(ray.pos + t * ray.dir, hitObj.normal); 
                // color = vec3f(1.);

                // ray = Ray(pos, dir, ray.throughput);
              // }
            // }

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

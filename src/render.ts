import {
  computePass,
  computePipeline,
  createStorageBuffer,
  getDevice,
  getTimestampHandler,
  renderBundlePass,
  renderPass,
  renderPipeline,
} from './gpu';
import { setGPUTime, setJSTime, setRenderTime, store } from './store';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
let imageBuffer: GPUBuffer;
let blitRenderBundle: GPURenderBundle;

const resize = () => {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;

  if (imageBuffer) imageBuffer.destroy();
  const size =
    Float32Array.BYTES_PER_ELEMENT * 4 * canvas.width * canvas.height;
  imageBuffer = createStorageBuffer(size, 'Raytraced Image Buffer');
  blitRenderBundle = getBlitToScreenBundle();
};

const tonemapping = /* wgsl */ `
  fn linear_to_srgb(x: vec3f) -> vec3f {
    let rgb = clamp(x, vec3(0.), vec3(1.));
      
    return mix(
      pow(rgb, vec3(1.0 / 2.4)) * 1.055 - 0.055,
      rgb * 12.92,
      vec3f(rgb < vec3(0.0031308))
    );
  }

  fn srgb_to_linear(x: vec3f) -> vec3f {
    let rgb = clamp(x, vec3(0.), vec3(1.));
      
    return mix(
      pow(((rgb + 0.055) / 1.055), vec3(2.4)),
      rgb / 12.92,
      vec3f(rgb < vec3(0.04045))
    );
  }

  // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
  @must_use
  fn aces(x: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate(x * (a * x + b)) / (x * (c * x + d) + e);
  }

  // Filmic Tonemapping Operators http://filmicworlds.com/blog/filmic-tonemapping-operators/
  @must_use
  fn filmic(x: vec3f) -> vec3f {
    let X = max(vec3f(0.0), x - 0.004);
    let result = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
    return pow(result, vec3(2.2));
  }

  // Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines"
  @must_use
  fn lottes(x: vec3f) -> vec3f {
    let a = vec3f(1.6);
    let d = vec3f(0.977);
    let hdrMax = vec3f(8.0);
    let midIn = vec3f(0.18);
    let midOut = vec3f(0.267);

    let b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    let c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, a) / (pow(x, a * d) * b + c);
  }

  @must_use
  fn reinhard(x: vec3f) -> vec3f {
    return x / (1.0 + x);
  }
`;

const getBlitToScreenBundle = () => {
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

      ${x.bindVarBuffer('read-only-storage', 'imageBuffer: array<vec3f>', imageBuffer)}
      // @group(0) @binding(1) var<uniform> commonUniforms: CommonUniforms;

      const viewport = vec2u(${canvas.width}, ${canvas.height});

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

  return renderBundlePass({}, (renderPass) => {
    renderPass.setPipeline(pipeline);
    bindGroups.forEach((bindGroup, i) => renderPass.setBindGroup(i, bindGroup));
    renderPass.draw(6);
  });
};

// const phi = (d: number) => {
//   const k = 1 / (d + 1);
//   let x = (d + 2) * k;
//   for (let i = 0; i < 10; i++) x = Math.pow(1 + x, k);
//   return x;
// };
// const alpha = (d: number) => {
//   const g = phi(d);
//   const a = Array.from({ length: d }, (_, i) => 0);
//   for (let i = 0; i < d; i++) a[i] = Math.pow(1 / g, i + 1);
//   return a;
// };

const randUtils = (outCount: number) => {
  return `
    @must_use
    fn random_${outCount}u() -> vec${outCount}<u32> {
      return vec${outCount}(${Array.from({ length: outCount }, () => `random_1u()`).join(', ')});
    }

    @must_use
    fn random_${outCount}() -> vec${outCount}<f32> {
      return vec${outCount}(${Array.from({ length: outCount }, () => `random_1()`).join(', ')});
    }
  `;
};

const constants = /* wgsl */ `
  const PHI     = 1.61803398874989484820459; // Golden Ratio
  const SRT     = 1.41421356237309504880169; // Square Root of Two
  const PI      = 3.14159265358979323846264;
  const E       = 2.71828182845904523536028;
  const TWO_PI  = 6.28318530717958647692528;
  const INV_PI  = 0.31830988618379067153776;
  
  const BV_MAX_STACK_DEPTH = 16;
  const EPSILON = 0.001;
  // const min_dist = 0x1p-126f;
  const min_dist = EPSILON;
  const max_dist = 0x1.fffffep+127;
`;

const rng = /* wgsl */ `
  ${constants}
  var<private> rng_state: u32 = 0;
  @must_use
  fn random_1u() -> u32 {
    rng_state = rng_state * 277803737u;
    return rng_state;
  }
  
  @must_use
  fn random_1() -> f32 {
    return f32(random_1u()) / f32(0xffffffffu);
  }

  ${randUtils(2)}

  ${randUtils(3)}
  
  ${randUtils(4)}
  
  fn cbrt(x: f32) -> f32 {
    var y = sign(x) * bitcast<f32>( bitcast<u32>( abs(x) ) / 3u + 0x2a514067u );

    for (var i = 0; i < 2; i = i + 1) { 
      y = (2. * y + x / (y * y)) * .333333333; 
    }

    for (var i = 0; i < 1; i = i + 1)
    {
      let y3 = y * y * y;
      y *= (y3 + 2. * x) / (2. * y3 + x);
    }
    
    return y;
  }

  fn sample_circle(t: f32) -> vec2f {
    let phi = t * TWO_PI;
    return vec2(cos(phi), sin(phi));
  }

  fn sample_incircle(t: vec2f) -> vec2f {
    return sample_circle(t.x) * sqrt(t.y);
  }

  fn sample_cosine_weighted_sphere(t: vec2f) -> vec3f {
    let phi = TWO_PI * t.y;
    let sin_theta = sqrt(1. - t.x);
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let z = sqrt(t.x);
    return vec3(x, y, z); 
  }

  fn sample_sphere(t: vec2f) -> vec3f {
    let uv = vec2(t.x * 2. - 1., t.y);
    let sin_theta = sqrt(1 - uv.x * uv.x); 
    let phi = TWO_PI * uv.y; 
    let x = sin_theta * cos(phi); 
    let z = sin_theta * sin(phi); 
    return vec3(x, uv.x, z);
  }
  
  fn sample_insphere(t: vec3f) -> vec3f {
    return sample_sphere(t.xy) * cbrt(t.z); 
  }

  fn sample_insquare(t: vec2f) -> vec2f {
    return (2. * t - 1.); 
  }
`;

export const init = async () => {
  const context = canvas.getContext('webgpu');
  const device = await getDevice(context as GPUCanvasContext);

  resize();
  window.addEventListener('resize', () => {
    resize();
  });

  const { canTimestamp, querySet, submit } = getTimestampHandler((times) => {
    setGPUTime(Number(times[1] - times[0]));
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

  const COMPUTE_WORKGROUP_SIZE_X = 16;
  const COMPUTE_WORKGROUP_SIZE_Y = 16;
  let _computePipeline: GPUComputePipeline, computeBindGroups: GPUBindGroup[];
  const setComputePipeline = () => {
    ({ pipeline: _computePipeline, bindGroups: computeBindGroups } =
      computePipeline({
        shader: (x) => /* wgsl */ `
          ${x.bindVarBuffer('storage', 'imageBuffer: array<vec3f>', imageBuffer)}

          ${rng}

          const viewport = vec2u(${canvas.width}, ${canvas.height});
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

          const sphere_center = vec3f(0, 0, 4);
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

          const cameraFovAngle = PI / 3;
          const paniniDistance = 1.f;
          const lensFocusDistance = 1.f;
          const circleOfConfusionRadius = 0.5f;
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

            return vec3(x, y, z);
          }

          fn thinLensRay(dir: vec3f, uv: vec2f) -> Ray {
            // var ray: Ray;
            // // ray.pos = vec3(uv * circleOfConfusionRadius, 0.f);
            // // ray.dir = normalize(dir * lensFocusDistance - ray.pos);
            // ray.pos = vec3(uv, 0.f);
            // ray.dir = dir;
            // return ray;
            return Ray(vec3(uv, 0.f), dir, vec3(1.));
          }

          const ambience = 0.1;
          const sun_color = vec3f(1);
          const sun_dir = normalize(vec3f(-1, -1, 1));
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
            // return dot(-sun_dir, norm) * sun_color;
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

            
            // for (int x = 0; x < int(samples); ++x) {
              let subpixel = random_2();
              let uv = (2. * (pos + subpixel) - viewportf) / viewportf.x;
              // let rayDirection = normalize(paniniRay(uv));
              let rayDirection = pinholeRay(uv);
              
              var ray = thinLensRay(rayDirection, sample_incircle(random_2()));

              // let ray_pos = u_view * vec4(ray.pos, 1.);
              let ray_pos = vec4(ray.pos, 1.);
              ray.pos = ray_pos.xyz;
              ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
              // ray.dir = (u_view * vec4(ray.dir, 0.)).xyz;
              ray.dir = (vec4(ray.dir, 0.)).xyz;
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
      }));
  };
  window.addEventListener('resize', setComputePipeline);
  setComputePipeline();

  async function frame() {
    const now = performance.now();
    setRenderTime(now);

    const encoder = device.createCommandEncoder();
    rpd.colorAttachments[0].view = context.getCurrentTexture().createView();

    // raytrace
    computePass(encoder, {}, (computePass) => {
      computePass.setPipeline(_computePipeline);
      computeBindGroups.forEach((bindGroup, i) =>
        computePass.setBindGroup(i, bindGroup)
      );
      computePass.dispatchWorkgroups(
        Math.ceil(canvas.width / COMPUTE_WORKGROUP_SIZE_X),
        Math.ceil(canvas.height / COMPUTE_WORKGROUP_SIZE_Y),
        1
      );
    });

    renderPass(encoder, rpd, (renderPass) => {
      renderPass.executeBundles([blitRenderBundle]);

      // debug BVH
      if (store.debugBVH) {
        renderPass.executeBundles([debugBVHRenderBundle]);
      }
    });

    submit(encoder, () => {
      device.queue.submit([encoder.finish()]);
    });

    setJSTime(performance.now() - now);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
};

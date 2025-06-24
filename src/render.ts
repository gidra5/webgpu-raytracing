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

          const viewport = vec2u(${canvas.width}, ${canvas.height});

          @compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE_X}, ${COMPUTE_WORKGROUP_SIZE_Y})
          fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
            if (any(globalInvocationId.xy > viewport)) {
              return;
            }

            let upos = globalInvocationId.xy;
            let idx = upos.x + upos.y * viewport.x;
            let pos = vec2f(upos);

            imageBuffer[idx] = vec3f((pos / vec2f(viewport)), 1.0);
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

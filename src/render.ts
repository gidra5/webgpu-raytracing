import { setGPUTime, setJSTime, setRenderTime, store } from './store';
import { assert } from './utils';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const features: Partial<Record<GPUFeatureName, boolean>> = {};
let presentationFormat: GPUTextureFormat;

const resize = () => {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
};

const getDevice = async () => {
  assert(navigator.gpu, 'WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter, 'No GPU adapter found');

  const requiredFeatures: GPUFeatureName[] = [];
  const canTimestamp = adapter.features.has('timestamp-query');

  if (canTimestamp) {
    features['timestamp-query'] = true;
    requiredFeatures.push('timestamp-query');
  }

  const device = await adapter.requestDevice({ requiredFeatures });
  presentationFormat =
    navigator.gpu.getPreferredCanvasFormat?.() ?? 'bgra8unorm';

  const context = canvas.getContext('webgpu');
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  return { device, context };
};

const mapBuffer = async (
  buffer: GPUBuffer,
  options: { mode: GPUMapModeFlags; offset?: GPUSize64; size?: GPUSize64 },
  handler: (buffer: ArrayBuffer) => void
): Promise<void> => {
  const { mode, offset, size } = options;
  await buffer.mapAsync(mode, offset, size);
  handler(buffer.getMappedRange());
  buffer.unmap();
};

type SubmitHandler = (
  encoder: GPUCommandEncoder,
  submit: () => void
) => Promise<void>;
type TimestampHandler = {
  querySet?: GPUQuerySet;
  canTimestamp: boolean;
  submit: SubmitHandler;
};
const getTimestampHandler = (device: GPUDevice): TimestampHandler => {
  if (!features['timestamp-query']) {
    return {
      canTimestamp: false,
      async submit(_, submit) {
        submit();
      },
    };
  }

  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: 2,
  });

  // buffers with MAP_READ usace can only have COPY_DST as another usage
  // so we need to create intermediate buffer to be able to map for read
  const resolveBuffer = device.createBuffer({
    size: querySet.count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const resultBuffer = device.createBuffer({
    size: resolveBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    querySet,
    canTimestamp: true,
    async submit(encoder: GPUCommandEncoder, submit: () => void) {
      encoder.resolveQuerySet(querySet, 0, querySet.count, resolveBuffer, 0);
      if (resultBuffer.mapState !== 'unmapped') {
        submit();
        return;
      }

      encoder.copyBufferToBuffer(
        resolveBuffer,
        0,
        resultBuffer,
        0,
        resultBuffer.size
      );

      submit();

      await mapBuffer(resultBuffer, { mode: GPUMapMode.READ }, (buffer) => {
        const times = new BigInt64Array(buffer);
        setGPUTime(Number(times[1] - times[0]));
      });
    },
  };
};

const renderPass = (
  encoder: GPUCommandEncoder,
  descriptor: GPURenderPassDescriptor,
  handler: (renderPass: GPURenderPassEncoder) => void
) => {
  const renderPass = encoder.beginRenderPass(descriptor);
  handler(renderPass);
  renderPass.end();
};

const computePass = (
  encoder: GPUCommandEncoder,
  descriptor: GPUComputePassDescriptor,
  handler: (computePass: GPUComputePassEncoder) => void
) => {
  const computePass = encoder.beginComputePass(descriptor);
  handler(computePass);
  computePass.end();
};

const renderBundlePass = (
  device: GPUDevice,
  descriptor: GPURenderBundleEncoderDescriptor,
  handler: (renderPass: GPURenderBundleEncoder) => void
) => {
  const renderBundle = device.createRenderBundleEncoder(descriptor);
  handler(renderBundle);
  return renderBundle.finish();
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

const getBlitToScreenBundle = (device: GPUDevice, imageBuffer: GPUBuffer) => {
  const fragmentShaderModule = device.createShaderModule({
    code: /* wgsl */ `
      ${tonemapping}

      @group(0) @binding(0) var<storage, read> imageBuffer: array<vec3f>;
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
  });
  const vertexShaderModule = device.createShaderModule({
    code: /* wgsl */ `
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
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' },
      },
      // {
      //   binding: 1,
      //   visibility: GPUShaderStage.FRAGMENT,
      //   buffer: { type: 'uniform' },
      // },.
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    // layout: 'auto',
    vertex: { module: vertexShaderModule },
    fragment: {
      module: fragmentShaderModule,
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: imageBuffer } },
      // {
      //   binding: 1,
      //   resource: {
      //     buffer: commonUniformsBuffer,
      //   },
      // },
    ],
  });

  return renderBundlePass(
    device,
    { colorFormats: [presentationFormat] },
    (renderPass) => {
      renderPass.setPipeline(pipeline);
      renderPass.setBindGroup(0, bindGroup);
      renderPass.draw(6);
    }
  );
};

const createStorageBuffer = (
  device: GPUDevice,
  size: number,
  label?: string
) => {
  const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
  buffer.label = label;
  return buffer;
};

export const init = async () => {
  resize();

  const { device, context } = await getDevice();

  const { canTimestamp, querySet, submit } = getTimestampHandler(device);

  const size =
    Float32Array.BYTES_PER_ELEMENT * 4 * canvas.width * canvas.height;
  let imageBuffer = createStorageBuffer(device, size, 'Raytraced Image Buffer');

  let blitRenderBundle = getBlitToScreenBundle(device, imageBuffer);
  window.addEventListener('resize', () => {
    resize();

    imageBuffer.destroy();
    const size =
      Float32Array.BYTES_PER_ELEMENT * 4 * canvas.width * canvas.height;
    imageBuffer = createStorageBuffer(device, size, 'Raytraced Image Buffer');
    blitRenderBundle = getBlitToScreenBundle(device, imageBuffer);
  });

  const debugBVHRenderBundle = renderBundlePass(
    device,
    { colorFormats: [presentationFormat] },
    (renderPass) => {
      //   renderPass.setPipeline(debugBVHPipeline);
      //   renderPass.setBindGroup(0, debugBVHBindGroup);
      //   renderPass.draw(2, Scene.AABBS_COUNT * 12);
    }
  );

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
  async function frame() {
    const now = performance.now();
    setRenderTime(now);

    const encoder = device.createCommandEncoder();
    rpd.colorAttachments[0].view = context.getCurrentTexture().createView();

    const raytraceBindGroup0Layout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        // {
        //   binding: 1,
        //   visibility: GPUShaderStage.COMPUTE,
        //   buffer: {
        //     type: 'storage',
        //   },
        // },
        // {
        //   binding: 2,
        //   visibility: GPUShaderStage.COMPUTE,
        //   buffer: {
        //     type: 'uniform',
        //   },
        // },
        // {
        //   binding: 3,
        //   visibility: GPUShaderStage.COMPUTE,
        //   buffer: {
        //     type: 'uniform',
        //   },
        // },
      ],
    });

    const raytraceBindGroup1Layout = device.createBindGroupLayout({
      entries: [
        // {
        //   binding: 0,
        //   visibility: GPUShaderStage.COMPUTE,
        //   buffer: {
        //     type: 'read-only-storage',
        //   },
        // },
        // {
        //   binding: 1,
        //   visibility: GPUShaderStage.COMPUTE,
        //   buffer: {
        //     type: 'read-only-storage',
        //   },
        // },
        // {
        //   binding: 2,
        //   visibility: GPUShaderStage.COMPUTE,
        //   buffer: {
        //     type: 'read-only-storage',
        //   },
        // },
      ],
    });

    const raytraceShaderModule = device.createShaderModule({
      code: /* wgsl */ `
        @group(0) @binding(0) var<storage, read_write> imageBuffer: array<vec3f>;
        
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
    });
    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [raytraceBindGroup0Layout, raytraceBindGroup1Layout],
      }),
      compute: {
        module: raytraceShaderModule,
        constants: {
          // WORKGROUP_SIZE_X: COMPUTE_WORKGROUP_SIZE_X,
          // WORKGROUP_SIZE_Y: COMPUTE_WORKGROUP_SIZE_Y,
          // OBJECTS_COUNT_IN_SCENE: Scene.MODELS_COUNT,
          // MAX_BVs_COUNT_PER_MESH: Scene.MAX_NUM_BVs_PER_MESH,
          // MAX_FACES_COUNT_PER_MESH: Scene.MAX_NUM_FACES_PER_MESH,
        },
      },
    });
    const computeBindGroup0 = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: imageBuffer } },
        // {
        //   binding: 1,
        //   resource: {
        //     buffer: rngStateBuffer,
        //   },
        // },
        // {
        //   binding: 2,
        //   resource: {
        //     buffer: commonUniformsBuffer,
        //   },
        // },
        // {
        //   binding: 3,
        //   resource: {
        //     buffer: cameraUniformBuffer,
        //   },
        // },
      ],
    });
    const computeBindGroup1 = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(1),
      entries: [
        // {
        //   binding: 0,
        //   resource: {
        //     buffer: scene.facesBuffer,
        //   },
        // },
        // {
        //   binding: 1,
        //   resource: {
        //     buffer: scene.aabbsBuffer,
        //   },
        // },
        // {
        //   binding: 2,
        //   resource: {
        //     buffer: scene.materialsBuffer,
        //   },
        // },
      ],
    });

    // raytrace
    computePass(encoder, {}, (computePass) => {
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, computeBindGroup0);
      computePass.setBindGroup(1, computeBindGroup1);
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

import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
} from 'vitest';
import {
  depthBinFragmentWGSL,
  depthBinVertexWGSL,
} from '../src/depthPeeling/shaders';
import {
  DEPTH_PEELING_EPSILON,
  runDepthPeelingPipelineCPUState,
} from '../src/depthPeeling/cpuPipeline';

const gpuSupported = typeof navigator !== 'undefined' && !!navigator.gpu;

const BYTES_PER_ROW = 256;

const approxEqualArrays = (
  actual: number[],
  expected: number[],
  epsilon = DEPTH_PEELING_EPSILON * 8
) => {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(epsilon);
  }
};

const mulberry32 = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

const buildRandomFragments = (count: number, seed: number) => {
  const rng = mulberry32(seed);
  const fragments = Array.from({ length: count }, () => {
    const base = rng();
    const jitter = (rng() - 0.5) * 0.05;
    return Math.min(
      0.999999,
      Math.max(DEPTH_PEELING_EPSILON * 0.5, base + jitter)
    );
  });
  return fragments;
};

type DepthPeelingTextures = {
  prevDepth: GPUTexture;
  prevCount: GPUTexture;
  currDepth: GPUTexture;
  currCount: GPUTexture;
};

const createDepthPeelingTextures = (
  device: GPUDevice,
  layers: number
): DepthPeelingTextures => {
  const size: GPUExtent3D = {
    width: 1,
    height: 1,
    depthOrArrayLayers: layers,
  };
  const usage =
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.COPY_DST;

  const descriptor: GPUTextureDescriptor = {
    size,
    format: 'r32float',
    usage,
  };

  return {
    prevDepth: device.createTexture(descriptor),
    prevCount: device.createTexture(descriptor),
    currDepth: device.createTexture(descriptor),
    currCount: device.createTexture(descriptor),
  };
};

const resetDepthPeelingTextures = async (
  device: GPUDevice,
  textures: DepthPeelingTextures,
  layers: number
) => {
  const encoder = device.createCommandEncoder();

  for (let layer = 0; layer < layers; layer += 1) {
    const upper = (layer + 1) / layers;
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textures.prevDepth.createView({
            dimension: '2d',
            baseArrayLayer: layer,
            arrayLayerCount: 1,
          }),
          clearValue: [upper, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: textures.prevCount.createView({
            dimension: '2d',
            baseArrayLayer: layer,
            arrayLayerCount: 1,
          }),
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: textures.currDepth.createView({
            dimension: '2d',
            baseArrayLayer: layer,
            arrayLayerCount: 1,
          }),
          clearValue: [1, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: textures.currCount.createView({
            dimension: '2d',
            baseArrayLayer: layer,
            arrayLayerCount: 1,
          }),
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.end();
  }

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
};

const createDepthBinPipeline = (device: GPUDevice, layers: number) => {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          viewDimension: '2d-array',
          sampleType: 'unfilterable-float',
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          viewDimension: '2d-array',
          sampleType: 'unfilterable-float',
        },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const vertexModule = device.createShaderModule({ code: depthBinVertexWGSL });
  const fragmentModule = device.createShaderModule({
    code: depthBinFragmentWGSL(
      layers,
      `
      @group(0) @binding(0) var prevDepthTexture: texture_2d_array<f32>;
      @group(0) @binding(1) var prevCountTexture: texture_2d_array<f32>;
    `,
      DEPTH_PEELING_EPSILON
    ),
  });

  const targets: GPUColorTargetState[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    targets.push({
      format: 'r32float',
      blend: {
        color: { operation: 'min', srcFactor: 'one', dstFactor: 'one' },
        alpha: { operation: 'min', srcFactor: 'one', dstFactor: 'one' },
      },
    });
    targets.push({
      format: 'r32float',
      blend: {
        color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
      },
    });
  }

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: vertexModule,
      entryPoint: 'main',
      buffers: [
        {
          arrayStride: 4 * Float32Array.BYTES_PER_ELEMENT,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }],
        },
      ],
    },
    fragment: {
      module: fragmentModule,
      entryPoint: 'main',
      targets,
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  return { pipeline, bindGroupLayout };
};

const createFragmentVertexBuffer = (
  device: GPUDevice,
  fragments: number[]
): GPUBuffer => {
  const verticesPerTriangle = 3;
  const componentsPerVertex = 4;
  const data = new Float32Array(
    fragments.length * verticesPerTriangle * componentsPerVertex
  );

  fragments.forEach((originalDepth, fragmentIndex) => {
    const depth = Math.min(0.999999, Math.max(0, originalDepth));
    const baseOffset =
      fragmentIndex * verticesPerTriangle * componentsPerVertex;
    const positions = [-1, -1, depth, 1, 3, -1, depth, 1, -1, 3, depth, 1];
    data.set(positions, baseOffset);
  });

  const byteLength = Math.max(4, data.byteLength);
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength > 0) {
    device.queue.writeBuffer(
      buffer,
      0,
      data.buffer,
      data.byteOffset,
      data.byteLength
    );
  }
  return buffer;
};
const readTextureLayers = async (
  device: GPUDevice,
  texture: GPUTexture,
  layers: number
): Promise<number[]> => {
  const buffer = device.createBuffer({
    size: BYTES_PER_ROW * layers,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow: BYTES_PER_ROW, rowsPerImage: 1 },
    { width: 1, height: 1, depthOrArrayLayers: layers }
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Float32Array(buffer.getMappedRange());
  const stride = BYTES_PER_ROW / Float32Array.BYTES_PER_ELEMENT;
  const result: number[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    result.push(mapped[layer * stride]);
  }
  buffer.unmap();
  buffer.destroy();
  return result;
};

const runDepthPeelingGPU = async (
  device: GPUDevice,
  fragments: number[],
  layers: number
): Promise<{ depths: number[]; counts: number[] }> => {
  if (layers <= 0) {
    return { depths: [], counts: [] };
  }

  const textures = createDepthPeelingTextures(device, layers);
  await resetDepthPeelingTextures(device, textures, layers);

  const { pipeline, bindGroupLayout } = createDepthBinPipeline(device, layers);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: textures.prevDepth.createView({ dimension: '2d-array' }),
      },
      {
        binding: 1,
        resource: textures.prevCount.createView({ dimension: '2d-array' }),
      },
    ],
  });

  const vertexBuffer = createFragmentVertexBuffer(device, fragments);
  const encoder = device.createCommandEncoder();

  for (let iteration = 0; iteration < layers; iteration += 1) {
    const colorAttachments: GPURenderPassColorAttachment[] = [];
    for (let layer = 0; layer < layers; layer += 1) {
      colorAttachments.push({
        view: textures.currDepth.createView({
          dimension: '2d',
          baseArrayLayer: layer,
          arrayLayerCount: 1,
        }),
        clearValue: [1, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      });
      colorAttachments.push({
        view: textures.currCount.createView({
          dimension: '2d',
          baseArrayLayer: layer,
          arrayLayerCount: 1,
        }),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      });
    }

    const pass = encoder.beginRenderPass({ colorAttachments });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(fragments.length * 3, 1, 0, 0);
    pass.end();

    encoder.copyTextureToTexture(
      { texture: textures.currDepth },
      { texture: textures.prevDepth },
      { width: 1, height: 1, depthOrArrayLayers: layers }
    );
    encoder.copyTextureToTexture(
      { texture: textures.currCount },
      { texture: textures.prevCount },
      { width: 1, height: 1, depthOrArrayLayers: layers }
    );
  }

  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const depths = await readTextureLayers(device, textures.prevDepth, layers);
  const counts = await readTextureLayers(device, textures.prevCount, layers);

  vertexBuffer.destroy();
  textures.prevDepth.destroy();
  textures.prevCount.destroy();
  textures.currDepth.destroy();
  textures.currCount.destroy();

  return { depths, counts };
};

const describeIf = gpuSupported ? describe : describe.skip;

describeIf('depth peeling GPU pipeline', () => {
  let device: GPUDevice;

  beforeAll(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    expect(adapter).toBeTruthy();
    expect(adapter!.features.has('float32-blendable')).toBe(true);
    device = await adapter!.requestDevice({
      requiredFeatures: ['float32-blendable'],
    });
  });

  beforeEach(() => {
    device.pushErrorScope('validation');
  });

  afterEach(async () => {
    const error = await device.popErrorScope();
    expect(error?.message).toBeUndefined();
  });

  afterAll(() => {
    device?.destroy();
  });

  it('matches CPU textures for a simple ordered set', async () => {
    const fragments = [0.15, 0.62, 0.33, 0.44];
    const layers = 4;

    const expected = runDepthPeelingPipelineCPUState(fragments, layers);
    const actual = await runDepthPeelingGPU(device, fragments, layers);

    approxEqualArrays(actual.depths, expected.depths);
    approxEqualArrays(actual.counts, expected.counts, 1e-4);
  });

  it('retains previous minima at depth bin boundaries', async () => {
    const fragments = [0.1, 0.25, 0.5, 0.75];
    const layers = 4;

    const expected = runDepthPeelingPipelineCPUState(fragments, layers);
    const actual = await runDepthPeelingGPU(device, fragments, layers);

    approxEqualArrays(actual.depths, expected.depths);
    approxEqualArrays(actual.counts, expected.counts, 1e-4);
  });

  it('supports a single-layer texture array', async () => {
    const fragments = [0.15, 0.62, 0.33];
    const layers = 1;

    const expected = runDepthPeelingPipelineCPUState(fragments, layers);
    const actual = await runDepthPeelingGPU(device, fragments, layers);

    approxEqualArrays(actual.depths, expected.depths);
    approxEqualArrays(actual.counts, expected.counts, 1e-4);
  });

  it('leaves empty depth bins at their clear values', async () => {
    const layers = 3;
    const expected = runDepthPeelingPipelineCPUState([], layers);
    const actual = await runDepthPeelingGPU(device, [], layers);

    approxEqualArrays(actual.depths, expected.depths);
    approxEqualArrays(actual.counts, expected.counts, 1e-4);
  });

  it('matches CPU textures for random fragment distributions', async () => {
    const seeds = Array.from({ length: 5 }, (_, index) => index + 1);

    for (const seed of seeds) {
      const layers = 2 + (seed % 3);
      const fragments = buildRandomFragments(layers + 2, seed);

      const expected = runDepthPeelingPipelineCPUState(fragments, layers);
      const actual = await runDepthPeelingGPU(device, fragments, layers);

      approxEqualArrays(actual.depths, expected.depths, 5e-4);
      approxEqualArrays(actual.counts, expected.counts, 1e-3);
    }
  });
});

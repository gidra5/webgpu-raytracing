import { beforeAll, describe, expect, it } from 'vitest';
import {
  DEPTH_PEELING_EPSILON,
  runDepthPeelingPipelineCPUState,
} from '../src/depthPeeling/cpuPipeline';

const gpuSupported = typeof navigator !== 'undefined' && !!navigator.gpu;

const depthBinVertexWGSL = `
  @vertex
  fn main(@location(0) position: vec4f) -> @builtin(position) vec4f {
    return position;
  }
`;

const depthBinFragmentWGSL = (layers: number) => {
  const outputLocations = Array.from({ length: layers }, (_, index) => {
    return [
      `      @location(${2 * index}) depth${index}: f32,`,
      `      @location(${2 * index + 1}) count${index}: f32,`,
    ].join('\n');
  }).join('\n');

  const assignments = Array.from({ length: layers }, (_, index) => {
    return [
      `      let inRange${index} = depth >= rangeMin[${index}u] && depth <= rangeMax[${index}u];`,
      `      output.depth${index} = select(1.0, depth, inRange${index});`,
      `      output.count${index} = select(0.0, 1.0, inRange${index});`,
    ].join('\n');
  }).join('\n');

  return `
    @group(0) @binding(0) var prevDepthTexture: texture_2d_array<f32>;
    @group(0) @binding(1) var prevCountTexture: texture_2d_array<f32>;

    const layerCount = ${layers}u;
    const epsilon = ${DEPTH_PEELING_EPSILON};

    struct FragmentInput {
      @builtin(position) fragmentPosition: vec4f,
    }

    struct FragmentOutput {
${outputLocations}
    }

    @fragment
    fn main(input: FragmentInput) -> FragmentOutput {
      let pixel = vec2u(input.fragmentPosition.xy);
      let depth = clamp(input.fragmentPosition.z, 0.0, 1.0);

      var prevDepths: array<f32, layerCount>;
      var prevCounts: array<u32, layerCount>;
      var rangeMin: array<f32, layerCount>;
      var rangeMax: array<f32, layerCount>;

      var totalCount: u32 = 0u;
      for (var i: u32 = 0u; i < layerCount; i = i + 1u) {
        let d = textureLoad(prevDepthTexture, pixel, i).r;
        let c = textureLoad(prevCountTexture, pixel, i).r;
        prevDepths[i] = clamp(d, 0.0, 1.0);
        let count = u32(round(max(c, 0.0)));
        prevCounts[i] = count;
        totalCount += count;
      }

      if (totalCount == 0u) {
        var lower = 0.0;
        for (var i: u32 = 0u; i < layerCount; i = i + 1u) {
          let upper = max(lower + epsilon, prevDepths[i]);
          rangeMin[i] = lower;
          rangeMax[i] = upper;
          lower = upper;
        }
      } else {
        var newIndex: u32 = 0u;
        var lastUpper = 0.0;
        var i: u32 = 0u;
        loop {
          if (i >= layerCount || newIndex >= layerCount) {
            break;
          }

          let count = prevCounts[i];
          if (count == 0u) {
            lastUpper = max(lastUpper, prevDepths[i]);
            i = i + 1u;
            continue;
          }

          var lower = 0.0;
          if (i > 0u) {
            lower = prevDepths[i - 1u];
          }
          lower = max(lower, lastUpper);
          var upper = max(lower + epsilon, prevDepths[i]);

          rangeMin[newIndex] = clamp(lower, 0.0, 1.0);
          rangeMax[newIndex] = clamp(upper, 0.0, 1.0);
          lastUpper = upper;
          newIndex = newIndex + 1u;

          if (count > 1u && newIndex < layerCount) {
            var nextUpper = 1.0;
            if (i + 1u < layerCount) {
              nextUpper = max(prevDepths[i + 1u], upper) + epsilon;
            }
            let available = layerCount - newIndex;
            let binsToAllocate = min(count - 1u, available);
            if (binsToAllocate > 0u) {
              let step = (nextUpper - upper) / f32(binsToAllocate);
              var k: u32 = 0u;
              loop {
                if (k >= binsToAllocate || newIndex >= layerCount) {
                  break;
                }
                let start = upper + step * f32(k);
                let finish = upper + step * f32(k + 1u);
                rangeMin[newIndex] = clamp(start, 0.0, 1.0);
                rangeMax[newIndex] = clamp(finish, 0.0, 1.0);
                lastUpper = finish;
                newIndex = newIndex + 1u;
                k = k + 1u;
              }
            }
          }

          i = i + 1u;
        }

        while (newIndex < layerCount) {
          var lower = 0.0;
          if (newIndex > 0u) {
            lower = rangeMax[newIndex - 1u];
          }
          rangeMin[newIndex] = clamp(lower, 0.0, 1.0);
          rangeMax[newIndex] = 1.0;
          newIndex = newIndex + 1u;
        }
      }

      var output: FragmentOutput;
${assignments}
      return output;
    }
  `;
};

const BYTES_PER_ROW = 256;

const approxEqualArrays = (actual: number[], expected: number[], epsilon = DEPTH_PEELING_EPSILON * 8) => {
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
    return Math.min(0.999999, Math.max(DEPTH_PEELING_EPSILON * 0.5, base + jitter));
  });
  return fragments;
};

type DepthPeelingTextures = {
  prevDepth: GPUTexture;
  prevCount: GPUTexture;
  currDepth: GPUTexture;
  currCount: GPUTexture;
};

const createDepthPeelingTextures = (device: GPUDevice, layers: number): DepthPeelingTextures => {
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
          view: textures.prevDepth.createView({ baseArrayLayer: layer, arrayLayerCount: 1 }),
          clearValue: [upper, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: textures.prevCount.createView({ baseArrayLayer: layer, arrayLayerCount: 1 }),
          clearValue: [0, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: textures.currDepth.createView({ baseArrayLayer: layer, arrayLayerCount: 1 }),
          clearValue: [1, 0, 0, 0],
          loadOp: 'clear',
          storeOp: 'store',
        },
        {
          view: textures.currCount.createView({ baseArrayLayer: layer, arrayLayerCount: 1 }),
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
        texture: { viewDimension: '2d-array', sampleType: 'float' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { viewDimension: '2d-array', sampleType: 'float' },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const vertexModule = device.createShaderModule({ code: depthBinVertexWGSL });
  const fragmentModule = device.createShaderModule({ code: depthBinFragmentWGSL(layers) });

  const targets: GPUColorTargetState[] = [];
  for (let layer = 0; layer < layers; layer += 1) {
    targets.push({
      format: 'r32float',
      blend: {
        color: { operation: 'min' },
        alpha: { operation: 'min' },
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
  const data = new Float32Array(fragments.length * verticesPerTriangle * componentsPerVertex);

  fragments.forEach((originalDepth, fragmentIndex) => {
    const depth = Math.min(0.999999, Math.max(0, originalDepth));
    const baseOffset = fragmentIndex * verticesPerTriangle * componentsPerVertex;
    const positions = [
      -1, -1, depth, 1,
      3, -1, depth, 1,
      -1, 3, depth, 1,
    ];
    data.set(positions, baseOffset);
  });

  const byteLength = Math.max(4, data.byteLength);
  const buffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  if (data.byteLength > 0) {
    device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
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
        view: textures.currDepth.createView({ baseArrayLayer: layer, arrayLayerCount: 1 }),
        clearValue: [1, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'store',
      });
      colorAttachments.push({
        view: textures.currCount.createView({ baseArrayLayer: layer, arrayLayerCount: 1 }),
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
    device = await adapter!.requestDevice();
  });

  it('matches CPU textures for a simple ordered set', async () => {
    const fragments = [0.15, 0.62, 0.33, 0.44];
    const layers = 4;

    const expected = runDepthPeelingPipelineCPUState(fragments, layers);
    const actual = await runDepthPeelingGPU(device, fragments, layers);

    approxEqualArrays(actual.depths, expected.depths);
    approxEqualArrays(actual.counts, expected.counts, 1e-4);
  });

  it('matches CPU textures for random fragment distributions', async () => {
    const seeds = Array.from({ length: 5 }, (_, index) => index + 1);

    for (const seed of seeds) {
      const layers = 2 + (seed % 4);
      const fragments = buildRandomFragments(layers + 2, seed);

      const expected = runDepthPeelingPipelineCPUState(fragments, layers);
      const actual = await runDepthPeelingGPU(device, fragments, layers);

      approxEqualArrays(actual.depths, expected.depths, 5e-4);
      approxEqualArrays(actual.counts, expected.counts, 1e-3);
    }
  });
});


import { Accessor, createEffect, createMemo, createSignal } from 'solid-js';
import { assert, omit } from './utils';
import { vec2, vec3 } from 'gl-matrix';
import { CreateTextureOptions } from 'webgpu-utils';

const features: Partial<Record<GPUFeatureName, boolean>> = {};
let presentationFormat: GPUTextureFormat;
let device: GPUDevice;

export const getDevice = async (context: GPUCanvasContext) => {
  assert(navigator.gpu, 'WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter, 'No GPU adapter found');

  const requiredFeatures: GPUFeatureName[] = [];
  const canTimestamp = adapter.features.has('timestamp-query');

  if (canTimestamp) {
    features['timestamp-query'] = true;
    requiredFeatures.push('timestamp-query');
  }

  if (adapter.features.has('subgroups')) {
    features['subgroups'] = true;
    requiredFeatures.push('subgroups');
  }

  // if (adapter.features.has('primitive-index')) {
  //   features['primitive-index'] = true;
  //   requiredFeatures.push('primitive-index');
  // }

  console.log([...adapter.features]);

  device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: 2147483644,
      maxBufferSize: 2147483644,
      maxStorageBuffersPerShaderStage: 10,
    },
  });
  presentationFormat =
    navigator.gpu.getPreferredCanvasFormat?.() ?? 'bgra8unorm';

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  return device;
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
type ResultHandler = (times: BigInt64Array<ArrayBuffer>) => void;
export const getTimestampHandler = (
  resultHandler: ResultHandler
): TimestampHandler => {
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

  // buffers with MAP_READ usage can only have COPY_DST as another usage
  // so we need to create intermediate buffer to be able to map for read
  const resolveBuffer = device.createBuffer({
    size: querySet.count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    label: 'Query Resolve Buffer',
  });
  const resultBuffer = device.createBuffer({
    size: resolveBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'Query Result Read-back Buffer',
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

      encoder.copyBufferToBuffer(resolveBuffer, resultBuffer);

      submit();

      await mapBuffer(resultBuffer, { mode: GPUMapMode.READ }, (buffer) => {
        const times = new BigInt64Array(buffer);
        resultHandler(times);
      });
    },
  };
};

export const mapBuffer = async (
  buffer: GPUBuffer,
  options: { mode: GPUMapModeFlags; offset?: GPUSize64; size?: GPUSize64 },
  handler: (buffer: ArrayBuffer) => void
): Promise<void> => {
  const { mode, offset, size } = options;
  await buffer.mapAsync(mode, offset, size);
  handler(buffer.getMappedRange());
  buffer.unmap();
};

export const writeBuffer = (
  buffer: GPUBuffer,
  offset: GPUSize64,
  data: GPUAllowSharedBufferSource,
  dataOffset?: GPUSize64,
  size?: GPUSize64
) => {
  device.queue.writeBuffer(buffer, offset, data, dataOffset, size);
};

export const writeUint32Buffer = (buffer: GPUBuffer, data: number) => {
  device.queue.writeBuffer(buffer, 0, new Uint32Array([data]));
};

export const writeFloat32Buffer = (buffer: GPUBuffer, data: number) => {
  device.queue.writeBuffer(buffer, 0, new Float32Array([data]));
};
export const writeVec3fBuffer = (buffer: GPUBuffer, data: vec3) => {
  device.queue.writeBuffer(buffer, 0, new Float32Array(data));
};
export const writeVec2fBuffer = (buffer: GPUBuffer, data: vec2) => {
  device.queue.writeBuffer(buffer, 0, new Float32Array(data));
};

export const reactiveUniformBuffer = <T extends number | Iterable<number>>(
  size: number,
  value: Accessor<T>,
  usage: GPUBufferUsageFlags = 0
) => {
  const bufferHi = createUniformBuffer(
    size * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    usage
  );
  const bufferLo = createUniformBuffer(
    size * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    usage
  );

  createEffect(() => {
    const _value = value();
    if (typeof _value === 'number') {
      let hi = new Float32Array([_value]);
      let lo = new Float32Array([_value - hi[0]]);
      writeBuffer(bufferLo, 0, lo);
      writeBuffer(bufferHi, 0, hi);
    } else {
      let hi = new Float32Array(_value);
      let lo = new Float32Array(
        [...(_value as Iterable<number>)].map((v, i) => v - hi[i])
      );

      writeBuffer(bufferLo, 0, lo);
      writeBuffer(bufferHi, 0, hi);
    }
  });

  return [bufferHi, bufferLo] as const;
};

export const renderPass = (
  encoder: GPUCommandEncoder,
  descriptor: GPURenderPassDescriptor,
  handler: (renderPass: GPURenderPassEncoder) => void
) => {
  const renderPass = encoder.beginRenderPass(descriptor);
  handler(renderPass);
  renderPass.end();
};

export const computePass = (
  encoder: GPUCommandEncoder,
  descriptor: GPUComputePassDescriptor,
  handler: (computePass: GPUComputePassEncoder) => void
) => {
  const computePass = encoder.beginComputePass(descriptor);
  handler(computePass);
  computePass.end();
};

export const renderBundlePass = (
  descriptor: Partial<GPURenderBundleEncoderDescriptor>,
  handler: (renderPass: GPURenderBundleEncoder) => void
) => {
  const renderBundle = device.createRenderBundleEncoder({
    colorFormats: [presentationFormat],
    ...descriptor,
  });
  handler(renderBundle);
  return renderBundle.finish();
};

const bufferBuilder = (d: GPUBufferDescriptor = { size: 0, usage: 0 }) => {
  return {
    size(size: number) {
      return bufferBuilder({ ...d, size });
    },
    usage(usage: GPUBufferUsageFlags) {
      return bufferBuilder({ ...d, usage });
    },
    mapped() {
      return bufferBuilder({ ...d, mappedAtCreation: true });
    },
    label(label: string) {
      return bufferBuilder({ ...d, label });
    },
    build() {
      return device.createBuffer(d);
    },
  };
};

export const createVertexBuffer = (
  size: number,
  label?: string,
  usage: GPUBufferUsageFlags = 0,
  mapped = false
) =>
  device.createBuffer({
    size,
    usage: GPUBufferUsage.VERTEX | usage,
    label,
    mappedAtCreation: mapped,
  });

export const createStorageBuffer = (
  size: number,
  label?: string,
  usage: GPUBufferUsageFlags = 0,
  mapped = false
) =>
  device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | usage,
    label,
    mappedAtCreation: mapped,
  });

export const createUniformBuffer = (
  size: number,
  label?: string,
  usage: GPUBufferUsageFlags = 0
) =>
  device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | usage,
    label,
  });

type CreateTextureSource = {
  data?: ArrayBuffer;
  width: number;
  height: number;
  depthOrArrayLayers?: number;
};
export const createTexture = (
  source: CreateTextureSource,
  options?: CreateTextureOptions
) => {
  const depth = source.depthOrArrayLayers ?? 1;
  const texture = device.createTexture({
    size: [source.width, source.height, depth],
    format: 'rgba32float',
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      (options.usage ?? 0),
    ...omit(options, ['usage']),
  });

  if (source.data) {
    // Upload the pixel data to the texture
    device.queue.writeTexture(
      { texture },
      source.data,
      {
        bytesPerRow: source.width * 4 * Float32Array.BYTES_PER_ELEMENT, // 4 channels * 4 bytes/float
        rowsPerImage: source.height,
      },
      { width: source.width, height: source.height, depthOrArrayLayers: depth }
    );
  }

  return texture;
};

export const createBindGroup = (d: GPUBindGroupDescriptor) =>
  device.createBindGroup(d);

type RenderPipelineDescriptor = {
  vertexShader: (builder: PipelineBuilder) => string;
  fragmentShader?: (builder: PipelineBuilder) => string;
  fragmentPresentationFormatTarget?: Omit<GPUColorTargetState, 'format'>;
  vertex?: Omit<GPUVertexState, 'module'>;
  fragment?: Omit<GPUFragmentState, 'module'>;
} & Omit<GPURenderPipelineDescriptor, 'fragment' | 'vertex' | 'layout'>;
type RenderPipelineBuilderResult = {
  pipeline: GPURenderPipeline;
  bindGroups: GPUBindGroup[];
};
export type PipelineBuilder = {
  bindGroup(): string;
  bindVarBuffer(
    access: GPUBufferBindingType,
    type: string,
    buffer: GPUBuffer
  ): string;
  bindTexture(
    name: string,
    type: GPUTextureSampleType,
    texture: GPUTexture,
    isArray?: boolean
  ): string;
  bindSampler(
    name: string,
    type: GPUSamplerBindingType,
    sampler: GPUSampler
  ): string;
  bindStorageTexture(
    name: string,
    access: GPUStorageTextureAccess,
    texture: GPUTexture,
    isArray?: boolean
  ): string;
};
type Binding =
  | {
      kind: 'buffer';
      visibility: GPUShaderStage;
      type: GPUBufferBindingType;
      buffer: GPUBuffer;
    }
  | {
      kind: 'texture';
      visibility: GPUShaderStage;
      type: GPUTextureSampleType;
      texture: GPUTexture;
      multisampled?: boolean;
      dimension?: GPUTextureViewDimension;
    }
  | {
      kind: 'storageTexture';
      visibility: GPUShaderStage;
      texture: GPUTexture;
      access: GPUStorageTextureAccess;
      multisampled?: boolean;
      dimension?: GPUTextureViewDimension;
    }
  | {
      kind: 'sampler';
      visibility: GPUShaderStage;
      type: GPUSamplerBindingType;
      sampler: GPUSampler;
    };

const createBindingBuilder = () => {
  const bindings: Binding[][] = [[]];
  const createPipelineBuilder = (
    visibility: GPUShaderStage
  ): PipelineBuilder => ({
    bindGroup(): string {
      bindings.push([]);
      return '';
    },
    bindVarBuffer(
      access: GPUBufferBindingType,
      type: string,
      buffer: GPUBuffer
    ) {
      const group = bindings.length - 1;
      const binding = bindings[group].length;
      bindings[group].push({
        kind: 'buffer',
        buffer,
        visibility,
        type: access,
      });
      const qualifier =
        access === 'read-only-storage'
          ? 'storage, read'
          : access === 'storage'
            ? 'storage, read_write'
            : 'uniform';
      return `@group(${group}) @binding(${binding}) var<${qualifier}> ${type};`;
    },
    bindTexture(
      name: string,
      type: GPUTextureSampleType,
      texture: GPUTexture,
      isArray = false
    ) {
      const group = bindings.length - 1;
      const binding = bindings[group].length;
      bindings[group].push({ kind: 'texture', texture, visibility, type });
      const valueType =
        type === 'float' || type === 'unfilterable-float' ? 'f32' : type;
      const containerType =
        texture.dimension === '1d'
          ? 'texture_1d'
          : texture.dimension === '2d'
            ? isArray
              ? 'texture_2d_array'
              : 'texture_2d'
            : 'texture_3d';
      return `@group(${group}) @binding(${binding}) var ${name}: ${containerType}<${valueType}>;`;
    },
    bindStorageTexture(
      name: string,
      access: GPUStorageTextureAccess,
      texture: GPUTexture,
      isArray = false
    ) {
      const group = bindings.length - 1;
      const binding = bindings[group].length;
      bindings[group].push({
        kind: 'storageTexture',
        texture,
        visibility,
        access,
      });

      const containerType =
        texture.dimension === '1d'
          ? 'texture_storage_1d'
          : texture.dimension === '2d'
            ? isArray
              ? 'texture_storage_2d_array'
              : 'texture_storage_2d'
            : 'texture_storage_3d';

      const accessType =
        access === 'read-only'
          ? 'read'
          : access === 'write-only'
            ? 'write'
            : 'read_write';
      return `@group(${group}) @binding(${binding}) var ${name}: ${containerType}<${texture.format}, ${accessType}>;`;
    },

    bindSampler(
      name: string,
      type: GPUSamplerBindingType,
      sampler: GPUSampler
    ) {
      const group = bindings.length - 1;
      const binding = bindings[group].length;
      bindings[group].push({ kind: 'sampler', sampler, visibility, type });

      return `@group(${group}) @binding(${binding}) var ${name}: sampler;`;
    },
  });

  return {
    createPipelineBuilder,
    bindingGroupLayouts: () =>
      bindings.map((group) =>
        device.createBindGroupLayout({
          entries: group.map((binding, i): GPUBindGroupLayoutEntry => {
            if (binding.kind === 'texture') {
              const { visibility, type, dimension, multisampled } = binding;
              return {
                binding: i,
                visibility: visibility as unknown as number,
                texture: {
                  sampleType: type,
                  viewDimension: dimension,
                  multisampled,
                },
              };
            }
            if (binding.kind === 'storageTexture') {
              const { visibility, dimension, access, texture } = binding;
              return {
                binding: i,
                visibility: visibility as unknown as number,
                storageTexture: {
                  viewDimension: '2d-array',
                  access,
                  format: texture.format,
                },
              };
            }
            if (binding.kind === 'sampler') {
              const { visibility, type } = binding;
              return {
                binding: i,
                visibility: visibility as unknown as number,
                sampler: { type },
              };
            }

            const { visibility, type } = binding;
            return {
              binding: i,
              visibility: visibility as unknown as number,
              buffer: { type },
            };
          }),
        })
      ),
    bindingGroups: (pipeline: GPURenderPipeline | GPUComputePipeline) =>
      bindings.map((group, i) =>
        device.createBindGroup({
          layout: pipeline.getBindGroupLayout(i),
          entries: group.map((binding, i): GPUBindGroupEntry => {
            if (binding.kind === 'texture') {
              const { texture } = binding;
              return { binding: i, resource: texture.createView() };
            }
            if (binding.kind === 'storageTexture') {
              const { texture } = binding;
              return {
                binding: i,
                resource: texture.createView({ dimension: '2d-array' }),
              };
            }

            if (binding.kind === 'sampler') {
              const { sampler } = binding;
              return { binding: i, resource: sampler };
            }

            const { buffer } = binding;
            return { binding: i, resource: { buffer } };
          }),
        })
      ),
  };
};

export const renderPipeline = (
  x: RenderPipelineDescriptor,
  prev?: GPURenderPipeline
): RenderPipelineBuilderResult => {
  const { createPipelineBuilder, bindingGroupLayouts, bindingGroups } =
    createBindingBuilder();

  // TODO: TYPESCRIPT BULLSHIT
  const fragmentShaderModule =
    x.fragmentShader &&
    device.createShaderModule({
      code: x.fragmentShader(
        createPipelineBuilder(
          GPUShaderStage.FRAGMENT as unknown as GPUShaderStage
        )
      ),
    });
  const vertexShaderModule = device.createShaderModule({
    code: x.vertexShader(
      createPipelineBuilder(GPUShaderStage.VERTEX as unknown as GPUShaderStage)
    ),
  });
  if (!prev) {
    const d: any = {
      layout: device.createPipelineLayout({
        bindGroupLayouts: bindingGroupLayouts(),
      }),
      vertex: {
        module: vertexShaderModule,
        ...x.vertex,
      },
      ...omit(x, ['vertex']),
    };

    if (fragmentShaderModule) {
      d.fragment = {
        module: fragmentShaderModule,
        targets: x.fragment?.targets ?? [
          { format: presentationFormat, ...x.fragmentPresentationFormatTarget },
        ],
      };
    }

    const pipeline = device.createRenderPipeline(
      d as GPURenderPipelineDescriptor
    );
    return { pipeline, bindGroups: bindingGroups(pipeline) };
  } else {
    return { pipeline: prev, bindGroups: bindingGroups(prev) };
  }
};

export type ComputePipelineDescriptor = {
  shader: (builder: PipelineBuilder) => string;
} & Omit<GPUProgrammableStage, 'module'>;
type ComputePipelineBuilderResult = {
  pipeline: GPUComputePipeline;
  bindGroups: GPUBindGroup[];
};
export const computePipeline = (
  x: ComputePipelineDescriptor,
  prev?: GPUComputePipeline
): ComputePipelineBuilderResult => {
  const { createPipelineBuilder, bindingGroupLayouts, bindingGroups } =
    createBindingBuilder();

  // TODO: TYPESCRIPT BULLSHIT
  const module = device.createShaderModule({
    code: x.shader(
      createPipelineBuilder(GPUShaderStage.COMPUTE as unknown as GPUShaderStage)
    ),
  });

  // create module and pipeline on creation
  // then just update bind groups
  if (!prev) {
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: bindingGroupLayouts(),
      }),
      compute: { module, ...x },
    });
    return { pipeline, bindGroups: bindingGroups(pipeline) };
  } else {
    return { pipeline: prev, bindGroups: bindingGroups(prev) } as const;
  }
};

export const reactiveComputePipeline = (x: ComputePipelineDescriptor) => {
  return createMemo<[GPUComputePipeline, GPUBindGroup[]]>((prev) => {
    const { pipeline, bindGroups } = computePipeline(x, prev?.[0]);
    return [pipeline, bindGroups] as const;
  });
};

type ComputeRunnerParams = {
  workgroups: [number, number, number];
} & GPUComputePassDescriptor;
export const computePipelineRunner = (
  x: ComputePipelineDescriptor & { params: ComputeRunnerParams }
) => {
  const computePipeline = reactiveComputePipeline(x);
  return (
    encoder: GPUCommandEncoder,
    runParams?: Partial<ComputeRunnerParams>
  ) => {
    const params = { ...x.params, ...runParams };
    return computePass(encoder, params, (computePass) => {
      const [_computePipeline, computeBindGroups] = computePipeline();
      computePass.setPipeline(_computePipeline);
      computeBindGroups.forEach((bindGroup, i) =>
        computePass.setBindGroup(i, bindGroup)
      );
      computePass.dispatchWorkgroups(...params.workgroups);
    });
  };
};

export const reactiveRenderPipeline = (x: RenderPipelineDescriptor) => {
  return createMemo<[GPURenderPipeline, GPUBindGroup[]]>((prev) => {
    const { pipeline, bindGroups } = renderPipeline(x, prev?.[0]);
    return [pipeline, bindGroups] as const;
  });
};

// type RenderRunnerParams = {};
// export const renderPipelineRunner = (
//   x: RenderPipelineDescriptor & { params: RenderRunnerParams }
// ) => {
//   const computePipeline = reactiveComputePipeline(x);
//   return (
//     encoder: GPUCommandEncoder,
//     runParams?: Partial<ComputeRunnerParams>
//   ) => {
//     const params = { ...x.params, ...runParams };
//     return computePass(encoder, params, (computePass) => {
//       const [_computePipeline, computeBindGroups] = computePipeline();
//       computePass.setPipeline(_computePipeline);
//       computeBindGroups.forEach((bindGroup, i) =>
//         computePass.setBindGroup(i, bindGroup)
//       );
//       computePass.dispatchWorkgroups(...params.workgroups);
//     });
//   };
// };

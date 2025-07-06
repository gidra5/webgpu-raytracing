import { Accessor, createEffect, createSignal } from 'solid-js';
import { assert } from './utils';
import { vec3 } from 'gl-matrix';

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

export const reactiveUniformBuffer = <T extends number | Iterable<number>>(
  size: number,
  value: Accessor<T>,
  usage: GPUBufferUsageFlags = 0
) => {
  const buffer = createUniformBuffer(
    size * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    usage
  );

  createEffect(() => {
    const _value = value();
    const arrayBuffer =
      typeof _value === 'number'
        ? new Float32Array([_value])
        : new Float32Array(_value);
    writeBuffer(buffer, 0, arrayBuffer);
  });

  return buffer;
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

export const createBindGroup = (d: GPUBindGroupDescriptor) =>
  device.createBindGroup(d);

type RenderPipelineDescriptor = {
  vertexShader: (builder: PipelineBuilder) => string;
  fragmentShader?: (builder: PipelineBuilder) => string;
  fragmentPresentationFormatTarget?: Omit<GPUColorTargetState, 'format'>;
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
};
export const renderPipeline = (
  x: RenderPipelineDescriptor
): RenderPipelineBuilderResult => {
  const bindings: {
    visibility: GPUShaderStage;
    type: GPUBufferBindingType;
    buffer: GPUBuffer;
  }[][] = [[]];
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
      bindings[group].push({ buffer, visibility, type: access });
      const qualifier =
        access === 'read-only-storage'
          ? 'storage, read'
          : access === 'storage'
            ? 'storage'
            : 'uniform';
      return `@group(${group}) @binding(${binding}) var<${qualifier}> ${type};`;
    },
  });

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
  const bindGroupLayouts = bindings.map((group) =>
    device.createBindGroupLayout({
      entries: group.map(
        ({ visibility, type }, i): GPUBindGroupLayoutEntry => ({
          binding: i,
          visibility: visibility as unknown as number,
          buffer: { type },
        })
      ),
    })
  );
  const d: any = {
    layout: device.createPipelineLayout({ bindGroupLayouts }),
    vertex: { module: vertexShaderModule },
    ...x,
  };

  if (fragmentShaderModule) {
    d.fragment = {
      module: fragmentShaderModule,
      targets: [
        { format: presentationFormat, ...x.fragmentPresentationFormatTarget },
      ],
    };
  }

  const pipeline = device.createRenderPipeline(
    d as GPURenderPipelineDescriptor
  );
  const bindGroups = bindings.map((group, i) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(i),
      entries: group.map(
        ({ buffer }, i): GPUBindGroupEntry => ({
          binding: i,
          resource: { buffer },
        })
      ),
    })
  );
  return { pipeline, bindGroups };
};

type ComputePipelineDescriptor = {
  shader: (builder: PipelineBuilder) => string;
} & Omit<GPUProgrammableStage, 'module'>;
type ComputePipelineBuilderResult = {
  pipeline: GPUComputePipeline;
  bindGroups: GPUBindGroup[];
};
export const computePipeline = (
  x: ComputePipelineDescriptor
): ComputePipelineBuilderResult => {
  const bindings: {
    visibility: GPUShaderStage;
    type: GPUBufferBindingType;
    buffer: GPUBuffer;
  }[][] = [[]];
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
      bindings[group].push({ buffer, visibility, type: access });
      const qualifier =
        access === 'read-only-storage'
          ? 'storage, read'
          : access === 'storage'
            ? 'storage, read_write'
            : 'uniform';
      return `@group(${group}) @binding(${binding}) var<${qualifier}> ${type};`;
    },
  });

  // TODO: TYPESCRIPT BULLSHIT
  const module = device.createShaderModule({
    code: x.shader(
      createPipelineBuilder(GPUShaderStage.COMPUTE as unknown as GPUShaderStage)
    ),
  });
  const bindGroupLayouts = bindings.map((group) =>
    device.createBindGroupLayout({
      entries: group.map(
        ({ visibility, type }, i): GPUBindGroupLayoutEntry => ({
          binding: i,
          visibility: visibility as unknown as number,
          buffer: { type },
        })
      ),
    })
  );

  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts }),
    compute: { module, ...x },
  });
  const bindGroups = bindings.map((group, i) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(i),
      entries: group.map(
        ({ buffer }, i): GPUBindGroupEntry => ({
          binding: i,
          resource: { buffer },
        })
      ),
    })
  );
  return { pipeline, bindGroups };
};

export const reactiveComputePipeline = (x: ComputePipelineDescriptor) => {
  const [_computePipeline, setComputePipeline] =
    createSignal<GPUComputePipeline>();
  const [computeBindGroups, setComputeBindGroups] =
    createSignal<GPUBindGroup[]>();

  createEffect(() => {
    const { pipeline, bindGroups } = computePipeline(x);
    setComputePipeline(pipeline);
    setComputeBindGroups(bindGroups);
  });

  return [_computePipeline, computeBindGroups] as const;
};

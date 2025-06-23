import { setGPUTime, setJSTime, setRenderTime, store } from './store';
import { assert } from './utils';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const features: Partial<Record<GPUFeatureName, boolean>> = {};

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
  const presentationFormat =
    navigator.gpu.getPreferredCanvasFormat?.() ?? 'bgra8unorm';

  const context = canvas.getContext('webgpu');
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  return { device, context, presentationFormat };
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

export const init = async () => {
  resize();
  window.addEventListener('resize', resize);

  const { device, context, presentationFormat } = await getDevice();

  const { canTimestamp, querySet, submit } = getTimestampHandler(device);

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        code: `
          @vertex
          fn main(
            @builtin(vertex_index) VertexIndex : u32
          ) -> @builtin(position) vec4f {
            var pos = array<vec2f, 3>(
              vec2(0.0, 0.5),
              vec2(-0.5, -0.5),
              vec2(0.5, -0.5)
            );

            return vec4f(pos[VertexIndex], 0.0, 1.0);
          }
        `,
      }),
    },
    fragment: {
      module: device.createShaderModule({
        code: `
          @fragment
          fn main() -> @location(0) vec4f {
            return vec4(1.0, 0.0, 0.0, 1.0);
          }
        `,
      }),
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  async function frame() {
    const now = performance.now();
    setRenderTime(now);

    const encoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    renderPass(
      encoder,
      {
        colorAttachments: [
          {
            view: textureView,
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
      },
      (renderPass) => {
        renderPass.setPipeline(pipeline);
        renderPass.draw(3);
      }
    );

    await submit(encoder, () => {
      device.queue.submit([encoder.finish()]);
    });

    setJSTime(performance.now() - now);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
};

import { assert } from './utils';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const features: GPUFeatureName[] = [];

const getFeatures = (adapter: GPUAdapter) => {
  const canTimestamp = adapter.features.has('timestamp-query');

  if (canTimestamp) {
    features.push('timestamp-query');
  }

  return features;
};

const resize = () => {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
};

export const init = async () => {
  resize();
  window.addEventListener('resize', resize);

  assert(navigator.gpu, 'WebGPU not supported');
  const adapter = await navigator.gpu.requestAdapter();
  assert(adapter, 'No GPU adapter found');

  const requiredFeatures = features;
  const device = await adapter.requestDevice({ requiredFeatures });
  const presentationFormat =
    navigator.gpu.getPreferredCanvasFormat?.() ?? 'bgra8unorm';

  const context = canvas.getContext('webgpu');
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

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

  function frame() {
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: [0, 0, 0, 0], // Clear to transparent
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.draw(3);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
};

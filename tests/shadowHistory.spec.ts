import { expect, it } from 'vitest';
import { historyFilterWGSL } from '../src/shaders/filtering';

it('keeps a shadow edge sharp through repeated history accumulation', async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  device.pushErrorScope('validation');
  const resources: (GPUBuffer | GPUTexture)[] = [];
  try {
    const geometry = new ArrayBuffer(8 * 32);
    const floats = new Float32Array(geometry);
    const ints = new Uint32Array(geometry);
    for (let x = 0; x < 8; x++) {
      floats[x * 8] = x;
      floats[x * 8 + 3] = 1;
      ints[x * 8 + 5] = 1;
    }
    const buffer = device.createBuffer({
      size: geometry.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, geometry);
    const textures = [0, 1].map(() =>
      device.createTexture({
        size: [8, 1, 1],
        format: 'rgba32float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC,
      })
    );
    resources.push(buffer, ...textures);
    const initial = new Float32Array(32);
    for (let x = 0; x < 8; x++)
      initial.set([x < 4 ? 0 : 1, x < 4 ? 0 : 1, x < 4 ? 0 : 1, 1], x * 4);
    device.queue.writeTexture(
      { texture: textures[0] },
      initial,
      { bytesPerRow: 128 },
      [8, 1, 1]
    );
    const module = device.createShaderModule({
      code: /* wgsl */ `
      struct Geometry { position:vec3f, depth:f32, faceIdx:u32, objectIdx:u32 }
      struct Face { plane:vec4f, materialIdx:u32 }
      const faces = array<Face,1>(Face(vec4f(0,1,0,0),0u));
      const viewport = vec2u(8,1);
      var<private> layer:u32;
      @group(0) @binding(0) var<storage,read> prevGeometryBuffer:array<Geometry>;
      @group(0) @binding(1) var prevImageTexture:texture_storage_2d_array<rgba32float,read>;
      @group(0) @binding(2) var outputImage:texture_storage_2d_array<rgba32float,write>;
      fn geometryIdx(pixel:vec2u)->u32 { return pixel.x; }
      ${historyFilterWGSL(1, 1e-4)}
      @compute @workgroup_size(8) fn main(@builtin(global_invocation_id) id:vec3u) {
        let g = prevGeometryBuffer[id.x];
        // A subpixel projection must not mix the lit and shadowed estimates.
        let history = bilateralFilter(vec2f(f32(id.x)+0.2,0),g,g.position,vec3f(0));
        let fresh = select(0.0,1.0,id.x>=4u);
        textureStore(outputImage,vec2u(id.xy),0,history+vec4f(fresh,fresh,fresh,1));
      }
    `,
    });
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    const groups = [0, 1].map((i) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          {
            binding: 1,
            resource: textures[i].createView({ dimension: '2d-array' }),
          },
          {
            binding: 2,
            resource: textures[1 - i].createView({ dimension: '2d-array' }),
          },
        ],
      })
    );
    const encoder = device.createCommandEncoder();
    for (let frame = 0; frame < 32; frame++) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, groups[frame % 2]);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    const readback = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    resources.push(readback);
    encoder.copyTextureToBuffer(
      { texture: textures[0] },
      { buffer: readback, bytesPerRow: 256 },
      [8, 1, 1]
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange());
    for (let x = 0; x < 8; x++) {
      expect(result[x * 4 + 3]).toBe(33);
      for (let channel = 0; channel < 3; channel++)
        expect(result[x * 4 + channel] / result[x * 4 + 3]).toBe(x < 4 ? 0 : 1);
    }
    readback.unmap();
    expect(await device.popErrorScope()).toBeNull();
  } finally {
    resources.forEach((resource) => resource.destroy());
    device.destroy();
  }
});

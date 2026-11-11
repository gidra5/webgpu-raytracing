import { expect, it } from 'vitest';
import { historyFilterWGSL } from '../src/shaders/filtering';

it('preserves surface history at grazing angles, silhouettes and viewport edges', async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter!.requestDevice();
  device.pushErrorScope('validation');
  const resources: (GPUBuffer | GPUTexture)[] = [];
  try {
    const scenarios = [
      {
        uv: [-0.2, 0],
        pixel: 0,
        layer: 0,
        height: 0,
        object: 1,
        face: 0,
        valid: true,
      },
      {
        uv: [1.8, 0],
        pixel: 2,
        layer: 0,
        height: 0,
        object: 1,
        face: 0,
        valid: true,
      },
      {
        uv: [1, 0],
        pixel: 1,
        layer: 1,
        height: 0,
        object: 1,
        face: 0,
        valid: true,
      },
      {
        uv: [1, 0],
        pixel: 1,
        layer: 0,
        height: 0.1,
        object: 1,
        face: 0,
        valid: false,
      },
      {
        uv: [1, 0],
        pixel: 1,
        layer: 0,
        height: 0,
        object: 2,
        face: 0,
        valid: false,
      },
      {
        uv: [1, 0],
        pixel: 1,
        layer: 0,
        height: 0,
        object: 1,
        face: 1,
        valid: false,
      },
    ];
    for (const scenario of scenarios) {
      const geometry = new ArrayBuffer(6 * 32);
      const floats = new Float32Array(geometry);
      const ints = new Uint32Array(geometry);
      // Surround the valid sample with unrelated foreground geometry.
      for (let i = 0; i < 6; i++) {
        floats[i * 8 + 3] = 1;
        ints[i * 8 + 5] = 99;
      }
      const index = (scenario.layer * 3 + scenario.pixel) * 8;
      // A large tangential displacement models a grazing surface footprint.
      floats.set([1000, scenario.height, 0, 1], index);
      ints[index + 4] = scenario.face;
      ints[index + 5] = scenario.object;
      const buffer = device.createBuffer({
        size: geometry.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, geometry);
      const image = device.createTexture({
        size: [3, 1, 2],
        format: 'rgba32float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
      });
      const imageData = new Float32Array(6 * 4);
      for (let i = 0; i < 6; i++) imageData.set([900, 900, 900, 1], i * 4);
      imageData.set(
        [40, 20, 10, 10],
        (scenario.layer * 3 + scenario.pixel) * 4
      );
      device.queue.writeTexture(
        { texture: image },
        imageData,
        { bytesPerRow: 48, rowsPerImage: 1 },
        [3, 1, 2]
      );
      const result = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const readback = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      resources.push(buffer, image, result, readback);
      const module = device.createShaderModule({
        code: /* wgsl */ `
        struct Geometry { position: vec3f, depth: f32, faceIdx: u32, objectIdx: u32 }
        struct Face { plane: vec4f, materialIdx: u32 }
        const faces = array<Face,2>(Face(vec4f(0,1,0,0),0u),Face(vec4f(0,1,0,0),0u));
        const viewport = vec2u(3,1);
        var<private> layer: u32 = 1u;
        @group(0) @binding(0) var<storage,read> prevGeometryBuffer: array<Geometry>;
        @group(0) @binding(1) var prevImageTexture: texture_storage_2d_array<rgba32float,read>;
        @group(0) @binding(2) var<storage,read_write> result: vec4f;
        fn geometryIdx(pixel:vec2u)->u32 { return pixel.x + layer*3u; }
        ${historyFilterWGSL(2, 1e-4)}
        @compute @workgroup_size(1) fn main() {
          result = bilateralFilter(vec2f(${scenario.uv[0]},${scenario.uv[1]}), Geometry(vec3f(0),1,0u,1u),vec3f(0),vec3f(0));
          if layer != 1u { result = vec4f(-1); }
        }
      `,
      });
      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: image.createView({ dimension: '2d-array' }) },
          { binding: 2, resource: { buffer: result } },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(result, 0, readback, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange());
      const expected = scenario.valid ? [40, 20, 10, 10] : [0, 0, 0, 0];
      actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i], 4));
      readback.unmap();
    }
    expect(await device.popErrorScope()).toBeNull();
  } finally {
    resources.forEach((resource) => resource.destroy());
    device.destroy();
  }
});

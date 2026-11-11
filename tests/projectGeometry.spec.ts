import { describe, expect, it } from 'vitest';
import {
  PROJECT_GEOMETRY_WORKGROUP_SIZE,
  projectGeometryWGSL,
} from '../src/shaders/projectGeometry';

const gpuSupported = typeof navigator !== 'undefined' && !!navigator.gpu;
const describeIf = gpuSupported ? describe : describe.skip;

describeIf('geometry projection GPU pipeline', () => {
  it('projects padded vec3 vertices into complete clip coordinates across workgroups', async () => {
    const adapter = await navigator.gpu.requestAdapter();
    expect(adapter).toBeTruthy();
    const device = await adapter!.requestDevice();
    const buffers: GPUBuffer[] = [];
    const uncapturedErrors: string[] = [];
    device.addEventListener('uncapturederror', (event) => {
      uncapturedErrors.push(event.error.message);
    });
    device.pushErrorScope('validation');

    let validationError: GPUError | null;
    try {
      // The final workgroup is partial, and array<vec3f> has a 16-byte stride.
      const vertexCount = PROJECT_GEOMETRY_WORKGROUP_SIZE + 3;
      const vertices = new Float32Array(vertexCount * 4);
      const expected = new Float32Array(vertices.length);
      for (let i = 0; i < vertexCount; i += 1) {
        const x = (i - 33) / 8;
        const y = ((i % 7) - 3) / 4;
        const z = ((i % 11) - 5) / 2;
        vertices.set([x, y, z, 12345], i * 4);
        expected.set(
          [2 * x + 5, 3 * y + 6, 4 * z + 7, 0.5 * x - 0.25 * y + 0.125 * z + 2],
          i * 4
        );
      }
      // Column-major matrix with a varying homogeneous w coordinate.
      const matrix = new Float32Array([
        2, 0, 0, 0.5, 0, 3, 0, -0.25, 0, 0, 4, 0.125, 5, 6, 7, 2,
      ]);

      const verticesBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      buffers.push(verticesBuffer);
      device.queue.writeBuffer(verticesBuffer, 0, vertices);

      const outVerticesBuffer = device.createBuffer({
        size: expected.byteLength,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.VERTEX |
          GPUBufferUsage.COPY_SRC,
      });
      buffers.push(outVerticesBuffer);

      const viewProjBuffer = device.createBuffer({
        size: matrix.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      buffers.push(viewProjBuffer);
      device.queue.writeBuffer(viewProjBuffer, 0, matrix);

      const readback = device.createBuffer({
        size: expected.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      buffers.push(readback);

      const shader = device.createShaderModule({
        code: projectGeometryWGSL(/* wgsl */ `
          @group(0) @binding(0) var<storage, read> vertices: array<vec3f>;
          @group(0) @binding(1) var<storage, read_write> outVertices: array<vec4f>;
          @group(0) @binding(2) var<uniform> viewProjMatrix: mat4x4f;
        `),
      });
      const compilationInfo = await shader.getCompilationInfo();
      expect(
        compilationInfo.messages.filter((message) => message.type === 'error')
      ).toEqual([]);

      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: shader, entryPoint: 'main' },
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: verticesBuffer } },
          { binding: 1, resource: { buffer: outVerticesBuffer } },
          { binding: 2, resource: { buffer: viewProjBuffer } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(vertexCount / PROJECT_GEOMETRY_WORKGROUP_SIZE)
      );
      pass.end();
      encoder.copyBufferToBuffer(
        outVerticesBuffer,
        0,
        readback,
        0,
        expected.byteLength
      );
      device.queue.submit([encoder.finish()]);

      await readback.mapAsync(GPUMapMode.READ);
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      expect(actual).toEqual(expected);
    } finally {
      validationError = await device.popErrorScope();
      buffers.forEach((buffer) => buffer.destroy());
      device.destroy();
    }

    expect(validationError).toBeNull();
    expect(uncapturedErrors).toEqual([]);
  });
});

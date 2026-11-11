import { expect, it } from 'vitest';

it('renders a visible frame to the canvas without GPU validation errors', async () => {
  const canvas = document.createElement('canvas');
  canvas.id = 'canvas';
  canvas.style.cssText = 'width:32px;height:32px';
  document.body.append(canvas);
  const errors: string[] = [];
  let device: GPUDevice;
  let output: GPUTexture;
  let readback: GPUBuffer;
  let capture = false;
  const requestDevice = GPUAdapter.prototype.requestDevice;
  const configure = GPUCanvasContext.prototype.configure;
  const getTexture = GPUCanvasContext.prototype.getCurrentTexture;
  const submit = GPUQueue.prototype.submit;
  GPUAdapter.prototype.requestDevice = async function (descriptor) {
    device = await requestDevice.call(this, descriptor);
    device.addEventListener('uncapturederror', (e) =>
      errors.push(e.error.message)
    );
    return device;
  };
  GPUCanvasContext.prototype.configure = function (configuration) {
    return configure.call(this, {
      ...configuration,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  };
  GPUCanvasContext.prototype.getCurrentTexture = function () {
    output = getTexture.call(this);
    return output;
  };
  GPUQueue.prototype.submit = function (commands) {
    if (capture && output) {
      const encoder = device.createCommandEncoder();
      readback = device.createBuffer({
        size: 256 * output.height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyTextureToBuffer(
        { texture: output },
        { buffer: readback, bytesPerRow: 256 },
        { width: output.width, height: output.height }
      );
      return submit.call(this, [...commands, encoder.finish()]);
    }
    return submit.call(this, commands);
  };
  try {
    const { renderFrame } = await import('../src/render');
    await renderFrame(performance.now());
    capture = true;
    await renderFrame(performance.now());
    expect(errors).toEqual([]);
    expect(readback).toBeTruthy();
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange());
    let litPixels = 0;
    const colors = new Set<number>();
    for (let y = 0; y < output.height; y++) {
      for (let x = 0; x < output.width; x++) {
        const i = y * 256 + x * 4;
        colors.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]);
        if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 0) litPixels++;
      }
    }
    expect(litPixels).toBeGreaterThan(0);
    expect(colors.size).toBeGreaterThan(1);
    expect(errors).toEqual([]);
    readback.unmap();
  } finally {
    GPUAdapter.prototype.requestDevice = requestDevice;
    GPUCanvasContext.prototype.configure = configure;
    GPUCanvasContext.prototype.getCurrentTexture = getTexture;
    GPUQueue.prototype.submit = submit;
    readback?.destroy();
    device?.destroy();
    canvas.remove();
  }
}, 30000);

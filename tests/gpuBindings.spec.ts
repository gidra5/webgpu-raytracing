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
  computePipeline,
  createStorageBuffer,
  createTexture,
  getDevice,
} from '../src/gpu';

const gpuSupported = typeof navigator !== 'undefined' && !!navigator.gpu;
const describeIf = gpuSupported ? describe : describe.skip;
const textureCases = [
  { name: 'plain 2D texture', isArray: false, layers: 1 },
  { name: 'single-layer texture array', isArray: true, layers: 1 },
  { name: 'two-layer texture array', isArray: true, layers: 2 },
];

describeIf('GPU texture binding builder', () => {
  let device: GPUDevice;
  let context: GPUCanvasContext;
  let resources: (GPUTexture | GPUBuffer)[];
  const uncapturedErrors: string[] = [];

  beforeAll(async () => {
    const canvas = document.createElement('canvas');
    context = canvas.getContext('webgpu')!;
    expect(context).toBeTruthy();
    device = await getDevice(context);
    device.addEventListener('uncapturederror', (event) => {
      uncapturedErrors.push(event.error.message);
    });
  });

  beforeEach(() => {
    resources = [];
    uncapturedErrors.length = 0;
    device.pushErrorScope('validation');
  });

  afterEach(async () => {
    const error = await device.popErrorScope();
    resources.forEach((resource) => resource.destroy());
    expect(error?.message).toBeUndefined();
    expect(uncapturedErrors).toEqual([]);
  });

  afterAll(() => {
    context?.unconfigure();
    device?.destroy();
  });

  it.each(textureCases)(
    'binds a sampled r32float $name',
    ({ isArray, layers }) => {
      const texture = createTexture(
        { width: 1, height: 1, depthOrArrayLayers: layers },
        { dimension: '2d', format: 'r32float' }
      );
      const resultBuffer = createStorageBuffer(16);
      resources.push(texture, resultBuffer);

      const result = computePipeline({
        shader: (builder) => /* wgsl */ `
        ${builder.bindTexture('image', 'unfilterable-float', texture, isArray)}
        ${builder.bindVarBuffer('storage', 'result: array<vec4f>', resultBuffer)}

        @compute @workgroup_size(1)
        fn main() {
          result[0] = textureLoad(image, vec2u(0), ${isArray ? '0, ' : ''}0);
        }
      `,
      });
      expect(result.bindGroups).toHaveLength(1);
    }
  );

  it.each(textureCases)(
    'binds storage reads and writes for a $name',
    ({ isArray, layers }) => {
      const textures = Array.from({ length: 3 }, () =>
        createTexture(
          { width: 1, height: 1, depthOrArrayLayers: layers },
          {
            dimension: '2d',
            format: 'r32float',
            usage: GPUTextureUsage.STORAGE_BINDING,
          }
        )
      );
      resources.push(...textures);
      const coordinates = `vec2u(0)${isArray ? ', 0' : ''}`;

      const result = computePipeline({
        shader: (builder) => /* wgsl */ `
        ${builder.bindStorageTexture('inputImage', 'read-only', textures[0], isArray)}
        ${builder.bindStorageTexture('outputImage', 'write-only', textures[1], isArray)}
        ${builder.bindStorageTexture('scratchImage', 'read-write', textures[2], isArray)}

        @compute @workgroup_size(1)
        fn main() {
          let value = textureLoad(inputImage, ${coordinates});
          textureStore(scratchImage, ${coordinates}, value);
          textureStore(outputImage, ${coordinates}, textureLoad(scratchImage, ${coordinates}));
        }
      `,
      });
      expect(result.bindGroups).toHaveLength(1);
    }
  );
});

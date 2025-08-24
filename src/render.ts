import { mat4, vec2 } from 'gl-matrix';
import {
  computePass,
  createStorageBuffer,
  createUniformBuffer,
  getDevice,
  getTimestampHandler,
  PipelineBuilder,
  reactiveComputePipeline,
  reactiveRenderPipeline,
  reactiveUniformBuffer,
  renderBundlePass,
  renderPass,
  renderPipeline,
  writeUint32Buffer,
  writeVec2fBuffer,
} from './gpu';
import {
  FovOrientation,
  incrementCounter,
  LensShape,
  invMat,
  ProjectionType,
  reprojectionFrustrum,
  setRenderGPUTime,
  setRenderJSTime,
  setView,
  NormalsType,
  store,
  Tonemapping,
  viewMatrix,
  viewProjectionMatrix,
  ReprojectionFiltering,
  SkyboxType,
  BlitView,
} from './store';
import { createEffect, createSignal } from 'solid-js';
import rng from './shaders/rng';
import tonemapping from './shaders/tonemapping';
import {
  loadMaterialsToBuffers,
  loadModels,
  loadModelsToBuffers,
  loadSkybox,
} from './scene';
import double from './shaders/double';
import intervals from './shaders/intervals';
import rayIntersect from './shaders/rayIntersect';
import bvh from './shaders/bvh';
import derivatives from './shaders/derivatives';
import filtering from './shaders/filtering';
import utils from './shaders/utils';
import constants from './shaders/constants';
import raygen from './shaders/raygen';
import structs from './shaders/structs';
import face from './shaders/face';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const context = canvas.getContext('webgpu');
const device = await getDevice(context as GPUCanvasContext);
const [imageBuffer, setImageBuffer] = createSignal<GPUBuffer>();
const [prevImageBuffer, setPrevImageBuffer] = createSignal<GPUBuffer>();
const [geometryBuffer, setGeometryBuffer] = createSignal<GPUBuffer>();
const [prevGeometryBuffer, setPrevGeometryBuffer] = createSignal<GPUBuffer>();
const [blitRenderBundle, setBlitRenderBundle] = createSignal<GPURenderBundle>();
const [debugBVHRenderBundle, setDebugBVHRenderBundle] =
  createSignal<GPURenderBundle>();
const [prevView, setPrevView] = createSignal<mat4>(mat4.create(), {
  equals: (a, b) => a && mat4.exactEquals(a, b),
});
const _prevViewInv = invMat(prevView);
const _viewInv = invMat(viewMatrix);
const [prevViewInvBufferHi, prevViewInvBufferLo] = reactiveUniformBuffer(
  16,
  _prevViewInv
);
const [viewInvBufferHi, viewInvBufferLo] = reactiveUniformBuffer(16, _viewInv);
const _reprojectionFrustrum = reprojectionFrustrum(prevView);
const [viewBuffer] = reactiveUniformBuffer(
  16,
  viewMatrix,
  GPUBufferUsage.COPY_SRC
);
const [prevViewBuffer] = reactiveUniformBuffer(
  16,
  prevView,
  GPUBufferUsage.COPY_SRC
);
const [viewProjBuffer] = reactiveUniformBuffer(16, viewProjectionMatrix);
const [reprojectionFrustrumBufferHi, reprojectionFrustrumBufferLo] =
  reactiveUniformBuffer(12, _reprojectionFrustrum, GPUBufferUsage.COPY_SRC);
const jitterBuffer = createUniformBuffer(
  16,
  'Jitter Buffer',
  GPUBufferUsage.COPY_SRC
);
const prevJitterBuffer = createUniformBuffer(
  16,
  'Prev Jitter Buffer',
  GPUBufferUsage.COPY_DST
);

console.log('loading models');
const { models, materials } = await loadModels();

console.log('loading materials');
const { materialsBuffer } = await loadMaterialsToBuffers(materials);

console.log('loading scene');
const { facesBuffer, bvhBuffer, bvhFacesBuffer, bvhCount, modelsBuffer } =
  // await loadModelsToBuffers(models);
  await loadModelsToBuffers([
    models[10],
    models[2],
    models[6],
    models[11],
    models[8],
    models[5],
    models[3],
    models[4],
  ]);
// await loadModelsToBuffers([models[1]]);

console.log('loading skybox');
type SkyboxData = {
  texture: GPUTexture;
  importanceSampleTexture: GPUTexture;
  sampler: GPUSampler;
};
const [skybox, setSkybox] = createSignal<SkyboxData>();

createEffect(() => {
  if (store.skybox === SkyboxType.Exr) {
    (async () => {
      const { texture, importanceSampleTexture } = await loadSkybox();
      const sampler = device.createSampler();

      setSkybox({ texture, importanceSampleTexture, sampler });
    })();
  }
});

const seedUniformBuffer = createUniformBuffer(4);
const counterUniformBuffer = createUniformBuffer(4);

const resize = () => {
  const scale = devicePixelRatio * store.resolutionScale;
  canvas.width = canvas.clientWidth * scale;
  canvas.height = canvas.clientHeight * scale;
  setView(vec2.fromValues(canvas.width, canvas.height));
};

createEffect<() => void>((destroy) => {
  destroy?.();
  resize();
  window.addEventListener('resize', resize);
  return () => window.removeEventListener('resize', resize);
});

createEffect<GPUBuffer[]>((prevBuffers) => {
  if (prevBuffers) prevBuffers.forEach((b) => b.destroy());
  const width = store.view[0] + 1;
  const height = store.view[1];

  // color + accumulated samples count
  // 4 images in 1 buffer
  const imageSize =
    store.imageLayers * Float32Array.BYTES_PER_ELEMENT * 4 * width * height;
  const current = createStorageBuffer(
    imageSize,
    'Raytraced Image Buffer',
    GPUBufferUsage.COPY_SRC
  );
  setImageBuffer(current);
  const prev = createStorageBuffer(
    imageSize,
    'Prev Raytraced Image Buffer',
    GPUBufferUsage.COPY_DST
  );
  setPrevImageBuffer(prev);

  const geometryBufferItemSize =
    store.imageLayers * Float32Array.BYTES_PER_ELEMENT * 32;
  const geometryBufferSize =
    store.geometryBufferScale * geometryBufferItemSize * width * height;
  const currentGeometry = createStorageBuffer(
    geometryBufferSize,
    'Geometry Buffer',
    GPUBufferUsage.COPY_SRC
  );
  setGeometryBuffer(currentGeometry);
  const prevGeometry = createStorageBuffer(
    geometryBufferSize,
    'Prev Geometry Buffer',
    GPUBufferUsage.COPY_DST
  );
  setPrevGeometryBuffer(prevGeometry);

  return [current, prev, currentGeometry, prevGeometry];
});

createEffect(() => {
  const { pipeline, bindGroups } = renderPipeline({
    vertexShader: () => /* wgsl */ `
      // xy pos + uv
      const FULLSCREEN_TRIANGLE = array<vec4<f32>, 3>(
        vec4(-1, 3, 0, 2),
        vec4(-1, -1, 0, 0),
        vec4(3, -1, 2, 0),
      );

      struct VertexOutput {
        @builtin(position) Position: vec4f,
        @location(0) uv: vec2f,
      }
      
      @vertex
      fn main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
        var output: VertexOutput;
        output.Position = vec4<f32>(FULLSCREEN_TRIANGLE[VertexIndex].xy, 0.0, 1.0);
        output.uv = FULLSCREEN_TRIANGLE[VertexIndex].zw;
        return output;
      }
    `,
    fragmentShader: (x) => /* wgsl */ `
      ${tonemapping}
      ${utils}
      ${constants}

      ${x.bindVarBuffer('read-only-storage', 'imageBuffer: array<vec4f>', imageBuffer())}
      ${x.bindVarBuffer('read-only-storage', 'prevImageBuffer: array<vec4f>', prevImageBuffer())}

      const viewport = vec2u(${store.view[0]}, ${store.view[1]});
      const viewportf = vec2f(viewport);
      const exposure = ${store.exposure};

      fn imageColor(color: vec3f) -> vec3f {
        let tonemapped = tonemap(color);
        return linear_to_srgb(tonemapped);
      }

      fn getColor(idx: u32, pos: vec2f) -> vec3f {
        if ${store.blitView == BlitView.Image} {
          let value = imageBuffer[idx];
          let color = value.rgb / value.w * exposure;
          if is_nan3(color) {
            return vec3f(1, 0, 0);
          }
          return imageColor(color); 
        } else if ${store.blitView == BlitView.Image2} {
          let value = imageBuffer[idx + 1 * viewport.x * viewport.y];
          let color = value.rgb / value.w * exposure;
          return imageColor(color); 
        } else if ${store.blitView == BlitView.Image3} {
          let value = imageBuffer[idx + 2 * viewport.x * viewport.y];
          let color = value.rgb / value.w * exposure;
          return imageColor(color);
        } else if ${store.blitView == BlitView.PrevImage} {
          let value = prevImageBuffer[idx];
          let color = value.rgb / value.w * exposure;
          return imageColor(color); 
        } else if ${store.blitView == BlitView.PrevImage2} {
          let value = prevImageBuffer[idx + 1 * viewport.x * viewport.y];
          let color = value.rgb / value.w * exposure;
          return imageColor(color); 
        } else if ${store.blitView == BlitView.PrevImage3} {
          let value = prevImageBuffer[idx + 2 * viewport.x * viewport.y];
          let color = value.rgb / value.w * exposure;
          return imageColor(color);
        } else if ${store.blitView == BlitView.Normals} {
          let value = imageBuffer[idx];
          return value.rgb;
        } else if ${store.blitView == BlitView.Reproject} {
          let value = imageBuffer[idx];
          return value.rgb;
        } else if ${store.blitView == BlitView.Depth} {
          let value = imageBuffer[idx];
          return value.rgb / value.w;
        } else if ${store.blitView == BlitView.PrevDepth} {
          let value = prevImageBuffer[idx];
          return value.rgb / value.w;
        } else if ${store.blitView == BlitView.DepthDelta} {
          let value = imageBuffer[idx] - prevImageBuffer[idx];
          return value.rgb / value.w;
        }
        return vec3f(0);
      }

      fn tonemap(c: vec3f) -> vec3f {
        if ${store.tonemapping} == ${Tonemapping.Reinhard} {
          return reinhard(c);
        } else if ${store.tonemapping} == ${Tonemapping.Filmic} {
          return filmic(c);
        } else if ${store.tonemapping} == ${Tonemapping.Aces} {
          return aces(c);
        } else if ${store.tonemapping} == ${Tonemapping.Lottes} {
          return lottes(c);
        } else {
          return c;
        }
      }

      @fragment
      fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
        let pos = uv * viewportf;
        let upos = vec2u(pos);
        let idx = upos.y * viewport.x + upos.x;
        return vec4f(getColor(idx, pos),1);
      }
    `,
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
    },
  });

  const bundle = renderBundlePass({}, (renderPass) => {
    renderPass.setPipeline(pipeline);
    bindGroups.forEach((bindGroup, i) => renderPass.setBindGroup(i, bindGroup));
    renderPass.draw(3);
  });

  setBlitRenderBundle(bundle);
});

const scene = () => /* wgsl */ `
  fn sceneAnyHit(ray: Ray, maxDist: f32) -> bool {
    return rayIntersectBVHAnyHit(ray, maxDist);
  }

  fn scene(ray: Ray, i: Interval) -> BVHIntersectionResult {
    return rayIntersectBVH(ray, i);
  }

  fn sceneBackface(ray: Ray, i: Interval) -> BVHIntersectionResult {
    return rayBackfaceIntersectBVH(ray, i);
  }

  fn objectFaceHit(faceIdx: u32, objectIdx: u32, ray: Ray, maxDist: f32) -> BVHIntersectionResult {
    var hit: BVHIntersectionResult;
    hit.hit = false;
    hit.barycentric = vec3f(maxDist, 0, 0);
    hit.faceIdx = faceIdx;
    hit.objectIdx = objectIdx;
    
    {
      let model = models[objectIdx];
      let face = faces[model.faces.offset + faceIdx];
      let _hit = rayIntersectFace(ray, face, Interval(min_dist, maxDist));
      if _hit.hit {
        hit.hit = true;
        hit.barycentric = _hit.barycentric;
      }
    }

    {
      let model = models[objectIdx];
      let _hit = rayIntersectObjectBVH(ray, objectIdx, Interval(min_dist, hit.barycentric.x + EPSILON));
      if _hit.hit {
        hit = _hit;
      }
    }

    return hit;
  }

  fn objectFaceAnyHit(faceIdx: u32, objectIdx: u32, ray: Ray, maxDist: f32) -> bool {
    {
      let model = models[objectIdx];
      let face = faces[model.faces.offset + faceIdx];
      let _hit = rayIntersectFace(ray, face, Interval(min_dist, maxDist));
      if _hit.hit {
        return true;
      }
    }

    {
      let model = models[objectIdx];
      let _hit = rayIntersectObjectBVH(ray, objectIdx, Interval(min_dist, maxDist));
      if _hit.hit {
        return true;
      }
    }

    return false;
  }

  struct SceneSample {
    p: f32, // 1/pdf
    point: vec3f,
    normal: vec3f,
    uv: vec2f,
    materialIdx: u32,
  }

  fn sampleScene() -> SceneSample {
    let randomModelIdx = random_1u() % arrayLength(&models); 
    let model = models[randomModelIdx];
    var sample = sampleModel(model);
    sample.p *= f32(arrayLength(&models));
    return sample;
  }

  fn sampleLights() -> SceneSample {
    var sample = sampleModel(models[0]);
    return sample;
  }

  fn sampleModel(model: Model) -> SceneSample {
    let randomFaceIdx = random_1u() % model.faces.count;
    let face = faces[model.faces.offset + randomFaceIdx];
    var sample = sampleFace(face);
    return SceneSample(sample.inv_pdf * f32(model.faces.count), sample.point, sample.normal, sample.texture, sample.materialIdx);
  }
`;

const skyboxModule = (x: PipelineBuilder) => {
  if (skybox()) {
    return /* wgsl */ `
      ${x.bindTexture('skyboxImportanceSampleTexture', 'unfilterable-float', skybox().importanceSampleTexture)}
      ${x.bindSampler('skyboxSampler', 'non-filtering', skybox().sampler)}
      ${x.bindTexture('skyboxTexture', 'unfilterable-float', skybox().texture)}

      // Function to sample the skybox
      fn sampleSkyboxTexture(uv: vec2f) -> vec3f {
        let color = textureSampleLevel(skyboxTexture, skyboxSampler, uv, 0);
        return srgb_to_linear(color.xyz);
      }

      // Function to sample the skybox
      fn sampleSkybox(dir: vec3f) -> vec3f {
        let u = (atan2(dir.z, dir.x) * INV_PI + 1) * 0.5;
        let v = 1 - acos(dir.y) * INV_PI;
        let uv = vec2<f32>(u, v);

        return sampleSkyboxTexture(uv);
      }
      
      fn importanceSampleSkybox(uv: vec2f) -> vec3f {
        let sample = textureSampleLevel(skyboxImportanceSampleTexture, skyboxSampler, uv, 0);
        return sample.xyz;
        // let uv_u = floor(uv);
        // let uv_f = fract(uv);
        // let m = mat4x4f(
        //   textureSampleLevel(skyboxImportanceSampleTexture, skyboxSampler, uv_u, 0),
        //   textureSampleLevel(skyboxImportanceSampleTexture, skyboxSampler, uv_u + vec2f(1, 0), 0),
        //   textureSampleLevel(skyboxImportanceSampleTexture, skyboxSampler, uv_u + vec2f(0, 1), 0),
        //   textureSampleLevel(skyboxImportanceSampleTexture, skyboxSampler, uv_u + vec2f(1, 1), 0),
        // );
        // let value = bilinearInterpolation4(uv_f, m);

        // return value.xyz;
      }

      // Function to convert UV coordinates to a 3D direction vector
      fn uv_to_direction(uv: vec2<f32>) -> vec3<f32> {
        // Inverse of the equirectangular projection
        let phi = (uv.x * 2 - 1) * PI; 
        let theta = (1 - uv.y) * PI;

        let sinTheta = sin(theta);
        let cosTheta = sqrt(1 - sinTheta * sinTheta);
        let sinPhi = sin(phi);
        let cosPhi = sqrt(1 - sinPhi * sinPhi);

        // Standard spherical coordinates to Cartesian (assuming Y-up)
        let x = sinTheta * cosPhi;
        let y = cosTheta;
        let z = sinTheta * sinPhi;
        return normalize(vec3<f32>(x, y, z));
      }
    `;
  }

  return /* wgsl */ `
    // Function to sample the skybox
    fn sampleSkybox(dir: vec3f) -> vec3f {
      return vec3f(0.2) / ${store.exposure};
    }
  `;
};

const reproject = () => /* wgsl */ `
  struct ReprojectionResult {
    color: vec4f, // color + accumulated samples count
  }

  fn roundUpTo2(x: vec2f, n: f32) -> vec2f {
    return round(x / n) * n;
  }

  fn roundUpTo(x: f32, n: f32) -> f32 {
    return round(x / n) * n;
  }

  fn reprojectPoint(p: vec3f) -> vec2f {
    if ${store.reprojection.doubleAccuracy} {
      return merge_2(reprojectPointPrecise(p)); 
    } else {
      return reprojectPure(p, prevViewInvMatrix, prevJitter);
    }
  }

  fn unprojectPoint(p: vec3f) -> vec2f {
    if ${store.reprojection.doubleAccuracy} {
      return merge_2(unprojectPointPrecise(p));
    } else {
      return reprojectPure(p, viewInvMatrix, jitter);
    }
  }

  fn reprojectPure(p: vec3f, viewInv: mat4x4f, jitter: vec2f) -> vec2f {
    let worldSpaceJitter = jitter / viewportf;
    let pp = viewInv * vec4(p, 1.);
    let duv = reprojectionFrustrum * (pp.xyz - vec3f(worldSpaceJitter, 0));
    return duv.xy / duv.zw;
  }

  fn reprojectPointPrecise(p: vec3f) -> F64_2 {
    return reprojectPurePrecise(p, F64_4x4(prevViewInvMatrix, prevViewInvMatrixLo), prevJitter);
  }

  fn unprojectPointPrecise(p: vec3f) -> F64_2 {
    return reprojectPurePrecise(p, F64_4x4(viewInvMatrix, viewInvMatrixLo), jitter);
  }

  fn reprojectPurePrecise(p: vec3f, viewInv: F64_4x4, jitter: vec2f) -> F64_2 {
    let worldSpaceJitter = jitter / viewportf;
    let pp = matmul_4x4(viewInv, split_4(vec4(p, 1.)));
    let dp = sub_3(F64_3(pp.hi.xyz, pp.lo.xyz), split_3(vec3f(worldSpaceJitter, 0)));
    let duv = matmul_3x4(F64_3x4(reprojectionFrustrum, reprojectionFrustrumLo), dp);
    return div_2(F64_2(duv.hi.xy, duv.lo.xy), F64_2(duv.hi.zw, duv.lo.zw));
  }

  const threshold = ${store.reprojection.pointAccuracy * store.reprojection.pointAccuracy};

  struct Reprojection {
    uv: vec2f,
    g: Geometry,
    p: vec3f,
    dp: vec3f,
    d: f32,
    layer: u32,
  }

  fn reprojection(uv: vec2f, p: vec3f) -> Reprojection {
    let uv_g = sampleGeometryAll(uv, &prevGeometryBuffer);
    let uv_p = uv_g.position;
    let dp = uv_p - p;
    let d = dot(dp, dp);
    return Reprojection(uv, uv_g, uv_p, dp, d, layer);
  }

  fn gradientReprojection(uv: vec2f, p: vec3f, dp: vec3f) -> Reprojection {
    let dx = sampleGeometryAll(uv + vec2f(1, 0), &prevGeometryBuffer).position - p;
    let dy = sampleGeometryAll(uv + vec2f(0, 1), &prevGeometryBuffer).position - p;

    let A = mat2x3f(dx, dy);
    let At = transpose(A);
    let AtA = At * A;
    let det = 1/(AtA[1][1] * AtA[0][0] - AtA[0][1] * AtA[1][0]);
    let AtAinv = mat2x2f(
      AtA[1][1], -AtA[1][0],
      -AtA[0][1], AtA[0][0]
    ) * det;
    let Atdp = At * dp;
    let duv = AtAinv * Atdp;

    return reprojection(uv + duv, p);
  }

  fn reprojectPrecise(uv_error: vec2f, p: vec3f) -> Reprojection {
    let uv = reprojectPoint(p) - uv_error;

    var reproj = reprojection(uv, p);

    {
      let uv = reproj.uv;
      let uv_p = reproj.p;
      let dp = reproj.dp;

      if ${store.reprojection.gradientErrorCorrection} {
        let _reproj = gradientReprojection(uv, p, dp);

        if _reproj.d < reproj.d {
          reproj = _reproj;
        }
      }
      
      if ${store.reprojection.doubleReprojectErrorCorrection} {
        let _reproj = reprojection(reprojectPoint(uv_p) - uv_error, p);

        if _reproj.d < reproj.d {
          reproj = _reproj;
        }
      }
    }

    return reproj;
  }

  fn reprojectFast(cuv: vec2f, p: vec3f) -> Reprojection {
    return reprojection(reprojectPoint(p), p);
  }

  // current uv, projected point, latest color for that point
  fn reprojectDirect(cuv: vec2f, p: vec3f, c: vec3f) -> Reprojection {
    layer = 0;
    let uv_error = ${store.reprojection.identityErrorCorrection ? 'unprojectPoint(p) - cuv' : 'vec2f(0)'};
    var reproj = reprojectPrecise(uv_error, p);
    for (var i = 1u; i < ${store.imageLayers}; i = i + 1u) {
      layer = i;
      let uv_error = ${store.reprojection.identityErrorCorrection ? 'unprojectPoint(p) - cuv' : 'vec2f(0)'};
      let result = reprojectPrecise(uv_error, p);
      if result.d < reproj.d {
        reproj = result;
      }
    }
    
    return reproj;
  }

  // current uv, projected point, latest color for that point
  fn reproject(p: vec3f, c: vec3f) -> Reprojection {
    layer = 0;
    var reproj = reprojectPrecise(vec2f(0), p);
    for (var i = 1u; i < ${store.imageLayers}; i = i + 1u) {
      layer = i;
      let result = reprojectPrecise(vec2f(0), p);
      if result.d < reproj.d {
        reproj = result;
      }
    }
    
    return reproj;
  }

  // current uv, projected point, latest color for that point
  fn reprojectionSample(g: Geometry, reproj: Reprojection, p: vec3f, c: vec3f) -> ReprojectionResult {
    let faceIdx = g.faceIdx;
    let objectIdx = g.objectIdx;
    layer = reproj.layer;
    
    if any(reproj.uv < vec2(0)) || any(reproj.uv > vec2(viewportf)) { // outside viewport
      return ReprojectionResult(vec4f(0));
    }

    if ${store.reprojection.distanceClamping} && !(reproj.d < threshold) {
      return ReprojectionResult(vec4f(0));
    }

    if ${store.reprojection.objectClamping} && reproj.g.objectIdx != objectIdx {
      return ReprojectionResult(vec4f(0));
    }

    if ${store.reprojection.faceClamping} && reproj.g.faceIdx != faceIdx {
      return ReprojectionResult(vec4f(0));
    }

    if ${store.reprojection.filtering === ReprojectionFiltering.Bilateral} {
      let color = bilateralFilter(reproj.uv, g, p, c);
      return ReprojectionResult(color);
    } else if ${store.reprojection.filtering === ReprojectionFiltering.Average} {
      let color = sampleImage4(reproj.uv, &prevImageBuffer, &prevGeometryBuffer);
      return ReprojectionResult(color);
    } else if ${store.reprojection.filtering === ReprojectionFiltering.ErrorWeighted} {
      let color = sampleImage4(reproj.uv, &prevImageBuffer, &prevGeometryBuffer);
      let q = pow(reproj.d, 0.15);
      let p = 1/(1+q);
      return ReprojectionResult(color * max(p, ${store.reprojection.filteringRate}));
    } else if ${store.reprojection.filtering === ReprojectionFiltering.ExponentialAverage} {
      let color = sampleImage4(reproj.uv, &prevImageBuffer, &prevGeometryBuffer);
      return ReprojectionResult(color * ${store.reprojection.filteringRate});
    } else {
      let color = sampleImage4(reproj.uv, &prevImageBuffer, &prevGeometryBuffer);
      return ReprojectionResult(color);
    }
  }

  fn reprojectFullDirect(cuv: vec2f, p: vec3f, c: vec3f) -> ReprojectionResult {
    let _layer = layer;

    let currentGeometry = geometryBuffer[geometryIdx(vec2u(cuv))];
    let reproj = reprojectDirect(cuv, p, c);
    let result = reprojectionSample(currentGeometry, reproj, p, c);

    layer = _layer;
    return result;
  }

  fn reprojectFull(g: Geometry, p: vec3f, c: vec3f) -> ReprojectionResult {
    let _layer = layer;

    let reproj = reproject(p, c);
    let result = reprojectionSample(g, reproj, p, c);

    layer = _layer;
    return result;
  }
`;

const computeColor = () => /* wgsl */ `
  fn pixelHitDist(idx: u32, ray: Ray) -> f32 {
    var hit: BVHIntersectionResult;
    hit.hit = false;
    hit.barycentric = vec3f(f32max, 0, 0);

    for (var i = 0u; i < 4; i = i + 1u) {
      let quadIdx = quad[i];
      let prevObjectIdx = prevGeometryBuffer[quadIdx].objectIdx;
      let prevFaceIdx = prevGeometryBuffer[quadIdx].faceIdx;
      if prevFaceIdx == hit.faceIdx {
        continue;
      }
      
      let _hit = objectFaceHit(prevFaceIdx, prevObjectIdx, ray, hit.barycentric.x + EPSILON);
      if _hit.hit {
        hit = _hit;
      }
    }
    
    return hit.barycentric.x + EPSILON;
  }

  fn pointColor(point: vec3f, normal: vec3f) -> vec3f {
    var color = vec3f(0);

    for (var i = 0u; i < ${store.samplesPerPoint}; i = i + 1u) {
      let s = sampleLights();
      let sMaterial = materials[s.materialIdx];
      let ds = s.point - point;
      let d_sq = dot(ds, ds);
      let d = ds * inverseSqrt(d_sq);
      let r = Ray(point, d);
      color += in_shadow(r, d_sq) * attenuation(d, normal) * sMaterial.emission / d_sq * s.p;
    }

    return color / ${store.samplesPerPoint};
  }

  fn skyboxColor(pos: vec3f, normal: vec3f) -> vec4f {
    ${(() => {
      if (store.skybox === SkyboxType.Exr && skybox()) {
        return /* wgsl */ `
          let sample = importanceSampleSkybox(random_2());
          let uv = sample.xy;
          let dir = uv_to_direction(uv);
          let ray = Ray(pos, dir);
          if !sceneAnyHit(ray, f32max) {
            let pdf = sample.z;
            let color = sampleSkyboxTexture(uv);
            return vec4f(max(0, dot(normal, dir)) * color.xyz / pdf, 1); 
          }
        `;
      } else if (store.skybox === SkyboxType.Plain) {
        return /* wgsl */ `
          let dir = normalize(vec3(1));
          let ray = Ray(pos, dir);
          if !sceneAnyHit(ray, f32max) {
            let color = vec3f(1) / ${store.exposure};
            return vec4f(max(0, dot(normal, dir)) * color.xyz, 1); 
          }
        `;
      } else {
        return '';
      }
    })()}

    return vec4f(0);
  }

  struct BounceStackEntry {
    ray: Ray,
    maxDist: f32,

    color: vec4f,
    throughput: vec3f,
  }
  const maxBounces = ${store.bouncesDepth};
  fn pixelColor(_hit: ptr<function, BVHIntersectionResult>, _ray: Ray, maxDist: f32) -> vec3f {
    var stack: array<BounceStackEntry, maxBounces>;
    var top: u32;

    top = 0;
    stack[top] = BounceStackEntry(_ray, maxDist, vec4f(0), vec3f(1));

    while (stack[top].color.w < ${store.samplesPerBounce}) {
      while (top <= maxBounces) {
        // shoot a ray out into the world
        let entry = stack[top];
        var color = select(entry.color.rgb / entry.color.w, entry.color.rgb, entry.color.w == 0);
        var throughput = entry.throughput;
        var hit: BVHIntersectionResult;

        if top == 0 {
          hit = scene(entry.ray, Interval(min_dist, entry.maxDist));

          for (var i = 0u; i < layer; i = i + 1u) {
            if !hit.hit {
              break;
            }
            
            let d = hit.barycentric.x;
            let hit1 = scene(entry.ray, Interval(d, f32max));
            let hit2 = sceneBackface(entry.ray, Interval(d, f32max));
            if !hit2.hit || hit1.barycentric.x < hit2.barycentric.x {
              hit = hit1;
            } else {
              hit = hit2;
            }
          }

          *_hit = hit;
        } else {
          hit = scene(entry.ray, Interval(min_dist, entry.maxDist));
        }

        if !hit.hit {
          stack[top].color += vec4f(sampleSkybox(entry.ray.dir) * throughput, 1);
          break;
        }

        let face = faces[hit.faceIdx];
        let material = materials[face.materialIdx];
        color += material.emission * throughput;
        throughput *= material.color;
  
        let normal = faceNormal(face, hit.barycentric.yz);
        let ray = Ray(
          facePointOffset(face, hit.barycentric.yz),
          sample_cosine_weighted_hemisphere(random_2(), 1, normal)
        );

        top++;
        stack[top] = BounceStackEntry(ray, f32max, vec4f(color, 1), throughput);
  
        {
          let color = skyboxColor(ray.pos, normal);
          if color.w == 1 {
            stack[top].color += vec4f(color.xyz * throughput, 1);
          }
        }

        // russian roulette
        {
          let p = max(throughput.x, max(throughput.y, throughput.z));
          if random_1() > p {
            break;
          }
          stack[top].throughput /= p;
        }
      }
    }

    let color = stack[top].color;
    return color.rgb / color.w;
  }
  
  fn in_shadow(ray: Ray, mag_sq: f32) -> f32 {
    return select(1., 0., sceneAnyHit(ray, sqrt(mag_sq)));
  }

  fn light_vis(pos: vec3f, dir: vec3f) -> f32 {
    return in_shadow(Ray(pos, dir), f32max);
  }

  fn attenuation(dir: vec3f, norm: vec3f) -> f32 {
    return max(dot(dir, norm), 0.);
  }
`;

const bilinearInterpolation = /* wgsl */ `
  fn bilinearInterpolation(uv: vec2f, p: vec4f) -> f32 {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }

  fn bilinearInterpolation2(uv: vec2f, p: mat4x2f) -> vec2f {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }

  fn bilinearInterpolation3(uv: vec2f, p: mat4x3f) -> vec3f {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }

  fn bilinearInterpolation4(uv: vec2f, p: mat4x4f) -> vec4f {
    let col_x = mix(p[0], p[2], uv.x);
    let col_y = mix(p[1], p[3], uv.x);
    let col = mix(col_x, col_y, uv.y);
    return col;
  }
`;

const imageSampler = /* wgsl */ `
  fn imageIdx(uv: vec2u) -> u32 {
    let x = clamp(uv.x, 0u, viewport.x - 1u);
    let y = clamp(uv.y, 0u, viewport.y - 1u);
    return x + y * viewport.x + layer * viewport.x * viewport.y;
  }

  fn sampleImage4(uv: vec2f, _image: ptr<storage, array<vec4f>, read>, buffer: ptr<storage, array<Geometry>, read_write>) -> vec4f {
    let uv_u = vec2u(floor(uv));
    let uv_f = fract(uv);
    var depths = max(vec4f(
      (*buffer)[geometryIdx(uv_u)].depth,
      (*buffer)[geometryIdx(uv_u + vec2u(1, 0))].depth,
      (*buffer)[geometryIdx(uv_u + vec2u(0, 1))].depth,
      (*buffer)[geometryIdx(uv_u + vec2u(1, 1))].depth,
    ), vec4f(1e-8));
    depths = 1 / depths;

    let m = mat4x4f(
      (*_image)[imageIdx(uv_u)] * depths[0],
      (*_image)[imageIdx(uv_u + vec2u(1, 0))] * depths[1],
      (*_image)[imageIdx(uv_u + vec2u(0, 1))] * depths[2],
      (*_image)[imageIdx(uv_u + vec2u(1, 1))] * depths[3],
    );
    let depth = 1 / bilinearInterpolation(uv_f, depths);
    let value = bilinearInterpolation4(uv_f, m) * depth;
    return value;
  }
`;

const geometrySampler = () => /* wgsl */ `
  fn geometryIdx(uv: vec2u) -> u32 {
    let x = clamp(uv.x, 0u, viewport.x - 1u);
    let y = clamp(uv.y, 0u, viewport.y - 1u);
    return x + y * viewport.x + layer * viewport.x * viewport.y;
  }

  fn sampleGeometryAll(uv: vec2f, buffer: ptr<storage, array<Geometry>, read_write>) -> Geometry {
    let uv_u = vec2u(floor(uv));
    let uv_f = fract(uv);
    let closest_uv = round(uv);
    
    let closest = (*buffer)[geometryIdx(vec2u(closest_uv))];
    var result: Geometry;
    let depths = vec4f(
      1 / (*buffer)[geometryIdx(uv_u)].depth,
      1 / (*buffer)[geometryIdx(uv_u + vec2u(1, 0))].depth,
      1 / (*buffer)[geometryIdx(uv_u + vec2u(0, 1))].depth,
      1 / (*buffer)[geometryIdx(uv_u + vec2u(1, 1))].depth,
    );
    let positions = mat4x3f(
      (*buffer)[geometryIdx(uv_u)].position * depths[0],
      (*buffer)[geometryIdx(uv_u + vec2u(1, 0))].position * depths[1],
      (*buffer)[geometryIdx(uv_u + vec2u(0, 1))].position * depths[2],
      (*buffer)[geometryIdx(uv_u + vec2u(1, 1))].position * depths[3],
    );
    result.depth = 1 / bilinearInterpolation(uv_f, depths);
    result.position = bilinearInterpolation3(uv_f, positions) * result.depth;
    result.faceIdx = closest.faceIdx;
    result.objectIdx = closest.objectIdx;
    return result;
  }

  fn sampleGeometryAllWithId(uv: vec2f, buffer: ptr<storage, array<Geometry>, read_write>, faceIdx: u32, objectIdx: u32) -> Geometry {
    let uv_u = vec2u(floor(uv));
    let uv_f = fract(uv);
    
    var result: Geometry;
    
    let p1 = buffer[geometryIdx(uv_u)];
    let p2 = buffer[geometryIdx(uv_u + vec2u(1, 0))];
    let p3 = buffer[geometryIdx(uv_u + vec2u(0, 1))];
    let p4 = buffer[geometryIdx(uv_u + vec2u(1, 1))];
    
    var p = vec4(0.);
    var n = 0u;
    if objectIdx == p1.objectIdx {
      var w = 1.;
      if faceIdx == p1.faceIdx {
        w = 2.;
      }

      p += vec4(p1.position, 1) * w * (1 - uv_f.x) * (1 - uv_f.y);
      n++;
    }
    if objectIdx == p2.objectIdx {
      var w = 1.;
      if faceIdx == p2.faceIdx {
        w = 2.;
      }
      
      p += vec4(p1.position, 1) * w * (uv_f.x) * (1 - uv_f.y);
      n++;
    }
    if objectIdx == p3.objectIdx {
      var w = 1.;
      if faceIdx == p3.faceIdx {
        w = 2.;
      }
      
      p += vec4(p1.position, 1) * w * (1 - uv_f.x) * (uv_f.y);
      n++;
    }
    if objectIdx == p4.objectIdx {
      var w = 1.;
      if faceIdx == p4.faceIdx {
        w = 2.;
      }
      
      p += vec4(p1.position, 1) * w * (uv_f.x) * (uv_f.y);
      n++;
    }

    if n > 1 {
      result.position = p.xyz / p.w;
      result.faceIdx = faceIdx;
      result.objectIdx = objectIdx;
    } else {
      result = sampleGeometryAll(uv, buffer);
    }

    return result;
  }
`;

const matInv = /* wgsl */ `
  fn inverse(m: mat4x4f) -> mat4x4f {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    return mat4x4f(
        a11 * b11 - a12 * b10 + a13 * b09,
        a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03,
        a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07,
        a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01,
        a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06,
        a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00,
        a21 * b02 - a20 * b04 - a23 * b00,
        a11 * b07 - a10 * b09 - a12 * b06,
        a00 * b09 - a01 * b07 + a02 * b06,
        a31 * b01 - a30 * b03 - a32 * b00,
        a20 * b03 - a21 * b01 + a22 * b00) * (1 / det);
  }

  // not working
  fn inverse3(m: mat3x3f) -> mat3x3f {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = 0.;
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = 0.;
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = 0.;
    let a30 = 0.; let a31 = 0.; let a32 = 0.; let a33 = 1.;

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    return mat3x3f(
        a11 * b11 - a12 * b10 + a13 * b09,
        a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03,
        // a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07,
        a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01,
        // a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06,
        a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00,
        // a21 * b02 - a20 * b04 - a23 * b00,
        // a11 * b07 - a10 * b09 - a12 * b06,
        // a00 * b09 - a01 * b07 + a02 * b06,
        // a31 * b01 - a30 * b03 - a32 * b00,
        /* a20 * b03 - a21 * b01 + a22 * b00 */) * (1 / det);
  }
`;

const COMPUTE_WORKGROUP_SIZE_X = 16;
const COMPUTE_WORKGROUP_SIZE_Y = 16;
const [computePipeline, computeBindGroups] = reactiveComputePipeline({
  shader: (x) => /* wgsl */ `
    enable subgroups;

    ${x.bindVarBuffer('storage', 'imageBuffer: array<vec4f>', imageBuffer())}
    ${x.bindVarBuffer('read-only-storage', 'prevImageBuffer: array<vec4f>', prevImageBuffer())}
    ${x.bindVarBuffer('storage', 'geometryBuffer: array<Geometry>', geometryBuffer())}
    ${x.bindVarBuffer('storage', 'prevGeometryBuffer: array<Geometry>', prevGeometryBuffer())}
    ${x.bindVarBuffer('uniform', 'viewMatrix: mat4x4f', viewBuffer)}
    ${x.bindVarBuffer('uniform', 'prevViewMatrix: mat4x4f', prevViewBuffer)}
    ${x.bindVarBuffer('uniform', 'viewInvMatrix: mat4x4f', viewInvBufferHi)}
    ${x.bindVarBuffer('uniform', 'prevViewInvMatrix: mat4x4f', prevViewInvBufferHi)}
    ${x.bindVarBuffer('uniform', 'reprojectionFrustrum: mat3x4f', reprojectionFrustrumBufferHi)}
    ${x.bindVarBuffer('uniform', 'viewInvMatrixLo: mat4x4f', viewInvBufferLo)}
    ${x.bindVarBuffer('uniform', 'prevViewInvMatrixLo: mat4x4f', prevViewInvBufferLo)}
    ${x.bindVarBuffer('uniform', 'reprojectionFrustrumLo: mat3x4f', reprojectionFrustrumBufferLo)}

    ${x.bindVarBuffer('read-only-storage', 'faces: array<Face>', facesBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'materials: array<Material>', materialsBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'models: array<Model>', modelsBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'bvh: array<BoundingVolume>', bvhBuffer)}
    ${x.bindVarBuffer('read-only-storage', 'bvhFaces: array<u32>', bvhFacesBuffer)}

    ${x.bindVarBuffer('uniform', 'seed: u32', seedUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'counter: u32', counterUniformBuffer)}
    ${x.bindVarBuffer('uniform', 'jitter: vec2f', jitterBuffer)}
    ${x.bindVarBuffer('uniform', 'prevJitter: vec2f', prevJitterBuffer)}

    const _reproject = ${store.reprojection.rate > 0};
    const viewport = vec2u(${store.view[0]}, ${store.view[1]});
    const viewportf = vec2f(viewport);
    const aspect = viewportf.y / viewportf.x;
    const viewportN = viewportf / viewportf.x; // viewport normalized

    ${double}
    ${skyboxModule(x)}
    ${tonemapping}
    ${filtering()}
    ${rng}
    ${intervals}
    ${bvh()}
    ${structs}
    ${rayIntersect}
    ${scene()}
    ${face()}
    ${raygen()}
    ${reproject()}
    ${computeColor()}
    ${bilinearInterpolation}
    ${imageSampler}
    ${geometrySampler()}
    ${matInv}
    ${derivatives()}
    ${utils}

    var<private> layer: u32;
    var<private> quadIdx: u32;
    var<private> quad: array<u32, 4>;
    var<private> quadNeighborXIdx: u32;
    var<private> quadNeighborYIdx: u32;

    @compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE_X}, ${COMPUTE_WORKGROUP_SIZE_Y})
    fn main(
      @builtin(global_invocation_id) globalInvocationId: vec3<u32>, 
      @builtin(local_invocation_index) localInvocationIndex: u32
    ) {
      let upos = globalInvocationId.xy;
      layer = globalInvocationId.z;
      let idx = imageIdx(upos);
      quadIdx = localInvocationIndex % 4;
      quad[0] = quadBroadcast(idx, 0);
      quad[1] = quadBroadcast(idx, 1);
      quad[2] = quadBroadcast(idx, 2);
      quad[3] = quadBroadcast(idx, 3);
      quadNeighborXIdx = quadSwapX(idx);
      quadNeighborYIdx = quadSwapY(idx);
      if (any(globalInvocationId.xy >= viewport)) {
        return;
      }

      let fpos = vec2f(upos);
      let pos = fpos;

      if ${store.reprojection.debug} {
        let ray = cameraRay(pos, viewMatrix);
        let hitDist = pixelHitDist(idx, ray);
        var hit: BVHIntersectionResult;
        hit = scene(ray, Interval(min_dist, hitDist));

        let dist = hit.barycentric.x;
        let point = ray.pos + ray.dir * dist;
        geometryBuffer[idx].position = point;
        geometryBuffer[idx].depth = dist;
        geometryBuffer[idx].faceIdx = hit.faceIdx;
        geometryBuffer[idx].objectIdx = hit.objectIdx;

        let result = reprojectFullDirect(fpos, point, vec3f(0));
        imageBuffer[idx] = result.color;

        return;
      }

      rng_state = seed + idx;
      if counter == 0u && !_reproject {
        imageBuffer[idx] = vec4f(0);
        geometryBuffer[idx].position = vec3f(0);
        geometryBuffer[idx].depth = 1e-8;
        geometryBuffer[idx].faceIdx = 0;
        geometryBuffer[idx].objectIdx = 0;
      }

      var color = vec3f(0);
      var samples = 0u;

      let ray = cameraRay(pos, viewMatrix);
      let hitDist = pixelHitDist(idx, ray);
      var hit: BVHIntersectionResult;
      color += pixelColor(&hit, ray, hitDist);
      samples++;

      if hit.hit {
        let dist = hit.barycentric.x;
        let point = ray.pos + ray.dir * dist;
        geometryBuffer[idx].position = point;
        geometryBuffer[idx].depth = dist;
        geometryBuffer[idx].faceIdx = hit.faceIdx;
        geometryBuffer[idx].objectIdx = hit.objectIdx;

        if _reproject {
          let result = reprojectFullDirect(fpos, point, color);
          imageBuffer[idx] = result.color;
        }
      } else {
        imageBuffer[idx] = vec4f(0);
        geometryBuffer[idx].position = vec3f(0);
        geometryBuffer[idx].depth = 1e+8;
        geometryBuffer[idx].faceIdx = 0;
        geometryBuffer[idx].objectIdx = 0;
      }


      if ${store.blitView == BlitView.Normals} {
        imageBuffer[idx] = vec4f(color, 1);
      } else {
        imageBuffer[idx] += vec4f(color, f32(samples));
      }
    }
  `,
});

const { canTimestamp, querySet, submit } = getTimestampHandler((times) => {
  setRenderGPUTime(Number(times[1] - times[0]));
});

createEffect(() => {
  const { pipeline: debugBVHPipeline, bindGroups: debugBVHBindGroup } =
    renderPipeline({
      vertexShader: (x) => /* wgsl */ `
        ${structs}
  
        ${x.bindVarBuffer('read-only-storage', 'bvh: array<BoundingVolume>', bvhBuffer)}
        ${x.bindVarBuffer('uniform', 'viewProjMatrix: mat4x4f', viewProjBuffer)}
  
        const EDGES_PER_CUBE = 12u;
  
        @vertex
        fn main(
          @builtin(instance_index) instanceIndex: u32,
          @builtin(vertex_index) vertexIndex: u32
        ) -> @builtin(position) vec4f {
          let lineInstanceIdx = instanceIndex % EDGES_PER_CUBE;
          let aabbInstanceIdx = instanceIndex / EDGES_PER_CUBE;
          let a = bvh[aabbInstanceIdx];
          let aMin = a.min;
          let aMax = a.max;
          // let aMin = vec3f(0, 0, 0);
          // let aMax = vec3f(1, 1, 1);
          var pos: vec3f;
          let fVertexIndex = f32(vertexIndex);
                        
            //      a7 _______________ a6
            //       / |             /|
            //      /  |            / |
            //  a4 /   |       a5  /  |
            //    /____|__________/   |
            //    |    |__________|___|
            //    |   / a3        |   / a2
            //    |  /            |  /
            //    | /             | /
            //    |/______________|/
            //    a0              a1
  
          let dx = aMax.x - aMin.x;
          let dy = aMax.y - aMin.y;
          let dz = aMax.z - aMin.z;
          
          let a0 = aMin;
          let a1 = vec3f(aMin.x + dx, aMin.y,      aMin.z     );
          let a2 = vec3f(aMin.x + dx, aMin.y,      aMin.z + dz);
          let a3 = vec3f(aMin.x,      aMin.y,      aMin.z + dz);
          let a4 = vec3f(aMin.x,      aMin.y + dy, aMin.z     );
          let a5 = vec3f(aMin.x + dx, aMin.y + dy, aMin.z     );
          let a6 = aMax;
          let a7 = vec3f(aMin.x,      aMin.y + dy, aMin.z + dz);
  
          if (lineInstanceIdx == 0) {
            pos = mix(a0, a1, fVertexIndex);
          } else if (lineInstanceIdx == 1) {
            pos = mix(a1, a2, fVertexIndex);
          } else if (lineInstanceIdx == 2) {
            pos = mix(a2, a3, fVertexIndex);
          } else if (lineInstanceIdx == 3) {
            pos = mix(a0, a3, fVertexIndex);
          } else if (lineInstanceIdx == 4) {
            pos = mix(a0, a4, fVertexIndex);
          } else if (lineInstanceIdx == 5) {
            pos = mix(a1, a5, fVertexIndex);
          } else if (lineInstanceIdx == 6) {
            pos = mix(a2, a6, fVertexIndex);
          } else if (lineInstanceIdx == 7) {
            pos = mix(a3, a7, fVertexIndex);
          } else if (lineInstanceIdx == 8) {
            pos = mix(a4, a5, fVertexIndex);
          } else if (lineInstanceIdx == 9) {
            pos = mix(a5, a6, fVertexIndex);
          } else if (lineInstanceIdx == 10) {
            pos = mix(a6, a7, fVertexIndex);
          } else if (lineInstanceIdx == 11) {
            pos = mix(a7, a4, fVertexIndex);
          }
          return viewProjMatrix * vec4(pos, 1);
        }
        `,
      fragmentShader: () => /* wgsl */ `
        @fragment
        fn main() -> @location(0) vec4f {
          return vec4f(0.01);
          // return vec4f(0.2); 
          // return vec4f(1);
        }
      `,
      fragmentPresentationFormatTarget: {
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
          },
        },
      },
      primitive: {
        topology: 'line-list',
      },
    });

  setDebugBVHRenderBundle(
    renderBundlePass({}, (renderPass) => {
      renderPass.setPipeline(debugBVHPipeline);
      debugBVHBindGroup.forEach((bindGroup, i) =>
        renderPass.setBindGroup(i, bindGroup)
      );
      renderPass.draw(2, bvhCount * 12);
    })
  );
});

// const geometryPass = reactiveRenderPipeline({
//   vertexShader: (x) => /* wgsl */ `
//     ${structs}
//   `,
// });

const rpd: GPURenderPassDescriptor = {
  colorAttachments: [
    {
      view: context.getCurrentTexture().createView(),
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
};

let frameCounter = 0;
export async function renderFrame(now: number) {
  const rate = store.reprojection.rate;
  const updatePrev = store.reprojection.debug
    ? rate === 0 || frameCounter % rate === 0
    : rate === 0 || frameCounter % rate === 0 || store.counter !== 0;
  frameCounter = (frameCounter + 1) % rate;
  writeUint32Buffer(seedUniformBuffer, Math.random() * 0xffffffff);
  writeUint32Buffer(counterUniformBuffer, store.counter);
  incrementCounter();

  if (updatePrev) {
    const jitter = vec2.fromValues(Math.random() - 0.5, Math.random() - 0.5);
    vec2.scale(jitter, jitter, store.jitterStrength);
    writeVec2fBuffer(jitterBuffer, jitter);
  }

  const view = viewMatrix();

  const encoder = device.createCommandEncoder();
  rpd.colorAttachments[0].view = context.getCurrentTexture().createView();

  // raytrace
  computePass(encoder, {}, (computePass) => {
    computePass.setPipeline(computePipeline());
    computeBindGroups().forEach((bindGroup, i) =>
      computePass.setBindGroup(i, bindGroup)
    );
    computePass.dispatchWorkgroups(
      Math.ceil(canvas.width / COMPUTE_WORKGROUP_SIZE_X),
      Math.ceil(canvas.height / COMPUTE_WORKGROUP_SIZE_Y),
      store.imageLayers
    );
  });

  renderPass(encoder, rpd, (renderPass) => {
    renderPass.executeBundles([blitRenderBundle()]);

    // debug BVH
    if (store.debugBVH) {
      renderPass.executeBundles([debugBVHRenderBundle()]);
    }
  });

  if (updatePrev) {
    encoder.copyBufferToBuffer(jitterBuffer, prevJitterBuffer);
    encoder.copyBufferToBuffer(imageBuffer(), prevImageBuffer());
    encoder.copyBufferToBuffer(geometryBuffer(), prevGeometryBuffer());
  }

  await submit(encoder, () => {
    device.queue.submit([encoder.finish()]);
  });

  await device.queue.onSubmittedWorkDone();
  setRenderJSTime(performance.now() - now);
  if (updatePrev) {
    setPrevView(view);
  }
}

import { vec3, vec4 } from 'gl-matrix';
import wavefrontObjParser from 'obj-file-parser';
import { createStorageBuffer, createTexture } from './gpu';
import { Iterator } from 'iterator-js';
import { BoundingVolume, BoundingVolumeHierarchy, facesBVH } from './bv';
import { triangleModel, unitCubeModel } from './testModels';
import MTLFile from './mtl';
import { makeStructuredView } from 'webgpu-utils';
import parseExr from 'parse-exr';
import parseHdr from 'parse-hdr';
import { preprocess as preprocessSkybox } from './skybox';
import { store } from './store';
import { defs } from './shaders/structs';

type Point = {
  position: vec3;
  normal: vec3;
  texture: vec3;
};
type OrientedLine = {
  dir: vec3;
  cross: vec3;
};
export type Face = {
  points: [Point, Point, Point];
  plane: vec4;
  uPlane: vec4;
  vPlane: vec4;
  e1: OrientedLine;
  e2: OrientedLine;
  e3: OrientedLine;
  materialIdx: number;
  area: number;
  idx: number;
};
export type Model = {
  name: string;
  faces: Face[];
  bvh: BoundingVolumeHierarchy;
};
export type Material = {
  name: string;
  color: vec3;
  emission: vec3;
};
type ObjVertex = { x: number; y: number; z: number };
type ObjTexture = { u: number; v: number; w: number };

const objVertToVec3 = (v: ObjVertex) => vec3.fromValues(v.x, v.y, v.z);
const objTexToVec3 = (v: ObjTexture) => vec3.fromValues(v.u, v.v, v.w);

type Allocation = { offset: number; count: number };
const faceSize = defs.structs.Face.size / Float32Array.BYTES_PER_ELEMENT;
const bvSize =
  defs.structs.BoundingVolume.size / Float32Array.BYTES_PER_ELEMENT;
const modelSize = defs.structs.Model.size / Float32Array.BYTES_PER_ELEMENT;
const materialSize =
  defs.structs.Material.size / Float32Array.BYTES_PER_ELEMENT;
// face offsets and counts are in faceSize units
const facesAllocations: Allocation[] = [];
// bvh offsets and counts are in bvSize units
const bvhAllocations: Allocation[] = [];
const bvhFacesAllocations: Allocation[] = [];

const allocate = (allocations: Allocation[], count: number) => {
  const lastAllocation = allocations[allocations.length - 1];
  const offset = lastAllocation
    ? lastAllocation.offset + lastAllocation.count
    : 0;
  allocations.push({ offset, count });
  return offset;
};

const allocateFace = (count: number) => allocate(facesAllocations, count);
const allocateBVH = (count: number) => allocate(bvhAllocations, count);
const allocateBVHFace = (count: number) => allocate(bvhFacesAllocations, count);
const totalAllocationSize = (allocations: Allocation[]) => {
  const lastAllocation = allocations[allocations.length - 1];
  return lastAllocation ? lastAllocation.offset + lastAllocation.count : 0;
};

export const createFace = (desc: {
  idx: number;
  materialIdx: number;
  p0: vec3;
  p1: vec3;
  p2: vec3;
  points: [Point, Point, Point];
}): Face => {
  const { points, materialIdx, idx, p0, p1, p2 } = desc;

  const l1 = {
    dir: vec3.sub(vec3.create(), p1, p0),
    cross: vec3.cross(vec3.create(), p1, p0),
  };
  const l2 = {
    dir: vec3.sub(vec3.create(), p2, p1),
    cross: vec3.cross(vec3.create(), p2, p1),
  };
  const l3 = {
    dir: vec3.sub(vec3.create(), p0, p2),
    cross: vec3.cross(vec3.create(), p0, p2),
  };

  const e1 = l1.dir;
  const e2 = vec3.negate(vec3.create(), l3.dir);

  const normal = vec3.cross(vec3.create(), e2, e1);
  const area = vec3.length(normal) / 2;
  vec3.normalize(normal, normal);
  const normalD = vec3.dot(p0, normal);

  const uNormal = vec3.cross(vec3.create(), e2, normal);
  vec3.normalize(uNormal, uNormal);
  const uNormalD = -vec3.dot(p0, uNormal);

  const vNormal = vec3.cross(vec3.create(), normal, e1);
  vec3.normalize(vNormal, vNormal);
  const vNormalD = -vec3.dot(p0, vNormal);

  return {
    points,
    plane: vec4.fromValues(normal[0], normal[1], normal[2], normalD),
    uPlane: vec4.fromValues(uNormal[0], uNormal[1], uNormal[2], uNormalD),
    vPlane: vec4.fromValues(vNormal[0], vNormal[1], vNormal[2], vNormalD),
    e1: l1,
    e2: l2,
    e3: l3,
    materialIdx,
    area,
    idx,
  };
};

export const loadModels = async () => {
  const objFile = await import('@assets/raytraced-scene.obj?raw');
  const objParser = new wavefrontObjParser(objFile.default);
  const objParsed = objParser.parse();

  const mtlFile = await import('@assets/raytraced-scene.mtl?raw');
  const mtlParser = new MTLFile(mtlFile.default);
  const mtlParsed = mtlParser.parse();

  const materials = mtlParsed.map((mtl): Material => {
    const { Kd, Ke } = mtl;
    console.log(mtl);

    if (mtl.name === 'Light') {
      const emission = vec3.fromValues(1, 1, 1);
      vec3.scale(emission, emission, 1 / store.exposure);
      return {
        color: vec3.fromValues(0, 0, 0),
        emission: emission,
        name: mtl.name,
      };
    }

    return {
      color: vec3.fromValues(Kd.red, Kd.green, Kd.blue),
      emission: vec3.fromValues(Ke.red, Ke.green, Ke.blue),
      name: mtl.name,
    };
  });

  let posArray: vec3[] = [];
  let nrmArray: vec3[] = [];
  let uvArray: vec3[] = [];
  const models: Model[] = [];

  models.push(unitCubeModel());
  models.push(triangleModel());

  // return modelsCache.map((_, i) => i);

  objParsed.models.forEach(
    ({ vertices, vertexNormals, textureCoords, faces, name }, i) => {
      console.log(name, i, faces[0].material);

      posArray = posArray.concat(vertices.map(objVertToVec3));
      nrmArray = nrmArray.concat(vertexNormals.map(objVertToVec3));
      uvArray = uvArray.concat(textureCoords.map(objTexToVec3));

      const _faces = faces
        .map((f): Face => {
          const i0 = f.vertices[0].vertexIndex - 1;
          const i1 = f.vertices[1].vertexIndex - 1;
          const i2 = f.vertices[2].vertexIndex - 1;

          const j0 = f.vertices[0].vertexNormalIndex - 1;
          const j1 = f.vertices[1].vertexNormalIndex - 1;
          const j2 = f.vertices[2].vertexNormalIndex - 1;
          const k0 = f.vertices[0].textureCoordsIndex - 1;
          const k1 = f.vertices[1].textureCoordsIndex - 1;
          const k2 = f.vertices[2].textureCoordsIndex - 1;

          const p0 = posArray[i0];
          const p1 = posArray[i1];
          const p2 = posArray[i2];

          const e1 = vec3.sub(vec3.create(), p1, p0);
          const e2 = vec3.sub(vec3.create(), p2, p0);

          const materialIdx = materials.findIndex(
            ({ name }) => name === f.material
          );

          return createFace({
            idx: i,
            materialIdx,
            p0,
            p1,
            p2,
            points: [
              { position: p0, normal: nrmArray[j0], texture: uvArray[k0] },
              { position: p1, normal: nrmArray[j1], texture: uvArray[k1] },
              { position: p2, normal: nrmArray[j2], texture: uvArray[k2] },
            ],
          });
        })
        .map((face, i) => ({ ...face, idx: i }));

      const bvh = facesBVH(_faces);

      models.push({ name, faces: _faces, bvh });
    }
  );

  return { models, materials };
};

const loadModelFacesToBuffer = async (
  _mapped: ArrayBuffer,
  model: Model,
  offset: number
) => {
  for (const [face, i] of Iterator.iter(model.faces).enumerate()) {
    const f32Offset =
      offset * Float32Array.BYTES_PER_ELEMENT + i * defs.structs.Face.size;
    const values = makeStructuredView(defs.structs.Face, _mapped, f32Offset);

    values.set({
      ...face,
      points: Iterator.iter(face.points)
        .map((p) => ({
          pos: p.position,
          normal: p.normal,
        }))
        .toArray(),
    });
  }
};

const loadModelData = async (mapped: ArrayBuffer) => {
  for (const [[faces, bvh], i] of Iterator.iter(facesAllocations)
    .zip(bvhAllocations)
    .enumerate()) {
    const f32Offset = modelSize * i * Float32Array.BYTES_PER_ELEMENT;
    const values = makeStructuredView(defs.structs.Model, mapped, f32Offset);
    values.set({ faces, bvh });
  }
};

const loadBVToBuffer = (
  mapped: ArrayBuffer,
  bv: BoundingVolume,
  offset: number
) => {
  const f32Offset = offset * Float32Array.BYTES_PER_ELEMENT;
  const values = makeStructuredView(
    defs.structs.BoundingVolume,
    mapped,
    f32Offset
  );
  const facesCount = bv.faces.length;
  const facesOffset = allocateBVHFace(facesCount);

  values.set({
    min: bv.min,
    max: bv.max,
    rightIdx: bv.rightIdx,
    facesCount: facesCount,
    facesOffset: facesOffset,
  });
};

const loadBVH = async (
  mapped: ArrayBuffer,
  model: Model,
  offset: number,
  bvhFaces: number[][]
) => {
  for (const [bv, i] of Iterator.iter(model.bvh).enumerate()) {
    let idx = offset + bvSize * i;
    loadBVToBuffer(mapped, bv, idx);
    bvhFaces.push(bv.faces);
  }
};

const loadMaterialToBuffer = (
  mapped: ArrayBuffer,
  material: Material,
  offset: number
) => {
  const f32Offset = offset * Float32Array.BYTES_PER_ELEMENT;
  const values = makeStructuredView(defs.structs.Material, mapped, f32Offset);

  values.set({ color: material.color, emission: material.emission });
};

export const loadMaterialsToBuffers = async (materials: Material[]) => {
  const materialsBuffer = createStorageBuffer(
    materials.length * materialSize * Float32Array.BYTES_PER_ELEMENT,
    'Materials Buffer',
    0,
    true
  );
  const materialsMapped = materialsBuffer.getMappedRange();

  for (const [material, i] of Iterator.iter(materials).enumerate()) {
    loadMaterialToBuffer(materialsMapped, material, i * materialSize);
  }

  materialsBuffer.unmap();

  return { materialsBuffer };
};

export const loadModelsToBuffers = async (models: Model[]) => {
  const facesCount = Iterator.iter(models).sum((m) => m.faces.length);
  const facesBuffer = createStorageBuffer(
    facesCount * faceSize * Float32Array.BYTES_PER_ELEMENT,
    'Faces Buffer',
    0,
    true
  );
  const facesMapped = facesBuffer.getMappedRange();

  for (const model of models) {
    const offset = allocateFace(model.faces.length);
    await loadModelFacesToBuffer(facesMapped, model, offset * faceSize);
  }

  facesBuffer.unmap();

  const bvhCount = Iterator.iter(models).sum((m) => m.bvh.length);
  const bvhBuffer = createStorageBuffer(
    bvhCount * bvSize * Float32Array.BYTES_PER_ELEMENT,
    'BVH Buffer',
    0,
    true
  );
  const bvhMapped = bvhBuffer.getMappedRange();

  const bvhFaces: number[][] = [];

  for (const model of models) {
    const offset = allocateBVH(model.bvh.length);
    await loadBVH(bvhMapped, model, offset * bvSize, bvhFaces);
  }

  bvhBuffer.unmap();

  const bvhFacesCount = totalAllocationSize(bvhFacesAllocations);
  const bvhFacesBuffer = createStorageBuffer(
    bvhFacesCount * Uint32Array.BYTES_PER_ELEMENT,
    'BVH Faces Buffer',
    0,
    true
  );
  const bvhFacesMapped = bvhFacesBuffer.getMappedRange();
  const bvhFacesU32Mapped = new Uint32Array(bvhFacesMapped);

  for (const [faces, i] of Iterator.iter(bvhFaces).enumerate()) {
    bvhFacesU32Mapped.set(faces, bvhFacesAllocations[i].offset);
  }

  bvhFacesBuffer.unmap();

  const modelsBuffer = createStorageBuffer(
    models.length * modelSize * Uint32Array.BYTES_PER_ELEMENT,
    'Models Buffer',
    0,
    true
  );

  await loadModelData(modelsBuffer.getMappedRange());

  modelsBuffer.unmap();

  return { facesBuffer, bvhBuffer, bvhFacesBuffer, bvhCount, modelsBuffer };
};

const loadExr = async (url: string) => {
  const exrData = await (await fetch(url)).arrayBuffer();
  const FloatType = 1015;
  // const HalfFloatType = 1016;
  return parseExr(exrData, FloatType);
};

const loadHdr = async (url: string) => {
  const hdrData = await (await fetch(url)).arrayBuffer();
  return parseHdr(hdrData);
};

export const loadSkybox = async () => {
  const url = await import('@assets/qwantani_afternoon_puresky_4k.exr?url');
  const data = await loadExr(url.default);

  console.log('preprocessing skybox');

  const importanceSampleBuffer = await preprocessSkybox(data);

  const texture = createTexture(
    {
      data: data.data as Float32Array,
      width: data.width,
      height: data.height,
    },
    {
      format: 'rgba32float',
      colorSpace: 'srgb',
      viewDimension: '2d',
      dimension: '2d',
    }
  );

  const importanceSampleTexture = createTexture(
    {
      data: importanceSampleBuffer,
      width: data.width,
      height: data.height,
    },
    {
      format: 'rgba32float',
      colorSpace: 'srgb',
      viewDimension: '2d',
      dimension: '2d',
    }
  );

  console.log('skybox done');

  return { texture, importanceSampleTexture };
};

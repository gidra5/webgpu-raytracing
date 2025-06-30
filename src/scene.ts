import { vec3 } from 'gl-matrix';
import wavefrontObjParser from 'obj-file-parser';
import { createStorageBuffer } from './gpu';
import { Iterator } from 'iterator-js';
import { BoundingVolumeHierarchy, facesBVH } from './bv';
import { triangleModel, unitCubeModel } from './testModels';
import MTLFile from './mtl';
import { makeShaderDataDefinitions, makeStructuredView } from 'webgpu-utils';

type Point = {
  position: vec3;
  normal: vec3;
  texture: vec3;
};
export type Face = {
  points: [Point, Point, Point];
  normal: vec3;
  materialIdx: number;
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
const facePointSize = 8;
const faceSize = 4 + 3 * facePointSize;
const bvSize = 12;
const modelSize = 4;
const materialSize = 8;
// face offsets and counts are in faceSize units
const facesAllocations: Allocation[] = [];
// bvh offsets and counts are in bvSize units
const bvhAllocations: Allocation[] = [];

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

const backface = (face: Face): Face => {
  const p0 = face.points[0].position;
  const e1 = face.points[1].position;
  const e2 = face.points[2].position;
  const n1 = vec3.negate(vec3.create(), face.points[0].normal);
  const n2 = vec3.negate(vec3.create(), face.points[1].normal);
  const n3 = vec3.negate(vec3.create(), face.points[2].normal);
  const normal = vec3.negate(vec3.create(), face.normal);

  return {
    materialIdx: face.materialIdx,
    idx: 0,
    normal,
    points: [
      { position: p0, normal: n1, texture: face.points[0].texture },
      { position: e2, normal: n3, texture: face.points[2].texture },
      { position: e1, normal: n2, texture: face.points[1].texture },
    ],
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

    if (mtl.name === 'Light')
      return {
        color: vec3.fromValues(0, 0, 0),
        emission: vec3.fromValues(1, 1, 1),
        name: mtl.name,
      };

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

  models.push(unitCubeModel);
  models.push(triangleModel);

  // return modelsCache.map((_, i) => i);

  objParsed.models.forEach(
    ({ vertices, vertexNormals, textureCoords, faces, name }, i) => {
      console.log(name, i, faces[0].material);

      posArray = posArray.concat(vertices.map(objVertToVec3));
      nrmArray = nrmArray.concat(vertexNormals.map(objVertToVec3));
      uvArray = uvArray.concat(textureCoords.map(objTexToVec3));

      const _faces = faces
        .map((f): Face[] => {
          const i0 = f.vertices[0].vertexIndex - 1;
          const i1 = f.vertices[1].vertexIndex - 1;
          const i2 = f.vertices[2].vertexIndex - 1;
          const p0 = posArray[i0];
          const p1 = posArray[i1];
          const p2 = posArray[i2];

          const j0 = f.vertices[0].vertexNormalIndex - 1;
          const j1 = f.vertices[1].vertexNormalIndex - 1;
          const j2 = f.vertices[2].vertexNormalIndex - 1;
          const k1 = f.vertices[0].textureCoordsIndex - 1;
          const k2 = f.vertices[1].textureCoordsIndex - 1;
          const k3 = f.vertices[2].textureCoordsIndex - 1;

          const e1 = vec3.create();
          const e2 = vec3.create();
          vec3.sub(e1, p1, p0);
          vec3.sub(e2, p2, p0);

          const normal = vec3.create();
          vec3.cross(normal, e1, e2);
          vec3.normalize(normal, normal);
          const materialIdx = materials.findIndex(
            ({ name }) => name === f.material
          );
          const face: Face = {
            materialIdx,
            normal,
            idx: 0,
            points: [
              { position: p0, normal: nrmArray[j0], texture: uvArray[k1] },
              { position: e1, normal: nrmArray[j1], texture: uvArray[k2] },
              { position: e2, normal: nrmArray[j2], texture: uvArray[k3] },
            ],
          };
          return [face, backface(face)];
        })
        .flat()
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
  // fuck alignment
  // https://www.w3.org/TR/WGSL/#alignment-and-size
  const mappedF32 = new Float32Array(_mapped);
  const mappedU32 = new Uint32Array(_mapped);
  for (const [face, i] of Iterator.iter(model.faces).enumerate()) {
    const { points, normal } = face;
    const i2 = offset + i * faceSize;
    mappedF32[i2 + 0] = normal[0];
    mappedF32[i2 + 1] = normal[1];
    mappedF32[i2 + 2] = normal[2];
    mappedU32[i2 + 3] = face.materialIdx;

    for (const [point, j] of Iterator.iter(points).enumerate()) {
      const { position, normal } = point;
      const k = i2 + 4 + j * facePointSize;
      mappedF32[k + 0] = position[0];
      mappedF32[k + 1] = position[1];
      mappedF32[k + 2] = position[2];
      /* padding */
      mappedF32[k + 4] = normal[0];
      mappedF32[k + 5] = normal[1];
      mappedF32[k + 6] = normal[2];
      /* padding */
    }
  }
};

const loadModelData = async (mapped: ArrayBuffer) => {
  const mappedU32 = new Uint32Array(mapped);
  for (const [[faces, bvh], i] of Iterator.iter(facesAllocations)
    .zip(bvhAllocations)
    .enumerate()) {
    mappedU32[modelSize * i + 0] = faces.offset;
    mappedU32[modelSize * i + 1] = faces.count;
    mappedU32[modelSize * i + 2] = bvh.offset;
    mappedU32[modelSize * i + 3] = bvh.count;
  }
};

const loadBVH = async (mapped: ArrayBuffer, model: Model, offset: number) => {
  const mappedF32 = new Float32Array(mapped);
  const mappedI32 = new Int32Array(mapped);

  for (const [bv, i] of Iterator.iter(model.bvh).enumerate()) {
    let idx = offset + bvSize * i;
    mappedF32[idx + 0] = bv.min[0];
    mappedF32[idx + 1] = bv.min[1];
    mappedF32[idx + 2] = bv.min[2];
    mappedI32[idx + 3] = bv.rightIdx;
    mappedF32[idx + 4] = bv.max[0];
    mappedF32[idx + 5] = bv.max[1];
    mappedF32[idx + 6] = bv.max[2];
    mappedI32[idx + 7] = bv.faces[0];
    mappedI32[idx + 8] = bv.faces[1];
  }
};

const loadMaterialToBuffer = async (
  mapped: ArrayBuffer,
  material: Material,
  offset: number
) => {
  const f32Offset = offset * Float32Array.BYTES_PER_ELEMENT;
  const code = `
    struct Material {
      color: vec3f,
      emission: vec3f
    };
  `;
  const defs = makeShaderDataDefinitions(code);
  const values = makeStructuredView(defs.structs.Material, mapped, f32Offset);

  values.set({
    color: material.color,
    emission: material.emission,
  });

  // console.log(
  //   material.name,
  //   new Float32Array(
  //     mapped,
  //     offset * Float32Array.BYTES_PER_ELEMENT,
  //     materialSize
  //   )
  // );
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
    await loadMaterialToBuffer(materialsMapped, material, i * materialSize);
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

  for (const model of models) {
    const offset = allocateBVH(model.bvh.length);
    await loadBVH(bvhMapped, model, offset * bvSize);
  }

  bvhBuffer.unmap();

  const modelsBuffer = createStorageBuffer(
    models.length * modelSize * Uint32Array.BYTES_PER_ELEMENT,
    'Models Buffer',
    0,
    true
  );

  await loadModelData(modelsBuffer.getMappedRange());

  modelsBuffer.unmap();

  return { facesBuffer, bvhBuffer, bvhCount, modelsBuffer };
};

import { vec3 } from 'gl-matrix';
import wavefrontObjParser from 'obj-file-parser';
import { createStorageBuffer } from './gpu';
import { Iterator } from 'iterator-js';
import { BoundingVolumeHierarchy, facesBVH } from './bv';
import { triangleModel, unitCubeModel } from './testModels';

type Point = {
  position: vec3;
  normal: vec3;
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
type ObjVector = { x: number; y: number; z: number };

const objVecToVec3 = (v: ObjVector) => vec3.fromValues(v.x, v.y, v.z);

type Allocation = { offset: number; count: number };
const facePointSize = 8;
const faceSize = 4 + 3 * facePointSize;
const bvSize = 12;
const modelSize = 4;
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

export const loadModels = async (): Promise<Model[]> => {
  const module = await import('@assets/raytraced-scene.obj?raw');
  const objParser = new wavefrontObjParser(module.default);
  const objFile = objParser.parse();

  let posArray: ObjVector[] = [];
  let nrmArray: ObjVector[] = [];
  const modelsCache: Model[] = [];

  modelsCache.push(unitCubeModel);
  modelsCache.push(triangleModel);

  // return modelsCache.map((_, i) => i);

  objFile.models.forEach(({ vertices, vertexNormals, faces, name }, i) => {
    console.log(name, i);

    posArray = posArray.concat(vertices);
    nrmArray = nrmArray.concat(vertexNormals);

    const _faces = faces
      .map((f, i): Face[] => {
        const i0 = f.vertices[0].vertexIndex - 1;
        const i1 = f.vertices[1].vertexIndex - 1;
        const i2 = f.vertices[2].vertexIndex - 1;
        const p0 = objVecToVec3(posArray[i0]);
        const p1 = objVecToVec3(posArray[i1]);
        const p2 = objVecToVec3(posArray[i2]);

        const j0 = f.vertices[0].vertexNormalIndex - 1;
        const j1 = f.vertices[1].vertexNormalIndex - 1;
        const j2 = f.vertices[2].vertexNormalIndex - 1;
        const n0 = objVecToVec3(nrmArray[j0]);
        const n1 = objVecToVec3(nrmArray[j1]);
        const n2 = objVecToVec3(nrmArray[j2]);

        const e1 = vec3.create();
        const e2 = vec3.create();
        vec3.sub(e1, p1, p0);
        vec3.sub(e2, p2, p0);

        const normal = vec3.create();
        vec3.cross(normal, e1, e2);
        vec3.normalize(normal, normal);
        return [
          {
            materialIdx: 0,
            normal,
            idx: i,
            points: [
              { position: p0, normal: n0 },
              { position: e1, normal: n1 },
              { position: e2, normal: n2 },
            ],
          },
          // also add backfaces
          // {
          //   materialIdx: 0,
          //   normal,
          //   idx: i,
          //   points: [
          //     { position: p0, normal: n0 },
          //     { position: e2, normal: n2 },
          //     { position: e1, normal: n1 },
          //   ],
          // },
        ];
      })
      .flat();

    const bvh = facesBVH(_faces);

    modelsCache.push({ name, faces: _faces, bvh });
  });

  return modelsCache;
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
    /* padding */
    mappedF32[idx + 4] = bv.max[0];
    mappedF32[idx + 5] = bv.max[1];
    mappedF32[idx + 6] = bv.max[2];
    mappedI32[idx + 7] = bv.leftIdx;
    mappedI32[idx + 8] = bv.rightIdx;
    mappedI32[idx + 9] = bv.faces[0];
    mappedI32[idx + 10] = bv.faces[1];
    /* padding */
  }
};

export const loadModelsToBuffers = async (models: Model[]) => {
  const facesCount = Iterator.iter(models).sum((m) => m.faces.length);
  const facesBuffer = createStorageBuffer(
    facesCount * faceSize * Float32Array.BYTES_PER_ELEMENT,
    'Faces Buffer',
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
    'Faces Offsets Buffer',
    true
  );

  await loadModelData(modelsBuffer.getMappedRange());

  modelsBuffer.unmap();

  return { facesBuffer, bvhBuffer, bvhCount, modelsBuffer };
};

import { vec3 } from 'gl-matrix';
import wavefrontObjParser from 'obj-file-parser';
import { createStorageBuffer } from './gpu';
import { Iterator } from 'iterator-js';

type Point = {
  position: vec3;
  normal: vec3;
};
type Face = {
  points: [Point, Point, Point];
  normal: vec3;
  materialIdx: number;
};
type Model = {
  name: string;
  faces: Face[];
};
type ObjVector = { x: number; y: number; z: number };

const objVecToVec3 = (v: ObjVector) => vec3.fromValues(v.x, v.y, v.z);

export const loadModel = async (): Promise<Model[]> => {
  const module = await import('@assets/raytraced-scene.obj?raw');
  const objParser = new wavefrontObjParser(module.default);
  const objFile = objParser.parse();

  let posArray: ObjVector[] = [];
  let nrmArray: ObjVector[] = [];

  return objFile.models.map(({ vertices, vertexNormals, faces, name }) => {
    posArray = posArray.concat(vertices);
    nrmArray = nrmArray.concat(vertexNormals);

    const _faces = faces.map((f): Face => {
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
      return {
        materialIdx: 0,
        normal,
        points: [
          { position: p0, normal: n0 },
          { position: e1, normal: n1 },
          { position: e2, normal: n2 },
        ],
      };
    });

    return { name, faces: _faces };
  });
};

export const loadModelToBuffer = async (model: Model): Promise<GPUBuffer> => {
  // fuck alignment
  // https://www.w3.org/TR/WGSL/#alignment-and-size
  const facePointSize = 6;
  const faceSize = 4 + 3 * facePointSize + 2;
  const buffer = createStorageBuffer(
    model.faces.length * faceSize * Float32Array.BYTES_PER_ELEMENT,
    'Faces Buffer',
    true
  );

  const _mapped = buffer.getMappedRange();
  const mappedF32 = new Float32Array(_mapped);
  const mappedU32 = new Uint32Array(_mapped);
  for (const [face, i] of Iterator.iter(model.faces).enumerate()) {
    const { points, normal } = face;
    const i2 = i * faceSize;
    mappedF32[i2 + 0] = normal[0];
    mappedF32[i2 + 1] = normal[1];
    mappedF32[i2 + 2] = normal[2];
    mappedU32[i2 + 3] = face.materialIdx;

    for (const [point, j] of Iterator.iter(points).enumerate()) {
      const { position, normal } = point;
      const k = i2 + 4 + j * facePointSize;
      mappedF32[k + 0] = position[0];
      mappedF32[k + 1] = normal[0];
      mappedF32[k + 2] = position[1];
      mappedF32[k + 3] = normal[1];
      mappedF32[k + 4] = position[2];
      mappedF32[k + 5] = normal[2];
    }
  }
  buffer.unmap();
  return buffer;
};

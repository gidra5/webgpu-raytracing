import { mat4, vec3 } from 'gl-matrix';
import { facesBVH } from './bv';
import { Face, Model } from './scene';

const makeModel = (
  name: string,
  vertices: vec3[],
  indices: [number, number, number][],
  modelMatrix: mat4
): Model => {
  vertices = vertices.map((v) => vec3.transformMat4(v, v, modelMatrix));

  const faces: Face[] = indices.map(([a, b, c], i): Face => {
    const p0 = vertices[a];
    const p1 = vertices[b];
    const p2 = vertices[c];

    const e1 = vec3.create();
    const e2 = vec3.create();
    vec3.sub(e1, p1, p0);
    vec3.sub(e2, p2, p0);

    const normal = vec3.create();
    vec3.cross(normal, e1, e2);
    vec3.normalize(normal, normal);
    return {
      points: [
        { position: p0, normal, texture: vec3.create() },
        { position: e1, normal, texture: vec3.create() },
        { position: e2, normal, texture: vec3.create() },
      ],
      normal,
      materialIdx: 0,
      idx: i,
    };
  });

  return { name, faces, bvh: facesBVH(faces) };
};

const cubeModelMatrix = mat4.create();
mat4.translate(cubeModelMatrix, cubeModelMatrix, vec3.fromValues(0, 0, -4));
mat4.scale(cubeModelMatrix, cubeModelMatrix, vec3.fromValues(0.5, 0.5, 0.5));

const unitCubeVertices: vec3[] = [
  vec3.fromValues(1, 1, 1),
  vec3.fromValues(-1, 1, 1),
  vec3.fromValues(-1, -1, 1),
  vec3.fromValues(1, -1, 1),
  vec3.fromValues(1, 1, -1),
  vec3.fromValues(-1, 1, -1),
  vec3.fromValues(-1, -1, -1),
  vec3.fromValues(1, -1, -1),
].map((v) => vec3.transformMat4(v, v, cubeModelMatrix));

const unitCubeIndices: [number, number, number][] = [
  [0, 1, 2],
  [2, 3, 0],
  [5, 4, 6],
  [7, 6, 4],
  [0, 4, 1],
  [5, 1, 4],
  [6, 2, 5],
  [5, 2, 1],
  [7, 3, 6],
  [6, 3, 2],
  [0, 3, 7],
  [7, 4, 0],
];

export const unitCubeModel = makeModel(
  'unitCube',
  unitCubeVertices,
  unitCubeIndices,
  cubeModelMatrix
);

// console.log(unitCubeModel);

const triangleModelMatrix = mat4.create();
mat4.translate(
  triangleModelMatrix,
  triangleModelMatrix,
  vec3.fromValues(-0.5, -0.5, -2)
);

export const triangleModel = makeModel(
  'triangle',
  [
    vec3.fromValues(0, 0, 0),
    vec3.fromValues(1, 0, 0),
    vec3.fromValues(0, 1, 0),
  ],
  [[0, 1, 2]],
  triangleModelMatrix
);

// console.log(triangleModel);

import { mat4, vec3 } from 'gl-matrix';
import { facesBVH } from './bv';
import { createFace, Face, Model } from './scene';

const makeModel = (
  name: string,
  vertices: vec3[],
  indices: [number, number, number][],
  modelMatrix: mat4
): Model => {
  vertices = vertices.map((v) => vec3.transformMat4(v, v, modelMatrix));

  const normals = [];
  const uvs = [];

  const faces: Face[] = indices.map(([a, b, c], i): Face => {
    const p0 = vertices[a];
    const p1 = vertices[b];
    const p2 = vertices[c];

    const e1 = vec3.create();
    const e2 = vec3.create();
    vec3.sub(e1, p1, p0);
    vec3.sub(e2, p2, p0);

    const normal = vec3.cross(vec3.create(), e1, e2);
    vec3.normalize(normal, normal);
    const normalsIdx = normals.push(normal) - 1;

    const textureIdx = uvs.push(vec3.create()) - 1;

    return createFace({
      idx: i,
      materialIdx: 0,
      p0,
      p1,
      p2,
      points: [
        { position: a, normal: normalsIdx, texture: textureIdx },
        { position: b, normal: normalsIdx, texture: textureIdx },
        { position: c, normal: normalsIdx, texture: textureIdx },
      ],
    });
  });

  return {
    name,
    faces,
    bvh: facesBVH(faces, vertices),
    normals,
    vertices,
    uvs,
  };
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

export const unitCubeModel = () =>
  makeModel('unitCube', unitCubeVertices, unitCubeIndices, cubeModelMatrix);

const triangleModelMatrix = mat4.create();
mat4.translate(
  triangleModelMatrix,
  triangleModelMatrix,
  vec3.fromValues(-0.5, -0.5, -2)
);

export const triangleModel = () =>
  makeModel(
    'triangle',
    [
      vec3.fromValues(0, 0, 0),
      vec3.fromValues(1, 0, 0),
      vec3.fromValues(0, 1, 0),
    ],
    [[0, 1, 2]],
    triangleModelMatrix
  );

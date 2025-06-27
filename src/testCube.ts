import { mat4, vec3 } from 'gl-matrix';
import { facesBVH } from './bv';
import { Face, Model } from './scene';

const modelMatrix = mat4.create();
mat4.translate(modelMatrix, modelMatrix, vec3.fromValues(0, 0, 4));
mat4.scale(modelMatrix, modelMatrix, vec3.fromValues(0.5, 0.5, 0.5));

console.log(modelMatrix);

const unitCubeVertices: vec3[] = [
  vec3.fromValues(1, 1, 1),
  vec3.fromValues(-1, 1, 1),
  vec3.fromValues(-1, -1, 1),
  vec3.fromValues(1, -1, 1),
  vec3.fromValues(1, 1, -1),
  vec3.fromValues(-1, 1, -1),
  vec3.fromValues(-1, -1, -1),
  vec3.fromValues(1, -1, -1),
  // ];
].map((v) => vec3.transformMat4(v, v, modelMatrix));

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

const unitCubeFaces: Face[] = unitCubeIndices.map(([a, b, c], i): Face => {
  const p0 = unitCubeVertices[a];
  const p1 = unitCubeVertices[b];
  const p2 = unitCubeVertices[c];

  const e1 = vec3.create();
  const e2 = vec3.create();
  vec3.sub(e1, p1, p0);
  vec3.sub(e2, p2, p0);

  const normal = vec3.create();
  vec3.cross(normal, e1, e2);
  vec3.normalize(normal, normal);
  return {
    points: [
      { position: p0, normal },
      { position: e1, normal },
      { position: e2, normal },
    ],
    normal,
    materialIdx: 0,
    idx: i,
  };
});

export const unitCubeModel: Model = {
  name: 'unitCube',
  faces: unitCubeFaces,
  bvh: facesBVH(unitCubeFaces),
};

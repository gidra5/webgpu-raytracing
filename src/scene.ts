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

export const loadModel = async (): Promise<Model[]> => {
  const module = await import('@assets/raytraced-scene.obj?raw');
  const objParser = new wavefrontObjParser(module.default);
  const objFile = objParser.parse();

  let posArray: { x: number; y: number; z: number }[] = [];
  let nrmArray: { x: number; y: number; z: number }[] = [];

  const fn = vec3.create();
  const p1p0Diff = vec3.create();
  const p2p0Diff = vec3.create();

  return objFile.models.map(({ vertices, vertexNormals, faces, name }) => {
    posArray = posArray.concat(vertices);
    nrmArray = nrmArray.concat(vertexNormals);

    const _faces = faces.map((f): Face => {
      const i0 = f.vertices[0].vertexIndex - 1;
      const i1 = f.vertices[1].vertexIndex - 1;
      const i2 = f.vertices[2].vertexIndex - 1;
      const a = 0;
      const p0 = vec3.fromValues(
        posArray[i0].x,
        posArray[i0].y + a,
        posArray[i0].z
      );
      const p1 = vec3.fromValues(
        posArray[i1].x,
        posArray[i1].y + a,
        posArray[i1].z
      );
      const p2 = vec3.fromValues(
        posArray[i2].x,
        posArray[i2].y + a,
        posArray[i2].z
      );

      const j0 = f.vertices[0].vertexNormalIndex - 1;
      const j1 = f.vertices[1].vertexNormalIndex - 1;
      const j2 = f.vertices[2].vertexNormalIndex - 1;

      const n0 = vec3.fromValues(
        nrmArray[j0].x,
        nrmArray[j0].y,
        nrmArray[j0].z
      );
      const n1 = vec3.fromValues(
        nrmArray[j1].x,
        nrmArray[j1].y,
        nrmArray[j1].z
      );
      const n2 = vec3.fromValues(
        nrmArray[j2].x,
        nrmArray[j2].y,
        nrmArray[j2].z
      );

      vec3.sub(p1p0Diff, p1, p0);
      vec3.sub(p2p0Diff, p2, p0);
      vec3.cross(fn, p1p0Diff, p2p0Diff);
      vec3.normalize(fn, fn);
      return {
        materialIdx: 0,
        normal: vec3.clone(fn),
        points: [
          { position: p0, normal: n0 },
          { position: p1, normal: n1 },
          { position: p2, normal: n2 },
        ],
      };
    });

    return { name, faces: _faces };
  });
};

export const loadModelToBuffer = async (model: Model): Promise<GPUBuffer> => {
  /* 
      struct FacePoint {
        // pos: vec3f,
        // normal: vec3f
        posNormalT: mat3x2f
      }
      struct Face {
        faceNormal: vec3f,
        materialIdx: u32,
        points: array<FacePoint, 3>
      }
 */
  // fuck alignment
  // https://www.w3.org/TR/WGSL/#alignment-and-size
  const facePointSize = 6;
  // const facePointSize = 8;
  const faceSize = 4 + 3 * facePointSize + 2;
  const buffer = createStorageBuffer(
    model.faces.length * faceSize * Float32Array.BYTES_PER_ELEMENT,
    'Faces Buffer',
    true
  );
  const _mapped = buffer.getMappedRange();
  const mapped = new Float32Array(_mapped);
  const faceColorData = new Uint32Array(_mapped);
  for (const [face, i] of Iterator.iter(model.faces).enumerate()) {
    const { points, normal } = face;
    const i2 = i * faceSize;
    mapped[i2 + 0] = normal[0];
    mapped[i2 + 1] = normal[1];
    mapped[i2 + 2] = normal[2];
    faceColorData[i2 + 3] = face.materialIdx;

    for (const [point, j] of Iterator.iter(points).enumerate()) {
      const { position, normal } = point;
      const k = i2 + 4 + j * facePointSize;
      mapped[k + 0] = position[0];
      mapped[k + 1] = normal[0];
      mapped[k + 2] = position[1];
      mapped[k + 3] = normal[1];
      mapped[k + 4] = position[2];
      mapped[k + 5] = normal[2];
    }
  }
  buffer.unmap();
  return buffer;
};

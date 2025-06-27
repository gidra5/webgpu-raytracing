import { vec3 } from 'gl-matrix';
import { Face } from './scene';
import { Iterator } from 'iterator-js';
import { assert } from './utils';

export enum Axis {
  X,
  Y,
  Z,
}

const BV_MIN_DELTA = 0.01;
export type BoundingVolume = {
  min: vec3;
  max: vec3;

  leftIdx: number; // left child BV index
  rightIdx: number; // right child BV index
  faces: number[]; // face indices
};
export type BoundingVolumeHierarchy = BoundingVolume[];

const bv = (min: vec3, max: vec3): BoundingVolume => {
  return { min, max, leftIdx: -1, rightIdx: -1, faces: [-1, -1] };
};

export const facesBV = (faces: Face[]): BoundingVolume => {
  // find root BV dimensions
  const min = vec3.fromValues(
    Number.MAX_SAFE_INTEGER,
    Number.MAX_SAFE_INTEGER,
    Number.MAX_SAFE_INTEGER
  );
  const max = vec3.fromValues(
    Number.MIN_SAFE_INTEGER,
    Number.MIN_SAFE_INTEGER,
    Number.MIN_SAFE_INTEGER
  );

  for (const face of faces) {
    // calculate min/max for root AABB bounding volume
    for (const { position: p } of face.points) {
      vec3.min(min, min, p);
      vec3.max(max, max, p);
    }
  }

  const d = vec3.create();
  vec3.sub(d, max, min);

  for (let i = 0; i < 3; i++) {
    if (d[i] < BV_MIN_DELTA) {
      max[i] += BV_MIN_DELTA;
    }
  }

  return bv(min, max);
};

export const facesBVH = (faces: Face[]): BoundingVolumeHierarchy => {
  const bvh: BoundingVolume[] = [];
  const bv = facesBV(faces);
  bvh.push(bv);
  bvh.push(...subdivide(faces, bv));
  return bvh;
};

const axisMidpoint = (axis: Axis, f: Face): number => {
  return (
    Iterator.iter(f.points)
      .map((p) => p.position[axis])
      .sum() / 3
  );
};

const splitAcross = (axis: Axis, faces: Face[]): BoundingVolume[] => {
  const sorted = faces.toSorted((a, b) => {
    return axisMidpoint(axis, a) - axisMidpoint(axis, b);
  });
  const mid = Math.floor(sorted.length / 2);
  const left = sorted.slice(0, mid);
  const right = sorted.slice(mid);
  const bvh: BoundingVolume[] = [];

  if (left.length > 0) {
    bvh.push(...facesBVH(left));
  }

  if (right.length > 0) {
    bvh.push(...facesBVH(right));
  }

  return bvh;
};

export const subdivide = (
  faces: Face[],
  bv: BoundingVolume
): BoundingVolume[] => {
  if (faces.length <= 2) {
    for (let i = 0; i < faces.length; i++) {
      bv.faces[i] = faces[i].idx;
    }
    return [];
  }
  const d = vec3.create();
  vec3.sub(d, bv.max, bv.min);
  const largestDelta = Math.max(...d);
  if (largestDelta === d[0]) {
    return splitAcross(Axis.X, faces);
  } else if (largestDelta === d[1]) {
    return splitAcross(Axis.Y, faces);
  } else {
    return splitAcross(Axis.Z, faces);
  }
};

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

  // left child BV index is implicitly the next value
  rightIdx: number; // right child BV index
  faces: number[]; // face indices
};
export type BoundingVolumeHierarchy = BoundingVolume[];

const bv = (min: vec3, max: vec3): BoundingVolume => {
  return { min, max, rightIdx: -1, faces: [-1, -1] };
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
    const points = Array.from({ length: 3 }, () => vec3.create());
    vec3.copy(points[0], face.points[0].position);
    vec3.add(points[1], points[0], face.points[1].position);
    vec3.add(points[2], points[0], face.points[2].position);

    // calculate min/max for root AABB bounding volume
    for (const p of points) {
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

export const facesBVH = (
  faces: Face[],
  bvh: BoundingVolumeHierarchy = []
): BoundingVolumeHierarchy => {
  if (faces.length === 0) return bvh;

  const bv = facesBV(faces);
  bvh.push(bv);

  subdivide(faces, bvh);
  return bvh;
};

const axisMidpoint = (axis: Axis, f: Face): number => {
  return (
    Iterator.iter(f.points)
      .map((p) => p.position[axis])
      .sum() / 3
  );
};

const splitAcross = (
  axis: Axis,
  faces: Face[],
  bvh: BoundingVolumeHierarchy
): BoundingVolume[] => {
  const parent = bvh[bvh.length - 1];
  const sorted = faces.toSorted((a, b) => {
    return axisMidpoint(axis, a) - axisMidpoint(axis, b);
  });
  const mid = Math.floor(sorted.length / 2);
  const left = sorted.slice(0, mid);
  const right = sorted.slice(mid);

  if (left.length > 0) {
    facesBVH(left, bvh);
  }
  if (right.length > 0) {
    parent.rightIdx = bvh.length;
    facesBVH(right, bvh);
  }

  return bvh;
};

export const subdivide = (faces: Face[], bvh: BoundingVolumeHierarchy) => {
  const parent = bvh[bvh.length - 1];

  if (faces.length <= 2) {
    for (let i = 0; i < faces.length; i++) {
      parent.faces[i] = faces[i].idx;
    }
    return [];
  }
  const d = vec3.create();
  vec3.sub(d, parent.max, parent.min);
  const largestDelta = Math.max(...d);
  if (largestDelta === d[0]) {
    return splitAcross(Axis.X, faces, bvh);
  } else if (largestDelta === d[1]) {
    return splitAcross(Axis.Y, faces, bvh);
  } else {
    return splitAcross(Axis.Z, faces, bvh);
  }
};

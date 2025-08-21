export default () => /* wgsl */ `
  struct BVHIntersectionResult {
    hit: bool,
    barycentric: vec3f,
    faceIdx: u32,
    objectIdx: u32,
  }

  struct BVHIntersectionStackEntry {
    idx: u32,
    t: f32,
  }
  const BV_MAX_STACK_DEPTH = 16;
  @must_use
  fn rayIntersectBVH(
    ray: Ray,
    int: Interval
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(int.max, 0, 0);
    result.hit = false;
    result.faceIdx = 0;

    for (var objectIdx = 0u; objectIdx < arrayLength(&models); objectIdx++) {
      let int = Interval(int.min, result.barycentric.x);
      let hit = rayIntersectObjectBVH(ray, objectIdx, int);
      if !hit.hit {
        continue;
      }
      result = hit;
    }

    return result;
  }
  
  @must_use
  fn rayBackfaceIntersectBVH(
    ray: Ray,
    int: Interval
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(int.max, 0, 0);
    result.hit = false;
    result.faceIdx = 0;

    for (var objectIdx = 0u; objectIdx < arrayLength(&models); objectIdx++) {
      let int = Interval(int.min, result.barycentric.x);
      let hit = rayBackfaceIntersectObjectBVH(ray, objectIdx, int);
      if !hit.hit {
        continue;
      }
      result = hit;
    }

    return result;
  }

  @must_use
  fn rayIntersectBVHAnyHit(
    ray: Ray,
    maxDist: f32
  ) -> bool {
    for (var objectIdx = 0u; objectIdx < arrayLength(&models); objectIdx++) {
      let hit = rayIntersectObjectBVHAnyHit(ray, objectIdx, maxDist);
      if hit {
        return true;
      }
    }

    return false;
  }

  @must_use
  fn rayIntersectObjectBVHAnyHit(
    ray: Ray,
    objectIdx: u32,
    maxDist: f32
  ) -> bool {
    var stack: array<u32, BV_MAX_STACK_DEPTH>;
    var top: i32;

    let model = models[objectIdx];
    let bv = bvh[model.bvh.offset];
    let bvResult = rayIntersectBV(ray, bv, Interval(min_dist, maxDist));
    if (!bvResult.hit) {
      return false;
    }

    top = 0;
    stack[top] = 0;

    while (top > -1) {
      let idx = stack[top];
      top--;
      let bv = bvh[model.bvh.offset + idx];

      let isLeaf = bv.rightIdx == -1;
      if (isLeaf) {
        for (var i = bv.facesOffset; i < bv.facesOffset + bv.facesCount; i = i + 1) {
          let offset = bvhFaces[i];
          let faceIdx = model.faces.offset + offset;
          let face = faces[faceIdx];
          let hit = rayIntersectFaceAnyHit(ray, face, Interval(min_dist, maxDist));
          if !hit {
            continue;
          }
          return true;
        }
        continue;
      }

      let leftIdx = u32(idx + 1);
      let rightIdx = u32(bv.rightIdx);
      let left = bvh[model.bvh.offset + leftIdx];
      let right = bvh[model.bvh.offset + rightIdx];
      let resultLeft = rayIntersectBV(ray, left, Interval(min_dist, maxDist));
      let resultRight = rayIntersectBV(ray, right, Interval(min_dist, maxDist));
      if resultLeft.hit && resultRight.hit {
        if resultLeft.t < resultRight.t {
          top++;
          stack[top] = rightIdx;
          top++;
          stack[top] = leftIdx;
        } else {
          top++;
          stack[top] = leftIdx;
          top++;
          stack[top] = rightIdx;
        }
      } else if resultLeft.hit {
        top++;
        stack[top] = leftIdx;
      } else if resultRight.hit {
        top++;
        stack[top] = rightIdx;
      }
    }

    return false;
  }

  @must_use
  fn rayIntersectObjectBVH(
    ray: Ray,
    objectIdx: u32,
    int: Interval
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(int.max, 0, 0);
    result.hit = false;
    result.faceIdx = 0;
    
    var stack: array<BVHIntersectionStackEntry, BV_MAX_STACK_DEPTH>;
    var top: i32;

    let model = models[objectIdx];
    let bv = bvh[model.bvh.offset];
    let bvResult = rayIntersectBV(ray, bv, Interval(int.min, result.barycentric.x));
    if (!bvResult.hit) {
      return result;
    }

    top = 0;
    stack[top] = BVHIntersectionStackEntry(0, bvResult.t);

    while (top > -1) {
      let stackEntry = stack[top];
      top--;
      if stackEntry.t > result.barycentric.x {
        continue;
      }

      let idx = stackEntry.idx;
      let bv = bvh[model.bvh.offset + idx];

      let isLeaf = bv.rightIdx == -1; // right will be -1 too
      if (isLeaf) {
        for (var i = bv.facesOffset; i < bv.facesOffset + bv.facesCount; i = i + 1) {
          let offset = bvhFaces[i];
          let faceIdx = model.faces.offset + offset;
          let face = faces[faceIdx];
          let hit = rayIntersectFace(ray, face, Interval(int.min, result.barycentric.x));
          if (!hit.hit) {
            continue;
          }
          result.barycentric = hit.barycentric;
          result.hit = true;
          result.faceIdx = faceIdx;
          result.objectIdx = objectIdx;
        }
        continue;
      }

      let leftIdx = u32(idx + 1);
      let rightIdx = u32(bv.rightIdx);
      let left = bvh[model.bvh.offset + leftIdx];
      let right = bvh[model.bvh.offset + rightIdx];
      let resultLeft = rayIntersectBV(ray, left, Interval(int.min, result.barycentric.x));
      let resultRight = rayIntersectBV(ray, right, Interval(int.min, result.barycentric.x));
      if resultLeft.hit && resultRight.hit {
        let leftEntry = BVHIntersectionStackEntry(leftIdx, resultLeft.t);
        let rightEntry = BVHIntersectionStackEntry(rightIdx, resultRight.t);
        if resultLeft.t < resultRight.t {
          top++;
          stack[top] = rightEntry;
          top++;
          stack[top] = leftEntry;
        } else {
          top++;
          stack[top] = leftEntry;
          top++;
          stack[top] = rightEntry;
        }
      } else if resultLeft.hit {
        top++;
        stack[top] = BVHIntersectionStackEntry(leftIdx, resultLeft.t);
      } else if resultRight.hit {
        top++;
        stack[top] = BVHIntersectionStackEntry(rightIdx, resultRight.t);
      }
    }

    return result;
  }

  @must_use
  fn rayBackfaceIntersectObjectBVH(
    ray: Ray,
    objectIdx: u32,
    int: Interval
  ) -> BVHIntersectionResult {
    var result: BVHIntersectionResult;
    result.barycentric = vec3f(int.max, 0, 0);
    result.hit = false;
    result.faceIdx = 0;
    
    var stack: array<BVHIntersectionStackEntry, BV_MAX_STACK_DEPTH>;
    var top: i32;

    let model = models[objectIdx];
    let bv = bvh[model.bvh.offset];
    let bvResult = rayIntersectBV(ray, bv, Interval(int.min, result.barycentric.x));
    if (!bvResult.hit) {
      return result;
    }

    top = 0;
    stack[top] = BVHIntersectionStackEntry(0, bvResult.t);

    while (top > -1) {
      let stackEntry = stack[top];
      top--;
      if stackEntry.t > result.barycentric.x {
        continue;
      }

      let idx = stackEntry.idx;
      let bv = bvh[model.bvh.offset + idx];

      let isLeaf = bv.rightIdx == -1; // right will be -1 too
      if (isLeaf) {
        for (var i = bv.facesOffset; i < bv.facesOffset + bv.facesCount; i = i + 1) {
          let offset = bvhFaces[i];
          let faceIdx = model.faces.offset + offset;
          let face = faces[faceIdx];
          let hit = rayIntersectBackface(ray, face, Interval(int.min, result.barycentric.x));
          if (!hit.hit) {
            continue;
          }
          result.barycentric = hit.barycentric;
          result.hit = true;
          result.faceIdx = faceIdx;
          result.objectIdx = objectIdx;
        }
        continue;
      }

      let leftIdx = u32(idx + 1);
      let rightIdx = u32(bv.rightIdx);
      let left = bvh[model.bvh.offset + leftIdx];
      let right = bvh[model.bvh.offset + rightIdx];
      let resultLeft = rayIntersectBV(ray, left, Interval(int.min, result.barycentric.x));
      let resultRight = rayIntersectBV(ray, right, Interval(int.min, result.barycentric.x));
      if resultLeft.hit && resultRight.hit {
        let leftEntry = BVHIntersectionStackEntry(leftIdx, resultLeft.t);
        let rightEntry = BVHIntersectionStackEntry(rightIdx, resultRight.t);
        if resultLeft.t < resultRight.t {
          top++;
          stack[top] = rightEntry;
          top++;
          stack[top] = leftEntry;
        } else {
          top++;
          stack[top] = leftEntry;
          top++;
          stack[top] = rightEntry;
        }
      } else if resultLeft.hit {
        top++;
        stack[top] = BVHIntersectionStackEntry(leftIdx, resultLeft.t);
      } else if resultRight.hit {
        top++;
        stack[top] = BVHIntersectionStackEntry(rightIdx, resultRight.t);
      }
    }

    return result;
  }
`;

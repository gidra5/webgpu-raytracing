export default /* wgsl */ `
  struct FaceIntersectonResult {
    barycentric: vec3f,
    hit: bool,
  }

  // Mäller-Trumbore algorithm
  // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
  // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
  @must_use
  fn rayIntersectBackface(ray: Ray, face: Face, interval: Interval) -> FaceIntersectonResult {
    var result: FaceIntersectonResult;
    result.hit = false;
    result.barycentric = vec3f(f32max, 0, 0);

    let p0 = faceVertex(face, 0);
    let e1 = face.e1.dir;
    let e2 = -face.e3.dir;

    let dir = -ray.dir;

    let h = cross(dir, e2);
    let det = dot(e1, h);
    
    // near zero determinant will detect parallel rays
    if det < EPSILON * EPSILON {
      return result;
    }

    let s = ray.pos - p0;
    let u = dot(s, h);

    if u < 0.0f || u > det {
      return result;
    }

    let q = cross(s, e1);
    let v = dot(dir, q);

    if v < 0.0f || u + v > det {
      return result;
    }

    let t = -dot(e2, q);
    let pt = vec3f(t, u, v) / det;

    if !intervalSurrounds(interval, pt.x) {
      return result;
    }

    result.barycentric = pt;
    result.hit = true;
    
    return result;
  }

  @must_use
  fn _rayIntersectFace(ray: Ray, face: Face, interval: Interval) -> FaceIntersectonResult {
      var result: FaceIntersectonResult;
      result.hit = false;
      result.barycentric = vec3f(interval.max, 0, 0);

      let n = face.normal;
      let u = face.uNormal;
      let v = face.vNormal;

      let det = dot(n.xyz, ray.dir);

      if (abs(det) < EPSILON) {
        return result;
      }

      let dett = n.w - dot(n.xyz, ray.pos);
      if (det > 0.0) {
        if (dett < 0.0 || dett > interval.max * det) {
          return result;
        }
        // if (dett > interval.max * det) {
        //   return result;
        // }
        // if (dett < 0.0) {
        //   return result;
        // }
        // return result;
      } else {
        // return result;
        // if (dett > 0.0) {
        // return result;
        // }
        // if (dett < interval.max * det) {
        //   return result;
        // }
        if (dett > 0.0 || dett < interval.max * det) {
          return result;
        }
      }
        
      let detp = ray.pos * det + dett * ray.dir;
      let detu = dot(detp, u.xyz) - u.w * det;
      if (det > 0.0) {
        if (detu < 0.0 || detu > det) {
          return result;
        }
      } else { // det < 0.0
        if (detu > 0.0 || detu < det) { // Note the swapped comparison due to negative det
          return result;
        }
      }
      
      let detv = dot(detp, v.xyz) + v.w * det;
      let detuv = detu + detv;
      if (det > 0.0) {
        if (detv < 0.0 || detuv > det) {
          return result;
        }
      } else { // det < 0.0
        if (detv > 0.0 || detuv < det) { // Note the swapped comparison
          return result;
        }
      }

      let inv_det = 1/det;
      let pt = vec3f(dett, detu, detv) * inv_det;

      result.barycentric = pt;
      result.hit = true;
      return result;
  }
  
  @must_use
  fn rayIntersectFace(ray: Ray, face: Face, interval: Interval) -> FaceIntersectonResult {
    var result: FaceIntersectonResult;
    result.hit = false;
    result.barycentric = vec3f(f32max, 0, 0);

    let p0 = faceVertex(face, 0);
    let e1 = face.e1.dir;
    let e2 = -face.e3.dir;

    let h = cross(ray.dir, e2);
    let det = dot(e1, h);
    
    // near zero determinant will detect parallel rays
    if det < EPSILON * EPSILON {
      return result;
    }

    let s = ray.pos - p0;
    let u = dot(s, h);

    if u < 0.0f || u > det {
      return result;
    }

    let q = cross(s, e1);
    let v = dot(ray.dir, q);

    if v < 0.0f || u + v > det {
      return result;
    }

    let t = dot(e2, q);
    let pt = vec3f(t, u, v) / det;

    if !intervalSurrounds(interval, pt.x) {
      return result;
    }

    result.barycentric = pt;
    result.hit = true;
    
    return result;
  }
  
  fn side(l: OrientedLine, r: OrientedLine) -> f32 {
    return dot(l.cross, r.dir) + dot(l.dir, r.cross);
  }
  
  @must_use
  fn _rayIntersectFaceAnyHit(l: OrientedLine, ray: Ray, face: Face, interval: Interval) -> bool {
    let n = face.normal;
    let det = dot(n.xyz, ray.dir);

    if det < EPSILON {
      return false;
    }

    let t = n.w - dot(n.xyz, ray.pos);
    if !intervalSurrounds(interval, t / det) {
      return false;
    }

    if !(side(l, face.e1) >= 0 && side(l, face.e2) >= 0 && side(l, face.e3) >= 0) {
      return false;
    }

    return true;
  }
  
  @must_use
  fn rayIntersectFaceAnyHit(l: OrientedLine, ray: Ray, face: Face, interval: Interval) -> bool {
    let p0 = faceVertex(face, 0);
    let e1 = face.e1.dir;
    let e2 = -face.e3.dir;

    let h = cross(ray.dir, e2);
    let det = dot(e1, h);
    
    // near zero determinant will detect parallel rays
    if det < EPSILON * EPSILON {
      return false;
    }

    let s = ray.pos - p0;
    let u = dot(s, h);

    if u < 0.0f || u > det {
      return false;
    }

    let q = cross(s, e1);
    let v = dot(ray.dir, q);

    if v < 0.0f || u + v > det {
      return false;
    }

    let t = dot(e2, q) / det;

    return intervalSurrounds(interval, t);
  }
  
  struct BVIntersectionResult {
    hit: bool,
    t: f32,
  }
  
  @must_use
  fn rayIntersectBV(ray: Ray, bv: BoundingVolume, interval: Interval) -> BVIntersectionResult {
    let t0 = (bv.min - ray.pos) / ray.dir;
    let t1 = (bv.max - ray.pos) / ray.dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let near = max(tmin.x, max(tmin.y, tmin.z));
    let far = min(tmax.x, min(tmax.y, tmax.z));
    if near < far && intervalOverlap(interval, Interval(near, far)) {
      return BVIntersectionResult(true, near);
    }
    return BVIntersectionResult(false, f32max);
  }
`;

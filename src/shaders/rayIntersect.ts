export default /* wgsl */ `
  struct FaceIntersectonResult {
    barycentric: vec3f,
    hit: bool,
  }

  struct Triangle {
    p0: vec3f,
    e1: vec3f,
    e2: vec3f,
  }

  @must_use
  fn rayIntersectBackface(ray: Ray, face: Face, interval: Interval) -> FaceIntersectonResult {
    var result: FaceIntersectonResult;
    result.hit = false;
    result.barycentric = vec3f(f32max, 0, 0);

    // Mäller-Trumbore algorithm
    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    
    let p0 = face.points[0].pos;
    let e1 = face.points[1].pos;
    let e2 = face.points[2].pos;

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
  fn rayIntersectFace(ray: Ray, face: Face, interval: Interval) -> FaceIntersectonResult {
    var result: FaceIntersectonResult;
    result.hit = false;
    result.barycentric = vec3f(f32max, 0, 0);

    // Mäller-Trumbore algorithm
    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    
    let p0 = face.points[0].pos;
    let e1 = face.points[1].pos;
    let e2 = face.points[2].pos;

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
  
  @must_use
  fn rayIntersectFaceAnyHit(ray: Ray, face: Face, interval: Interval) -> bool {
    // Mäller-Trumbore algorithm
    // https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    
    let p0 = face.points[0].pos;
    let e1 = face.points[1].pos;
    let e2 = face.points[2].pos;

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

import { NormalsType, store } from '../store';

export default () => /* wgsl */ `
  struct FaceSample {
    inv_pdf: f32,
    point: vec3f,
    normal: vec3f,
    texture: vec2f,
    uv: vec2f,
    materialIdx: u32,
  }

  fn sampleFace(face: Face) -> FaceSample {
    let uv = sample_intriangle(random_2());
    let point = facePointOffset(face, uv);
    let normal = faceNormal(face, uv);
    let texture = faceTexture(face, uv);
    let p = cross(face.points[1].pos, face.points[2].pos);
    return FaceSample(length(p)/2, point, normal, texture, uv, face.materialIdx);
  }

  fn faceVertex(face: Face, i: u32) -> vec3f {
    return face.points[i].pos;
  }

  fn faceVertexNormal(face: Face, i: u32) -> vec3f {
    return face.points[i].normal;
  }

  fn faceVertexTexture(face: Face, i: u32) -> vec2f {
    return vec2f(0, 0);
  }

  // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf
  // p. 83
  // computing from uv and offsetting the point 
  // makes sure there are no self-intersections 
  // due to floating point errors
  fn facePoint(face: Face, uv: vec2f) -> vec3f {
    let p1 = faceVertex(face, 0);
    let e1 = face.e1.dir;
    let e2 = -face.e3.dir;
    return p1 + mat2x3f(e1, e2) * uv;
  }
  fn facePointOffset(face: Face, uv: vec2f) -> vec3f {
    return offsetRay(facePoint(face, uv), face.normal.xyz);
  }

  const origin = 1.0 / 32.0;
  const floatScale = 1.0 / 65536.0;
  const intScale = 256.0;
  fn offsetRay(p: vec3f, n: vec3f) -> vec3f {
    let ofI = vec3i(intScale * n);
    let pI = vec3f(
      bitcast<f32>(bitcast<i32>(p.x) + select(-ofI.x, ofI.x, p.x < 0)),
      bitcast<f32>(bitcast<i32>(p.y) + select(-ofI.y, ofI.y, p.y < 0)),
      bitcast<f32>(bitcast<i32>(p.z) + select(-ofI.z, ofI.z, p.z < 0))
    );
    return vec3f(
      select(p.x + floatScale * n.x, pI.x, abs(p.x) < origin),
      select(p.y + floatScale * n.y, pI.y, abs(p.y) < origin),
      select(p.z + floatScale * n.z, pI.z, abs(p.z) < origin)
    );
  }

  fn faceNormal(face: Face, uv: vec2f) -> vec3f {
    if ${store.normalsType} == ${NormalsType.Interpolated} {
      let n1 = faceVertexNormal(face, 0);
      let n2 = faceVertexNormal(face, 1);
      let n3 = faceVertexNormal(face, 2);
      return normalize(mat3x3f(n1, n2, n3) * toBarycentric(uv));
    } else {
      return face.normal.xyz;
    }
  }

  fn faceTexture(face: Face, uv: vec2f) -> vec2f {
    let t1 = faceVertexTexture(face, 0);
    let t2 = faceVertexTexture(face, 1);
    let t3 = faceVertexTexture(face, 2);
    return mat3x2f(t1, t2, t3) * toBarycentric(uv);
  }

  fn toBarycentric(uv: vec2f) -> vec3f {
    return vec3f(1 - uv.x - uv.y, uv.x, uv.y);
  }
`;

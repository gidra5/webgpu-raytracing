import { makeShaderDataDefinitions } from 'webgpu-utils';

const structs = /* wgsl */ `
  struct Geometry {
    position: vec3f,
    depth: f32,
    faceIdx: u32,
    objectIdx: u32,
  }

  struct Ray {
    pos: vec3f, // Origin
    dir: vec3f, // Direction (normalized)
  };

  struct BoundRay {
    pos: vec3f, // Origin
    dir: vec3f, // Direction (normalized)
    maxDist: f32,
  };

  struct BoundingVolume {
    min: vec3f,
    rightIdx: i32,
    max: vec3f,
    facesCount: u32,
    facesOffset: u32,
  }
  
  struct Offset {
    offset: u32,
    count: u32,
  }

  struct Model {
    faces: Offset,
    bvh: Offset,
  }

  struct FacePoint {
    pos: vec3f,
    normal: vec3f,
  }

  struct Face {
    normal: vec4f,
    uNormal: vec4f,
    vNormal: vec4f,
    materialIdx: u32,
    points: array<FacePoint, 3>
  }

  struct Material {
    color: vec3f,
    emission: vec3f
  };
`;

export const defs = makeShaderDataDefinitions(structs);
export default structs;

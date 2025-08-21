import { FovOrientation, LensShape, ProjectionType, store } from '../store';

export default () => /* wgsl */ `
  const cameraFovAngle = ${store.fov};
  const cameraFovDistance = ${(store.fov / Math.PI) * 4};
  const cameraRayZ = -1/tan(cameraFovAngle / 2);
  const paniniDistance = ${store.paniniDistance};
  const lensFocusDistance = ${store.focusDistance};
  const circleOfConfusionRadius = ${store.circleOfConfusion};
  const projectionType = ${store.projectionType};
  const verticalCompression = ${store.verticalCompression};
  const fovOrientation = ${store.fovOrientation};

  fn pinholeRayDirection(pixel: vec2f) -> vec3f {
    return normalize(vec3(pixel, cameraRayZ));
  }

  fn paniniRayDirection(pixel: vec2f) -> vec3f {
    let half_fov = cameraFovAngle / 2.0;
    let hv = pixel * half_fov;
    let half_panini_fov = atan2(sin(half_fov), cos(half_fov) + paniniDistance);
    let hv_pan = hv * half_panini_fov; 

    let M = sqrt(1.0 - square(sin(hv_pan.x) * paniniDistance)) + paniniDistance * cos(hv_pan.x);
    let x = sin(hv_pan.x) * M;
    let z = (cos(hv_pan.x) * M) - paniniDistance;
    
    let y = tan(hv_pan.y) * (z + paniniDistance * (1.0 - verticalCompression));

    return normalize(vec3<f32>(x, y, -z));
  }

  fn square(x: f32) -> f32 {
    return x * x;
  }

  fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
  }

  fn fisheyeRayDirection(pixel: vec2f) -> vec3f {
    let clampedHalfFOV = cameraFovAngle / 2;
    let angle = pixel * clampedHalfFOV;

    return normalize(vec3(
      -sin(angle.x), 
      -sin(angle.y) * cos(angle.x), 
      cos(angle.y) * cos(angle.x)
    ));
  }

  fn orthographicRayDirection(uv: vec2f) -> vec3f {
    return vec3(0, 0, -1);
  }

  fn thinLensRay(dir: vec3f, uv: vec2f) -> Ray {
    let pos = vec3(uv * circleOfConfusionRadius, 0.f);
    let focusPoint = -dir * lensFocusDistance / dir.z;
    return Ray(
      pos,
      normalize(focusPoint - pos)
    );
  }

  fn cameraRayDirection(uv: vec2f) -> vec3f {
    switch (projectionType) {
      case ${ProjectionType.Panini}: {
        return paniniRayDirection(uv);
      }
      case ${ProjectionType.Perspective}: {
        return pinholeRayDirection(uv);
      }
      case ${ProjectionType.Orthographic}: {
        return orthographicRayDirection(uv);
      }
      case ${ProjectionType.Fisheye}: {
        return fisheyeRayDirection(uv);
      }
      default: {
        return vec3(0);
      }
    }
  }

  fn cameraRayPosition(uv: vec2f) -> vec3f {
    if projectionType == ${ProjectionType.Orthographic} {
      return vec3(uv * cameraFovDistance, 0);
    }
    return vec3(0);
  }

  fn ray_transform(_ray: Ray, view: mat4x4f) -> Ray {
    let worldSpaceJitter = jitter / viewportf;
    var ray = _ray;
    let ray_pos = view * vec4(ray.pos + vec3f(worldSpaceJitter, 0), 1.);
    ray.pos = ray_pos.xyz;
    ray.dir = normalize(vec3(ray.dir.xy, ray.dir.z * ray_pos.w));
    ray.dir = (view * vec4(ray.dir, 0.)).xyz;
    return ray;
  }

  fn sampleLens() -> vec2f {
    if ${store.lensShape} == ${LensShape.Circle} {
      return sample_incircle(random_2());
    } else if ${store.lensShape} == ${LensShape.Square} {
      return sample_insquare(random_2());
    }
    return vec2f(0);
  }

  fn cameraRay(pos: vec2f, view: mat4x4f) -> Ray {
    var uv = (2. * pos - viewportf);

    if ${store.fovOrientation} == ${FovOrientation.Vertical} {
      uv /= viewportf.y;
    } else if ${store.fovOrientation} == ${FovOrientation.Horizontal} {
      uv /= viewportf.x;
    } else if ${store.fovOrientation} == ${FovOrientation.Diagonal} {
      uv /= length(viewportf);
    }

    let rayDirection = cameraRayDirection(uv);
    
    var ray = thinLensRay(rayDirection, sampleLens());
    ray.pos += cameraRayPosition(uv);
    return ray_transform(ray, view);
  }
`;

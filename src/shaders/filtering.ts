import { store } from '../store';

export const historyFilterWGSL = (
  layers: number,
  accuracy: number
) => /* wgsl */ `
  fn bilateralFilter(uv: vec2f, g: Geometry, p: vec3f, c: vec3f) -> vec4f {
    // A footprint can overlap the viewport even when its center is outside it.
    // These ordered comparisons also reject NaN before integer conversion.
    if !all(uv >= vec2f(-2)) || !all(uv < vec2f(viewport) + 2.0) {
      return vec4f(0);
    }
    // Validate texels before filtering: interpolating geometry across a silhouette
    // invents a surface and discards otherwise usable history.
    let savedLayer = layer;
    let normal = normalize(faces[g.faceIdx].plane.xyz);
    let tolerance = max(${accuracy}, 2e-6 * max(1.0, length(p)));
    let base = vec2i(floor(uv));
    var color = vec4f(0);
    var weight = 0.0;
    for (var l = 0u; l < ${layers}u; l++) {
      layer = l;
      for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
          let pixel = base + vec2i(x, y);
          // Keep signed coordinates until bounds have been checked.
          if any(pixel < vec2i(0)) || any(pixel >= vec2i(viewport)) {
            continue;
          }
          let previous = prevGeometryBuffer[geometryIdx(vec2u(pixel))];
          if previous.depth <= 0.0 || previous.depth >= 1e8 ||
             previous.objectIdx != g.objectIdx || previous.faceIdx != g.faceIdx {
            continue;
          }
          let previousNormal = normalize(faces[previous.faceIdx].plane.xyz);
          if dot(normal, previousNormal) < 0.99 ||
             faces[previous.faceIdx].materialIdx != faces[g.faceIdx].materialIdx {
            continue;
          }
          // Tangential separation grows at grazing angles; only separation
          // normal to the surface indicates a different surface.
          if abs(dot(normal, previous.position - p)) > tolerance ||
             abs(dot(previousNormal, previous.position - p)) > tolerance {
            continue;
          }
          let history = textureLoad(prevImageTexture, pixel, l);
          if !(history.w > 0.0) {
            continue;
          }
          let delta = vec2f(pixel) - uv;
          let tapWeight = exp(-0.5 * dot(delta, delta));
          color += history * tapWeight;
          weight += tapWeight;
        }
      }
    }
    layer = savedLayer;
    if weight > 0.0 {
      return color / weight;
    }
    return vec4f(0);
  }
`;

export default () =>
  historyFilterWGSL(store.gBuffer.layers, store.reprojection.pointAccuracy);

import { store } from '../store';

export default () => /* wgsl */ `
  fn bilateralFilterWeight(g: Geometry, p: vec3f, c: vec3f, prev_g: Geometry, prev_c: vec3f) -> f32 {
    let sigmaPos = 1e-12;
    let sigmaColor = 1e-4;
    let faceSigma = 1e-1;
    let objectSigma = 1e-2;
    let faceSigma2 = 1e-3;
    let objectSigma2 = 1e-3;

    let dp = p - prev_g.position;
    let dc = (c - prev_c) * ${store.exposure};
    let d = dot(dp, dp);
    var w = d / sigmaPos + dot(dc, dc) / sigmaColor;

    if prev_g.faceIdx != g.faceIdx {
      w += 1/faceSigma;
    } else {
      w *= faceSigma2;
    }

    if prev_g.objectIdx != g.objectIdx {
      w += 1/objectSigma;
    } else {
      w *= objectSigma2;
    }

    return exp(-w);
  }

  fn bilateralFilter(uv: vec2f, g: Geometry, p: vec3f, c: vec3f) -> vec4f {
    let r = 2;
    let s = 1.;
    let _layer = layer;
    let k = 1/(1e+4 * f32(r));

    var color = vec4f(0);
    var weight = 0.;

    for (var i = -r; i <= r; i = i + 1) {
      for (var j = -r; j <= r; j = j + 1) {
        for (var l = 0u; l < ${store.imageLayers}; l = l + 1) {
          layer = l;
          let _uv = uv + vec2f(f32(i), f32(j)) * s;
          let _color = sampleImage4(_uv, &prevGeometryBuffer);
          if _color.w <= 0 {
            continue;
          }
          let _c = _color.xyz/_color.w;
  
          let _g = sampleGeometryAll(_uv, &prevGeometryBuffer);
          let _weight = bilateralFilterWeight(g, p, c, _g, _c) * k;
          color += _color * _weight;
          weight += _weight;
        }
      }
    }

    layer = _layer;

    if weight == 0. {
      return vec4f(0);
    }

    return color / weight;
  }
`;

export default /* wgsl */ `

fn bilateralFilter(uv: vec2f, p: vec3f, c: vec3f) -> vec4f {
    let r = 2;
    let sigmaPos = 0.01;
    let sigmaColor = 0.01;
    let s = 0.1;

    var color = vec4f(0);
    var weight = 0.;

    for (var i = -r; i <= r; i = i + 1) {
      for (var j = -r; j <= r; j = j + 1) {
        let _uv = uv + vec2f(f32(i), f32(j)) * s;
        let _color = sampleImage4(_uv, &prevImageBuffer, &prevGeometryBuffer);
        if _color.w <= 0 {
          continue;
        }

        let _pos = sampleGeometryAll(_uv, &prevGeometryBuffer).position;
        let dp = p - _pos;
        let dc = c - _color.xyz/_color.w;
        let w = dot(dp, dp) / sigmaPos + dot(dc, dc) / sigmaColor;
        let _weight = exp(-w);
        color += _color * _weight;
        weight += _weight;
      }
    }

    if weight == 0. {
      return vec4f(0);
    }

    return color / weight;
  }
`;

export default /* wgsl */ `
  fn is_nan1(x: f32) -> bool { 
    return min(x, f32max) == f32max;
  }

  ${['vec2f', 'vec3f', 'vec4f']
    .map(
      (type, i) => /* wgsl */ `
        fn is_nan${i + 2}(x: ${type}) -> bool {
          let v = ${type}(f32max);
          return any(min(x, v) == v);
        }
      `
    )
    .join('\n\n')}
    
  fn cbrt(x: f32) -> f32 {
    var y = sign(x) * bitcast<f32>( bitcast<u32>( abs(x) ) / 3u + 0x2a514067u );

    for (var i = 0; i < 2; i = i + 1) { 
      y = (2. * y + x / (y * y)) * .333333333; 
    }

    for (var i = 0; i < 1; i = i + 1)
    {
      let y3 = y * y * y;
      y *= (y3 + 2. * x) / (2. * y3 + x);
    }
    
    return y;
  }
`;

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
`;

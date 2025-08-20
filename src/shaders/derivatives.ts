export default () => /* wgsl */ `
  ${['f32', 'vec2f', 'vec3f', 'vec4f']
    .map(
      (type, i) => /* wgsl */ `
      fn dFdx${i + 1}(p: ${type}) -> ${type} {
        var dx = p - quadSwapX(p);
        if quadIdx == 0 || quadIdx == 2 {
          dx = -dx;
        }
        return dx;
      }

      fn dFdy${i + 1}(p: ${type}) -> ${type} {
        var dy = p - quadSwapY(p);
        if quadIdx == 0 || quadIdx == 1 {
          dy = -dy;
        }
        return dy;
      }
    `
    )
    .join('\n\n')}
`;

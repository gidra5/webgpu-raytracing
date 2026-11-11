export const depthBinVertexWGSL = /* wgsl */ `
  @vertex
  fn main(@location(0) position: vec4f) -> @builtin(position) vec4f {
    return position;
  }
`;

export const depthBinFragmentWGSL = (
  layers: number,
  bindings: string,
  epsilon = 1e-5
) => {
  const outputLocations = Array.from(
    { length: layers },
    (_, index) => /* wgsl */ `
    @location(${2 * index}) depth${index}: f32,
    @location(${2 * index + 1}) count${index}: f32,
  `
  ).join('\n');
  const assignments = Array.from(
    { length: layers },
    (_, index) => /* wgsl */ `
    let inRange${index} = depth >= rangeMin[${index}u] - epsilon && depth <= rangeMax[${index}u] + epsilon;
    output.depth${index} = select(1.0, depth, inRange${index});
    output.count${index} = select(0.0, 1.0, inRange${index});
  `
  ).join('\n');

  return /* wgsl */ `
    ${bindings}

    const layerCount = ${layers}u;
    const epsilon = ${epsilon};

    struct FragmentInput {
      @builtin(position) fragmentPosition: vec4f,
    }

    struct FragmentOutput {
      ${outputLocations}
    }

    @fragment
    fn main(input: FragmentInput) -> FragmentOutput {
      let pixel = vec2u(input.fragmentPosition.xy);
      let depth = clamp(input.fragmentPosition.z, 0.0, 1.0);

      var prevDepths: array<f32, layerCount>;
      var prevCounts: array<u32, layerCount>;
      var rangeMin: array<f32, layerCount>;
      var rangeMax: array<f32, layerCount>;

      var totalCount: u32 = 0u;
      for (var i: u32 = 0u; i < layerCount; i = i + 1u) {
        let d = textureLoad(prevDepthTexture, pixel, i, 0).r;
        let c = textureLoad(prevCountTexture, pixel, i, 0).r;
        prevDepths[i] = clamp(d, 0.0, 1.0);
        let count = u32(round(max(c, 0.0)));
        prevCounts[i] = count;
        totalCount += count;
      }

      if (totalCount == 0u) {
        var lower = 0.0;
        for (var i: u32 = 0u; i < layerCount; i = i + 1u) {
          let upper = max(lower + epsilon, prevDepths[i]);
          rangeMin[i] = clamp(lower, 0.0, 1.0);
          rangeMax[i] = clamp(upper, 0.0, 1.0);
          lower = upper;
        }
      } else {
        var newIndex: u32 = 0u;
        var lastUpper = 0.0;
        var i: u32 = 0u;
        loop {
          if (i >= layerCount || newIndex >= layerCount) {
            break;
          }

          let count = prevCounts[i];
          if (count == 0u) {
            lastUpper = max(lastUpper, prevDepths[i]);
            i = i + 1u;
            continue;
          }

          var lower = 0.0;
          if (i > 0u) {
            lower = prevDepths[i - 1u];
          }
          lower = max(lower, lastUpper);
          var upper = max(lower + epsilon, prevDepths[i]);

          rangeMin[newIndex] = clamp(lower, 0.0, 1.0);
          rangeMax[newIndex] = clamp(upper, 0.0, 1.0);
          lastUpper = upper;
          newIndex = newIndex + 1u;

          if (count > 1u && newIndex < layerCount) {
            var nextUpper = 1.0;
            if (i + 1u < layerCount) {
              nextUpper = max(prevDepths[i + 1u], upper) + epsilon;
            }
            let available = layerCount - newIndex;
            let binsToAllocate = min(count - 1u, available);
            if (binsToAllocate > 0u) {
              let step = (nextUpper - upper) / f32(binsToAllocate);
              var k: u32 = 0u;
              loop {
                if (k >= binsToAllocate || newIndex >= layerCount) {
                  break;
                }
                let start = upper + step * f32(k);
                let finish = upper + step * f32(k + 1u);
                rangeMin[newIndex] = clamp(start, 0.0, 1.0);
                rangeMax[newIndex] = clamp(finish, 0.0, 1.0);
                lastUpper = finish;
                newIndex = newIndex + 1u;
                k = k + 1u;
              }
            }
          }

          i = i + 1u;
        }

        while (newIndex < layerCount) {
          var lower = 0.0;
          if (newIndex > 0u) {
            lower = rangeMax[newIndex - 1u];
          }
          rangeMin[newIndex] = clamp(lower, 0.0, 1.0);
          rangeMax[newIndex] = 1.0;
          newIndex = newIndex + 1u;
        }
      }

      var output: FragmentOutput;
      ${assignments}
      return output;
    }
  `;
};

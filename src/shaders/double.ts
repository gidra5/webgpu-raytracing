import { Iterator } from 'iterator-js';

export default /* wgsl */ `
  const MANTISSA_WIDTH_HALF = 12;
  const F32_SCALE = f32((1 << MANTISSA_WIDTH_HALF) + 1);

  ${['f32', 'vec2f', 'vec3f', 'vec4f']
    .map((type, i) => {
      const _i = i + 1;
      const postfix = _i > 1 ? '_' + _i : '';
      return /* wgsl */ `
        struct F64_${_i} {
          hi: ${type},
          lo: ${type},
        };

        fn split${postfix}(x: ${type}) -> F64_${_i} {
          var result: F64_${_i};

          let p = x * F32_SCALE;
          result.hi = x - p + p;
          result.lo = x - result.hi;
          
          return result;
        }

        fn merge${postfix}(x: F64_${_i}) -> ${type} {
          return x.hi + x.lo;
        }
        
        fn mul2${postfix}(a: ${type}, b: ${type}) -> F64_${_i} {
          let a_split = split${postfix}(a);
          let b_split = split${postfix}(b);

          let p = a_split.hi * b_split.hi;
          let q = a_split.hi * b_split.lo + a_split.lo * b_split.hi;

          var result: F64_${_i};
          result.hi = p + q;
          result.lo = p - result.hi + q + a_split.lo * b_split.lo;

          return result;
        }

        fn mul${postfix}(a: F64_${_i}, b: F64_${_i}) -> F64_${_i} {
          let t = mul2${postfix}(a.hi, b.hi);
          let c = a.hi * b.lo + a.lo * b.hi + t.lo;

          var result: F64_${_i};
          result.hi = t.hi + c;
          result.lo = t.hi - result.hi + c;
          
          return result;
        }

        fn div${postfix}(a: F64_${_i}, b: F64_${_i}) -> F64_${_i} {
          var u: F64_${_i};
          u.hi = a.hi / b.hi;

          let t = mul2${postfix}(u.hi, b.hi);
          let l = (a.hi - t.hi - t.lo + a.lo - u.hi * b.lo) / b.hi;

          var result: F64_${_i};
          result.hi = u.hi + l;
          result.lo = u.hi - result.hi + l;
          
          return result;
        }

        fn add${postfix}(a: F64_${_i}, b: F64_${_i}) -> F64_${_i} {
          var r: F64_${_i};
          r.hi = a.hi + b.hi;
          
          ${
            i > 0
              ? Iterator.range(0, _i)
                  .map((i) => {
                    return /* wgsl */ `
                      if abs(r.lo[${i}]) > abs(a.lo[${i}]) {
                        r.lo[${i}] = a.hi[${i}] - r.hi[${i}] + b.hi[${i}] + b.lo[${i}] + a.lo[${i}];
                      } else {
                        r.lo[${i}] = b.hi[${i}] - r.hi[${i}] + a.hi[${i}] + a.lo[${i}] + b.lo[${i}];
                      }
                    `;
                  })
                  .join('\n\n')
              : /* wgsl */ `
                if abs(r.lo) > abs(a.lo) {
                  r.lo = a.hi - r.hi + b.hi + b.lo + a.lo;
                } else {
                  r.lo = b.hi - r.hi + a.hi + a.lo + b.lo;
                }
              `
          }
          
          var result: F64_${_i};
          result.hi = r.hi + r.lo;
          result.lo = r.hi - result.hi + r.lo;
          
          return result;
        }

        fn sub${postfix}(a: F64_${_i}, b: F64_${_i}) -> F64_${_i} {
          var r: F64_${_i};
          r.hi = a.hi - b.hi;
          
          ${
            i > 0
              ? Iterator.range(0, _i)
                  .map((i) => {
                    return /* wgsl */ `
                      if abs(r.lo[${i}]) > abs(a.lo[${i}]) {
                        r.lo[${i}] = a.hi[${i}] - r.hi[${i}] - b.hi[${i}] - b.lo[${i}] + a.lo[${i}];
                      } else {
                        r.lo[${i}] = -b.hi[${i}] - r.hi[${i}] + a.hi[${i}] + a.lo[${i}] - b.lo[${i}];
                      }
                    `;
                  })
                  .join('\n\n')
              : /* wgsl */ `
                if abs(r.lo) > abs(a.lo) {
                  r.lo = a.hi - r.hi - b.hi - b.lo + a.lo;
                } else {
                  r.lo = -b.hi - r.hi + a.hi + a.lo - b.lo;
                }
              `
          }

          var result: F64_${_i};
          result.hi = r.hi + r.lo;
          result.lo = r.hi - result.hi + r.lo;
          
          return result;
        }

        fn add32${postfix}(a: ${type}, b: ${type}) -> ${type} {
          return merge${postfix}(add${postfix}(split${postfix}(a), split${postfix}(b)));
        }

        fn sub32${postfix}(a: ${type}, b: ${type}) -> ${type} {
          return merge${postfix}(sub${postfix}(split${postfix}(a), split${postfix}(b)));
        }

        fn mul32${postfix}(a: ${type}, b: ${type}) -> ${type} {
          return merge${postfix}(mul${postfix}(split${postfix}(a), split${postfix}(b)));
        }

        fn div32${postfix}(a: ${type}, b: ${type}) -> ${type} {
          return merge${postfix}(div${postfix}(split${postfix}(a), split${postfix}(b)));
        }
      `;
    })
    .join('\n\n')};
    
  
  ${[
    [2, 2],
    [2, 3],
    [2, 4],
    [3, 2],
    [3, 3],
    [3, 4],
    [4, 2],
    [4, 3],
    [4, 4],
  ]
    .map((d, i) => {
      const type = `mat${d[0]}x${d[1]}f`;
      const _i = `${d[0]}x${d[1]}`;
      const postfix = `_${_i}`;
      return /* wgsl */ `
        struct F64_${_i} {
          hi: ${type},
          lo: ${type},
        };

        fn split${postfix}(x: ${type}) -> F64_${_i} {
          var result: F64_${_i};

          let p = x * F32_SCALE;
          result.hi = x - p + p;
          result.lo = x - result.hi;
          
          return result;
        }

        fn merge${postfix}(x: F64_${_i}) -> ${type} {
          return x.hi + x.lo;
        }
      `;
    })
    .join('\n\n')};
  
  ${[2, 3, 4]
    .map((_i) => {
      const postfix = _i > 1 ? '_' + _i : '';
      return /* wgsl */ `
        fn dot${postfix}(a: F64_${_i}, b: F64_${_i}) -> F64_1 {
          let p = mul${postfix}(a, b);

          var result = split(0);

          ${Iterator.range(0, _i)
            .map((i) => {
              return /* wgsl */ `
                result = add(result, F64_1(p.hi[${i}], p.lo[${i}]));
              `;
            })
            .join('\n\n')}

          return result;
        }

        fn scale${postfix}(a: F64_${_i}, b: F64_1) -> F64_${_i} {
          var result: F64_${_i};

          ${Iterator.range(0, _i)
            .map((i) => {
              return /* wgsl */ `
                {
                  let r = mul(F64_1(a.hi[${i}], a.lo[${i}]), b);
                  result.hi[${i}] = r.hi;
                  result.lo[${i}] = r.lo;
                }
              `;
            })
            .join('\n\n')}
            
          return result;
        }

        ${[2, 3, 4]
          .map((j) => {
            return /* wgsl */ `
            fn matmul${postfix}x${j}(a: F64_${_i}x${j}, b: F64_${_i}) -> F64_${j} {
              var result = scale_${j}(F64_${j}(a.hi[0], a.lo[0]), F64_1(b.hi[0], b.lo[0]));

              ${Iterator.range(1, _i)
                .map((i) => {
                  return /* wgsl */ `
                    result = add_${j}(
                      result,
                      scale_${j}(F64_${j}(a.hi[${i}], a.lo[${i}]), F64_1(b.hi[${i}], b.lo[${i}]))
                    );
                  `;
                })
                .join('\n\n')}

              return result;
            }
          `;
          })
          .join('\n\n')}
      `;
    })
    .join('\n\n')};
`;

import constants from './constants';

const phi = (d: number) => {
  const k = 1 / (d + 1);
  let x = (d + 2) * k;
  for (let i = 0; i < 10; i++) x = Math.pow(1 + x, k);
  return x;
};
const alpha = (d: number) => {
  const g = phi(d);
  const a = Array.from({ length: d }, (_, i) => 0);
  for (let i = 0; i < d; i++) a[i] = Math.pow(1 / g, i + 1);
  return a;
};

const randVectors = ['vec2', 'vec3', 'vec4']
  .map((t, i) => {
    const outCount = i + 2;
    const values = Array.from({ length: outCount }, () => 'random_1u()').join(
      ', '
    );
    const valuesFloat = Array.from(
      { length: outCount },
      () => 'random_1()'
    ).join(', ');
    return /* wgsl */ `
      @must_use
      fn random_${outCount}u() -> ${t}<u32> {
        return ${t}(${values});
      }

      @must_use
      fn random_${outCount}() -> ${t}<f32> {
        return ${t}(${valuesFloat});
      }
    `;
  })
  .join('\n\n');

const randUtils = /* wgsl */ `
  var<private> rng_state: u32 = 0;
  @must_use
  fn random_1u() -> u32 {
    let oldState = rng_state + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    rng_state = (word >> 22u) ^ word;
    return rng_state;
  }
  
  @must_use
  fn random_1() -> f32 {
    return f32(random_1u()) / f32(0xffffffffu);
  }

  ${randVectors}
`;

const randUtilsLowDiscrepancy = ['f32', 'vec2f', 'vec3f', 'vec4f']
  .map((t, i) => {
    let _i = i + 1;
    return /* wgsl */ `
      const rng_state_ld_${_i}_a = ${t}(${alpha(_i).join(', ')});
      var<private> rng_state_ld_${_i} = ${t}(0);
      @must_use
      fn random_ld_${_i}() -> ${t} {
        rng_state_ld_${_i} = fract(rng_state_ld_${_i} + rng_state_ld_${_i}_a);
        return rng_state_ld_${_i};
      }
    `;
  })
  .join('\n\n');

export default /* wgsl */ `
  ${constants}
  ${randUtils}
  ${randUtilsLowDiscrepancy}

  fn sample_circle(t: f32) -> vec2f {
    let phi = t * TWO_PI;
    return vec2(cos(phi), sin(phi));
  }

  fn sample_incircle(t: vec2f) -> vec2f {
    return sample_circle(t.x) * sqrt(t.y);
  }

  fn sample_cosine_weighted_sphere(t: vec2f, p: f32) -> vec3f {
    let phi = TWO_PI * t.y;
    let cos_theta = pow(t.x, 1 / (1 + p));
    let sin_theta = sqrt(1. - cos_theta * cos_theta);
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let z = cos_theta;
    return vec3(x, y, z); 
  }

  fn sample_cosine_weighted_hemisphere(t: vec2f, p: f32, n: vec3f) -> vec3f {
    // let phi = TWO_PI * t.y;
    // let cos_theta = pow(t.x, 1 / (1 + p));
    // let sin_theta = sqrt(1. - cos_theta * cos_theta);
    // let x = sin_theta * cos(phi);
    // let y = sin_theta * sin(phi);
    // let z = cos_theta;
    // return vec3(x, y, z);
    return normalize(n + sample_sphere(t));
    // let s = sample_sphere(t);
    // let q = 2 * (1 + dot(s, n));
    // return (n + s) * inverseSqrt(q);
  }

  fn sample_sphere(t: vec2f) -> vec3f {
    let uv = vec2(t.x * 2. - 1., t.y);
    let sin_theta = sqrt(1 - uv.x * uv.x); 
    let phi = TWO_PI * uv.y; 
    let x = sin_theta * cos(phi); 
    let z = sin_theta * sin(phi); 
    return vec3(x, uv.x, z);
  }

  fn sample_hemisphere(t: vec2f, n: vec3f) -> vec3f {
    let uv = vec2(t.x * 2. - 1., t.y);
    let sin_theta = sqrt(1 - uv.x * uv.x); 
    let phi = TWO_PI * uv.y; 
    let x = sin_theta * cos(phi); 
    let z = sin_theta * sin(phi); 
    let v = vec3(x, uv.x, z);
    return faceForward(v, v, -n);
  }
  
  fn sample_insphere(t: vec3f) -> vec3f {
    return sample_sphere(t.xy) * cbrt(t.z); 
  }

  fn sample_insquare(t: vec2f) -> vec2f {
    return (2. * t - 1.); 
  }

  fn sample_intriangle(t: vec2f) -> vec2f {
    return select(t, vec2f(1. - t.y, t.x), t.x < t.y);
  }

  fn pdf_inv_cosine_weighted_hemisphere(s: vec3f, p: f32) -> f32 {
    return TWO_PI / ((1 + p) * pow(s.z, p));
  }

  fn pdf_inv_cosine_weighted_sphere(s: vec3f, p: f32) -> f32 {
    return 2 * TWO_PI / ((1 + p) * pow(s.z, p));
  }

  fn pdf_inv_sphere() -> f32 {
    return 2 * TWO_PI;
  }

  fn pdf_inv_hemisphere() -> f32 {
    return TWO_PI;
  }

  fn pdf_inv_circle() -> f32 {
    return TWO_PI;
  }

  fn pdf_inv_incircle() -> f32 {
    return PI;
  }

  fn pdf_inv_insphere() -> f32 {
    return PI * 4/3;
  }

  fn pdf_inv_intriangle() -> f32 {
    return 0.5;
  }

  fn pdf_inv_insquare() -> f32 {
    return 4;
  }
`;

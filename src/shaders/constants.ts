export default /* wgsl */ `
  const PHI     = 1.61803398874989484820459; // Golden Ratio
  const SRT     = 1.41421356237309504880169; // Square Root of Two
  const PI      = 3.14159265358979323846264;
  const E       = 2.71828182845904523536028;
  const TWO_PI  = 6.28318530717958647692528;
  const INV_PI  = 0.31830988618379067153776;
  
  const EPSILON = 0.001;
  const f32min = 0x1p-126f;
  const f32max = 0x1.fffffep+127;
  // const min_dist = EPSILON;
  const min_dist = 0;
  const max_dist = f32max;
`;

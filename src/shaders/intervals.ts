export default /* wgsl */ `
  struct Interval {
    min: f32,
    max: f32,
  };
  
  @must_use
  fn intervalOverlap(interval1: Interval, interval2: Interval) -> bool {
    return interval1.min <= interval2.max || interval2.min <= interval1.max;
  }
  
  @must_use
  fn intervalContains(interval: Interval, x: f32) -> bool {
    return interval.min <= x && x <= interval.max;
  }

  @must_use
  fn intervalSurrounds(interval: Interval, x: f32) -> bool {
    return interval.min < x && x < interval.max;
  }

  @must_use
  fn intervalClamp(interval: Interval, x: f32) -> f32 {
    return min(max(x, interval.min), interval.max);
  }

  const emptyInterval = Interval(f32max, f32min);
  const universeInterval = Interval(f32min, f32max);
  const positiveUniverseInterval = Interval(EPSILON, f32max);
`;

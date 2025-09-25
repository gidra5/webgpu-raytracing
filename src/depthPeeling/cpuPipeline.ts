export const DEPTH_PEELING_EPSILON = 1e-5;

const EPSILON_FOR_COMPARE = 1e-6;

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const dedupeSorted = (values: number[]) => {
  const result: number[] = [];
  for (const value of values.slice().sort((a, b) => a - b)) {
    if (result.length === 0 || Math.abs(result[result.length - 1] - value) > EPSILON_FOR_COMPARE) {
      result.push(value);
    }
  }
  return result;
};

export const runNaiveDepthPeeling = (fragments: number[], layers: number) => {
  if (layers <= 0 || fragments.length === 0) return [];
  const limited = Math.min(layers, fragments.length);
  return fragments
    .slice()
    .sort((a, b) => a - b)
    .slice(0, limited);
};

type RangeSet = {
  min: number[];
  max: number[];
};

const buildInitialRanges = (prevDepths: number[]): RangeSet => {
  const layerCount = prevDepths.length;
  const min: number[] = new Array(layerCount);
  const max: number[] = new Array(layerCount);
  let lower = 0;
  for (let i = 0; i < layerCount; i += 1) {
    const upper = Math.max(lower + DEPTH_PEELING_EPSILON, clamp01(prevDepths[i]));
    min[i] = clamp01(lower);
    max[i] = clamp01(upper);
    lower = upper;
  }
  return { min, max };
};

const buildSecondaryRanges = (prevDepths: number[], prevCounts: number[]): RangeSet => {
  const layerCount = prevDepths.length;
  const min: number[] = new Array(layerCount).fill(0);
  const max: number[] = new Array(layerCount).fill(1);
  let newIndex = 0;
  let lastUpper = 0;

  for (let i = 0; i < layerCount && newIndex < layerCount; i += 1) {
    const count = prevCounts[i];
    if (count === 0) {
      lastUpper = Math.max(lastUpper, prevDepths[i]);
      continue;
    }

    let lower = i > 0 ? prevDepths[i - 1] : 0;
    lower = Math.max(lower, lastUpper);
    const upper = Math.max(lower + DEPTH_PEELING_EPSILON, prevDepths[i]);

    min[newIndex] = clamp01(lower);
    max[newIndex] = clamp01(upper);
    lastUpper = upper;
    newIndex += 1;

    if (count > 1 && newIndex < layerCount) {
      let nextUpper = 1;
      if (i + 1 < layerCount) {
        nextUpper = Math.max(prevDepths[i + 1], upper) + DEPTH_PEELING_EPSILON;
      }
      const binsToAllocate = Math.min(count - 1, layerCount - newIndex);
      if (binsToAllocate > 0) {
        const step = (nextUpper - upper) / binsToAllocate;
        for (let k = 0; k < binsToAllocate && newIndex < layerCount; k += 1) {
          const start = upper + step * k;
          const finish = upper + step * (k + 1);
          min[newIndex] = clamp01(start);
          max[newIndex] = clamp01(finish);
          lastUpper = finish;
          newIndex += 1;
        }
      }
    }
  }

  while (newIndex < layerCount) {
    const lower = newIndex === 0 ? 0 : max[newIndex - 1];
    min[newIndex] = clamp01(lower);
    max[newIndex] = 1;
    newIndex += 1;
  }

  return { min, max };
};

type SimulationResult = {
  depths: number[];
  counts: number[];
  discovered: number[];
};

const simulateDepthPeeling = (fragments: number[], layers: number): SimulationResult => {
  if (layers <= 0) {
    return { depths: [], counts: [], discovered: [] };
  }

  if (fragments.length === 0) {
    return {
      depths: new Array(layers).fill(1),
      counts: new Array(layers).fill(0),
      discovered: [],
    };
  }

  let prevDepths = Array.from({ length: layers }, (_, i) => (i + 1) / layers);
  let prevCounts = new Array(layers).fill(0);
  const discovered: number[] = [];

  for (let iteration = 0; iteration < layers; iteration += 1) {
    const totalCount = prevCounts.reduce((acc, value) => acc + value, 0);
    const ranges =
      totalCount === 0 ? buildInitialRanges(prevDepths) : buildSecondaryRanges(prevDepths, prevCounts);

    const currDepths = new Array(layers).fill(1);
    const currCounts = new Array(layers).fill(0);

    for (const depth of fragments) {
      const clampedDepth = clamp01(depth);
      for (let bin = 0; bin < layers; bin += 1) {
        if (
          clampedDepth >= ranges.min[bin] - DEPTH_PEELING_EPSILON &&
          clampedDepth <= ranges.max[bin] + DEPTH_PEELING_EPSILON
        ) {
          currDepths[bin] = Math.min(currDepths[bin], clampedDepth);
          currCounts[bin] += 1;
        }
      }
    }

    for (let bin = 0; bin < layers; bin += 1) {
      if (currCounts[bin] === 1 && currDepths[bin] < 1) {
        discovered.push(currDepths[bin]);
      }
    }

    prevDepths = currDepths;
    prevCounts = currCounts;
  }

  return {
    depths: prevDepths.slice(),
    counts: prevCounts.slice(),
    discovered,
  };
};

export const runDepthPeelingPipelineCPUState = (fragments: number[], layers: number) => {
  const { depths, counts } = simulateDepthPeeling(fragments, layers);
  return { depths, counts };
};

export const runDepthPeelingPipelineCPU = (fragments: number[], layers: number) => {
  const { depths, counts, discovered } = simulateDepthPeeling(fragments, layers);
  const finalCandidates = depths.filter((depth, index) => counts[index] > 0 && depth < 1);
  const combined = dedupeSorted([...discovered, ...finalCandidates]);
  const limit = Math.min(layers, fragments.length);
  return combined.slice(0, limit);
};

export function assert(condition: any, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

export const unreachable = (message: string): never => {
  throw new Error(message);
};

export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

export const lerp = (a: number, b: number, t: number): number => {
  return a * (1 - t) + b * t;
};

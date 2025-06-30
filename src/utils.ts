import { Accessor, createEffect, createSignal } from 'solid-js';

export function assert(condition: any, message?: string): asserts condition {
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

type Box<T> = { value: T };
export const usePrevious = <T>(value: Accessor<T>): Accessor<T | null> => {
  let [prev, setPrev] = createSignal<T | null>(null);
  createEffect<T>((_prev) => {
    setPrev(() => _prev);
    return value();
  });
  return prev;
};
export const useDiff = (
  value: Accessor<number>,
  init = 0
): Accessor<number> => {
  const prevValue = usePrevious(value);
  return () => value() - (prevValue() ?? 0);
};
export const useSmoothedValue = (
  value: Accessor<number>,
  smooth: number
): Accessor<number> => {
  const [_value, setValue] = createSignal(value());
  createEffect<number>((prev) => {
    const next = lerp(prev, value(), smooth);
    setValue(next);
    return next;
  }, value());
  return _value;
};

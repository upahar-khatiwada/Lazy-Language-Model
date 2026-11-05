// Linear position for a value, as a percentage from 0 to 100. A single value spread over no range
// sits at the start. Everything a chart draws or labels goes through one of these.
export function linearPercent(domain: [number, number], value: number): number {
  const [low, high] = domain;
  return high === low ? 0 : ((value - low) / (high - low)) * 100;
}

export const SEQUENTIAL_STEPS = 6;

// Which of the --seq-N steps a value gets, for values from low to high.
export function sequentialStep(low: number, high: number, value: number): number {
  if (high === low) return 0;
  const step = Math.floor(((value - low) / (high - low)) * SEQUENTIAL_STEPS);
  return Math.min(SEQUENTIAL_STEPS - 1, Math.max(0, step));
}

// Four significant digits, without trailing zeros: 0.001, 0.0005, 12.35.
export function shortNumber(value: number): string {
  return String(Number(value.toPrecision(4)));
}

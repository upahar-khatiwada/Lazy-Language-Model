import type { Matrix, Vector } from "./tensor";

export type OptimTensor = Vector | Matrix;
export type TensorMap = Record<string, OptimTensor>;

function isMatrix(tensor: OptimTensor): tensor is Matrix {
  return Array.isArray(tensor[0]);
}

function zip(a: OptimTensor, b: OptimTensor, f: (x: number, y: number) => number): OptimTensor {
  if (isMatrix(a)) {
    const rows = b as Matrix;
    return a.map((row, r) => row.map((value, c) => f(value, rows[r]![c]!)));
  }
  const values = b as Vector;
  return a.map((value, i) => f(value, values[i]!));
}

function squaredSum(tensor: OptimTensor): number {
  const flat = isMatrix(tensor) ? tensor.flat() : tensor;
  return flat.reduce((total, value) => total + value * value, 0);
}

// Mirrors optim.lr_schedule. Linear warmup, then a cosine from lr_max to lr_min. Python raises
// ZeroDivisionError for warmup = 0 at step 0 and for total = warmup past warmup; so does this.
export function lrSchedule(
  step: number,
  warmup: number,
  total: number,
  lrMax: number,
  lrMin: number,
): number {
  if (step <= warmup) {
    if (warmup === 0) throw new Error("lrSchedule: warmup is 0");
    return (lrMax * step) / warmup;
  }
  if (total === warmup) throw new Error("lrSchedule: total equals warmup");
  const frac = (step - warmup) / (total - warmup);
  return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * frac));
}

// Mirrors optim.clip_grad_norm. Python scales in place; this returns new tensors. `norm` is the
// length before clipping.
export function clipGradNorm(grads: TensorMap, maxNorm: number) {
  const norm = Math.sqrt(
    Object.values(grads).reduce((total, tensor) => total + squaredSum(tensor), 0),
  );
  if (!(norm > maxNorm)) return { grads, norm };
  const factor = maxNorm / norm;
  const clipped: TensorMap = {};
  for (const [name, tensor] of Object.entries(grads)) {
    clipped[name] = zip(tensor, tensor, (value) => value * factor);
  }
  return { grads: clipped, norm };
}

export type AdamState = {
  m: TensorMap;
  v: TensorMap;
  t: number;
  b1: number;
  b2: number;
  eps: number;
  wd: number;
};

export type AdamSettings = {
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
};

// Mirrors optim.Adam.__init__, with the same defaults. train.main passes its values from config.py,
// so decay in the pipeline is not the class default of 0.0.
export function createAdamState(
  params: TensorMap,
  { beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.0 }: AdamSettings = {},
): AdamState {
  const zeros: TensorMap = {};
  for (const [name, tensor] of Object.entries(params)) zeros[name] = zip(tensor, tensor, () => 0);
  return { m: zeros, v: zeros, t: 0, b1: beta1, b2: beta2, eps, wd: weightDecay };
}

export type AdamTrace = {
  mhat: OptimTensor;
  vhat: OptimTensor;
  // The update before decay, and the update params[k] uses. The two differ only where ndim >= 2.
  raw: OptimTensor;
  update: OptimTensor;
};

// Mirrors optim.Adam.step. Python updates params and its own state in place; this returns both, and
// the values of its locals per parameter. Decay applies only where ndim >= 2, and outside the moments.
export function adamStep(state: AdamState, params: TensorMap, grads: TensorMap, lr: number) {
  const t = state.t + 1;
  const { b1, b2, eps, wd } = state;
  const m: TensorMap = {};
  const v: TensorMap = {};
  const next: TensorMap = {};
  const trace: Record<string, AdamTrace> = {};

  for (const [name, param] of Object.entries(params)) {
    const g = grads[name]!;
    m[name] = zip(state.m[name]!, g, (old, value) => b1 * old + (1 - b1) * value);
    v[name] = zip(state.v[name]!, g, (old, value) => b2 * old + (1 - b2) * value * value);

    const mhat = zip(m[name]!, m[name]!, (value) => value / (1 - b1 ** t));
    const vhat = zip(v[name]!, v[name]!, (value) => value / (1 - b2 ** t));
    const raw = zip(mhat, vhat, (top, bottom) => top / (Math.sqrt(bottom) + eps));
    const update = isMatrix(param) ? zip(raw, param, (value, weight) => value + wd * weight) : raw;
    next[name] = zip(param, update, (weight, value) => weight - lr * value);
    trace[name] = { mhat, vhat, raw, update };
  }

  return { state: { ...state, m, v, t }, params: next, trace };
}

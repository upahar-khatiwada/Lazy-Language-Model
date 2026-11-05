import { matmul, transpose, type Matrix, type Vector } from "./tensor";

const sum = (values: Vector) => values.reduce((total, value) => total + value, 0);
const mean = (values: Vector) => sum(values) / values.length;
const maximum = (values: Vector) => values.reduce((top, value) => Math.max(top, value), -Infinity);

function sumRows(m: Matrix): Vector {
  const cols = m[0]?.length ?? 0;
  return Array.from({ length: cols }, (_, c) => sum(m.map((row) => row[c]!)));
}

// Mirrors ops.linear_forward. Y = X @ W + b
export function linearForward(X: Matrix, W: Matrix, b: Vector): Matrix {
  return matmul(X, W).map((row) => row.map((value, c) => value + b[c]!));
}

// Mirrors ops.linear_backward. dW = X^T @ dY, db = sum of rows of dY, dX = dY @ W^T
export function linearBackward(X: Matrix, W: Matrix, dY: Matrix) {
  return {
    dX: matmul(dY, transpose(W)),
    dW: matmul(transpose(X), dY),
    db: sumRows(dY),
  };
}

// Mirrors ops.relu_forward. max(0, X), element by element.
export function relu(X: Matrix): Matrix {
  return X.map((row) => row.map((value) => Math.max(0, value)));
}

// Mirrors ops.relu_backward. dA * (Z > 0), so the gradient at Z = 0 is 0.
export function reluBackward(Z: Matrix, dA: Matrix): Matrix {
  return dA.map((row, r) => row.map((value, c) => (Z[r]![c]! > 0 ? value : 0)));
}

// Mirrors ops.softmax_forward. The maximum is subtracted first; without that, exp overflows for
// large inputs and the result is NaN.
export function softmax(x: Vector): Vector {
  const top = maximum(x);
  const e = x.map((value) => Math.exp(value - top));
  const total = sum(e);
  return e.map((value) => value / total);
}

// Mirrors ops.softmax_forward on a matrix: axis=-1 means each row on its own.
export function softmaxRows(x: Matrix): Matrix {
  return x.map(softmax);
}

// Mirrors ops.softmax_backward. y * (dY - sum(y * dY))
export function softmaxBackward(y: Vector, dY: Vector): Vector {
  const inner = sum(y.map((value, i) => value * dY[i]!));
  return y.map((value, i) => value * (dY[i]! - inner));
}

// Mirrors ops.softmax_backward on a matrix: axis=-1 means each row on its own.
export function softmaxBackwardRows(y: Matrix, dY: Matrix): Matrix {
  return y.map((row, r) => softmaxBackward(row, dY[r]!));
}

// Mirrors ops.cross_entropy_forward. z = logits - max, logp = z - log(sum exp z),
// loss = -mean of logp at each target, probs = exp(logp).
export function crossEntropy(logits: Matrix, targets: number[]) {
  const logp = logits.map((row) => {
    const top = maximum(row);
    const z = row.map((value) => value - top);
    const logTotal = Math.log(sum(z.map(Math.exp)));
    return z.map((value) => value - logTotal);
  });
  const picked = targets.map((target, t) => {
    const value = logp[t]?.[target];
    if (value === undefined) throw new Error(`crossEntropy: no logit at (${t}, ${target})`);
    return value;
  });
  return { loss: -mean(picked), probs: logp.map((row) => row.map(Math.exp)) };
}

// Mirrors ops.cross_entropy_backward. dZ = (probs - onehot(targets)) / T
export function crossEntropyBackward(probs: Matrix, targets: number[]): Matrix {
  const T = targets.length;
  return probs.map((row, t) => row.map((value, v) => (value - (v === targets[t] ? 1 : 0)) / T));
}

export type LayerNormCache = { Xhat: Matrix; s: Matrix };

// Mirrors ops.layer_norm_forward. Statistics are per row, across d. s has shape (T, 1), as in Python.
export function layerNormForward(X: Matrix, gamma: Vector, beta: Vector, epsilon = 1e-5) {
  const s: Matrix = [];
  const Xhat = X.map((row) => {
    const mu = mean(row);
    const variance = mean(row.map((value) => (value - mu) ** 2));
    const scale = 1 / Math.sqrt(variance + epsilon);
    s.push([scale]);
    return row.map((value) => (value - mu) * scale);
  });
  const Y = Xhat.map((row) => row.map((value, i) => gamma[i]! * value + beta[i]!));
  return { Y, cache: { Xhat, s } satisfies LayerNormCache };
}

// Mirrors ops.layer_norm_backward.
export function layerNormBackward(dY: Matrix, cache: LayerNormCache, gamma: Vector) {
  const { Xhat, s } = cache;
  const dXhat = dY.map((row) => row.map((value, i) => value * gamma[i]!));
  const dX = dXhat.map((row, t) => {
    const rowMean = mean(row);
    const hat = Xhat[t]!;
    const projection = mean(row.map((value, i) => value * hat[i]!));
    return row.map((value, i) => s[t]![0]! * (value - rowMean - hat[i]! * projection));
  });
  return {
    dX,
    dgamma: sumRows(dY.map((row, t) => row.map((value, i) => value * Xhat[t]![i]!))),
    dbeta: sumRows(dY),
  };
}

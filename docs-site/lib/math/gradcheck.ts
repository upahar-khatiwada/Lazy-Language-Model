import type { Matrix, Vector } from "./tensor";

// Mirrors gradcheck.numerical_grad. Central difference on every entry. Unlike the Python function,
// this works on a copy, so the caller's matrix is never changed, even for a moment.
export function numericalGrad(f: (x: Matrix) => number, x: Matrix, h = 1e-5): Matrix {
  const live = x.map((row) => [...row]);
  return live.map((row) =>
    row.map((_, c) => {
      const old = row[c]!;
      row[c] = old + h;
      const above = f(live);
      row[c] = old - h;
      const below = f(live);
      row[c] = old;
      return (above - below) / (2 * h);
    }),
  );
}

// Mirrors gradcheck.numerical_grad, on a vector.
export function numericalGradVector(f: (x: Vector) => number, x: Vector, h = 1e-5): Vector {
  return numericalGrad((m) => f(m[0]!), [x], h)[0]!;
}

// Mirrors gradcheck.relative_error.
export function relativeError(a: number, b: number, eps = 1e-12): number {
  return Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), eps);
}

// Mirrors the scalar that make_golden.py hands to gradcheck.numerical_grad for a backward function
// that takes dY and no loss: (Y * dY).sum(), whose gradient on Y is exactly dY.
export function outputDot(Y: Matrix, dY: Matrix): number {
  return Y.reduce(
    (total, row, t) => row.reduce((inner, value, i) => inner + value * dY[t]![i]!, total),
    0,
  );
}

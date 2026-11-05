// Plain arrays stand in for NumPy arrays. Every function in lib/math is pure: it never changes its
// arguments, even where the Python function works in place.

export type Vector = number[];
export type Matrix = number[][];
export type Tensor3 = number[][][];

// Mirrors the X.T used in ops.linear_backward.
export function transpose(m: Matrix): Matrix {
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  return Array.from({ length: cols }, (_, c) => Array.from({ length: rows }, (_, r) => m[r]![c]!));
}

// Mirrors the @ operator in ops.linear_forward.
export function matmul(a: Matrix, b: Matrix): Matrix {
  const inner = b.length;
  const cols = b[0]?.length ?? 0;
  return a.map((row) => {
    if (row.length !== inner) {
      throw new Error(`matmul: shapes do not align, ${row.length} columns and ${inner} rows`);
    }
    const out = new Array<number>(cols).fill(0);
    for (let k = 0; k < inner; k++) {
      const left = row[k]!;
      const right = b[k]!;
      for (let c = 0; c < cols; c++) out[c]! += left * right[c]!;
    }
    return out;
  });
}

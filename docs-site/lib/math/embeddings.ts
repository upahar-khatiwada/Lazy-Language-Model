import type { Matrix } from "./tensor";

const zeros = (rows: number, columns: number): Matrix =>
  Array.from({ length: rows }, () => Array.from({ length: columns }, () => 0));

// Mirrors transformer.embed_forward. The two terms are returned too, because a page shows them.
export function embedForward(tokEmb: Matrix, posEmb: Matrix, ids: number[]) {
  if (ids.length > posEmb.length) throw new Error("sequence longer than the position table");
  const tokens = ids.map((id) => [...tokEmb[id]!]);
  const positions = posEmb.slice(0, ids.length).map((row) => [...row]);
  const X0 = tokens.map((row, t) => row.map((value, j) => value + positions[t]![j]!));
  return { tokens, positions, X0 };
}

// Mirrors transformer.embed_backward, starting from zeros: each row of dX0 is added to the table row
// of its id, so a repeated id accumulates, and to the row of its position.
export function embedBackward(dX0: Matrix, ids: number[], V: number, Tmax: number) {
  const d = dX0[0]!.length;
  const tok = zeros(V, d);
  const pos = zeros(Tmax, d);
  dX0.forEach((row, t) => {
    row.forEach((value, j) => {
      tok[ids[t]!]![j]! += value;
      pos[t]![j]! += value;
    });
  });
  return { tok, pos };
}

// Mirrors what transformer.embed_backward avoids: grads[ids] += dX0 with NumPy fancy indexing. Each
// id is read once and written once, so a repeated id keeps only the last of its rows.
export function overwritingBackward(dX0: Matrix, ids: number[], V: number): Matrix {
  const tok = zeros(V, dX0[0]!.length);
  dX0.forEach((row, t) => {
    tok[ids[t]!] = [...row];
  });
  return tok;
}

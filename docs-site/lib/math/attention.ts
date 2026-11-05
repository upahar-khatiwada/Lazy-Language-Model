import { linearForward, softmaxBackwardRows, softmaxRows } from "./ops";
import { matmul, transpose, type Matrix, type Tensor3, type Vector } from "./tensor";

// attention_forward computes these steps inline. They are separate here so that a page can show
// each one; attentionForward below runs them in the same order as the Python function.

// Mirrors transformer.split_heads. (T, d) to (H, T, dh)
export function splitHeads(M: Matrix, H: number): Tensor3 {
  const d = M[0]?.length ?? 0;
  if (d % H !== 0) throw new Error(`splitHeads: ${d} is not divisible by ${H}`);
  const dh = d / H;
  return Array.from({ length: H }, (_, h) => M.map((row) => row.slice(h * dh, (h + 1) * dh)));
}

// Mirrors transformer.merge_heads. (H, T, dh) to (T, d)
export function mergeHeads(Mh: Tensor3): Matrix {
  const T = Mh[0]?.length ?? 0;
  return Array.from({ length: T }, (_, t) => Mh.flatMap((head) => head[t]!));
}

// Mirrors transformer.attention_forward, the product Qh @ Kh^T on line S = ... of that function.
export function attentionScores(Qh: Tensor3, Kh: Tensor3): Tensor3 {
  return Qh.map((head, h) => matmul(head, transpose(Kh[h]!)));
}

// Mirrors transformer.attention_forward, the division by np.sqrt(dh).
export function scaleScores(raw: Tensor3, dh: number): Tensor3 {
  const root = Math.sqrt(dh);
  return raw.map((head) => head.map((row) => row.map((value) => value / root)));
}

// Mirrors transformer.attention_forward, the -inf above the diagonal (np.triu(..., 1)).
export function causalMask(S: Tensor3): Tensor3 {
  return S.map((head) => head.map((row, i) => row.map((value, j) => (j > i ? -Infinity : value))));
}

// Mirrors transformer.attention_forward, A = softmax_forward(S), row by row in each head.
export function attentionWeights(S: Tensor3): Tensor3 {
  return S.map(softmaxRows);
}

// Mirrors transformer.attention_forward, O = merge_heads(A @ Vh).
export function weightedValues(A: Tensor3, Vh: Tensor3): Matrix {
  return mergeHeads(A.map((head, h) => matmul(head, Vh[h]!)));
}

// The parameters of one block, without the "b{l}." prefix the Python dictionary uses.
export type AttentionParams = {
  wq: Matrix;
  bq: Vector;
  wk: Matrix;
  bk: Vector;
  wv: Matrix;
  bv: Vector;
  wo: Matrix;
  bo: Vector;
};

// Mirrors transformer.attention_forward. Also returns the scores before and after the mask, which
// Python overwrites and does not put in its cache.
export function attentionForward(Xn: Matrix, params: AttentionParams, H: number) {
  const d = Xn[0]?.length ?? 0;
  const dh = Math.floor(d / H);
  const Q = linearForward(Xn, params.wq, params.bq);
  const K = linearForward(Xn, params.wk, params.bk);
  const V = linearForward(Xn, params.wv, params.bv);
  const Qh = splitHeads(Q, H);
  const Kh = splitHeads(K, H);
  const Vh = splitHeads(V, H);
  const raw = attentionScores(Qh, Kh);
  const scaled = scaleScores(raw, dh);
  const masked = causalMask(scaled);
  const A = attentionWeights(masked);
  const O = weightedValues(A, Vh);
  const out = linearForward(O, params.wo, params.bo);
  return { Q, K, V, Qh, Kh, Vh, raw, scaled, masked, A, O, out };
}

// Mirrors transformer.attention_backward, the sum over rows in grads[p + "bo"] and the other biases.
function sumRows(m: Matrix): Vector {
  return transpose(m).map((column) => column.reduce((total, value) => total + value, 0));
}

// Mirrors transformer.attention_backward. Python fills grads in place and keeps the intermediates
// local; they are returned here so that a page can show each one. Only A, Xn, Qh, Kh, Vh and O are
// read from the forward pass, the same entries Python reads from its cache.
export function attentionBackward(
  dOut: Matrix,
  cache: { Xn: Matrix; Qh: Tensor3; Kh: Tensor3; Vh: Tensor3; A: Tensor3; O: Matrix },
  params: AttentionParams,
  H: number,
) {
  const { Xn, Qh, Kh, Vh, A, O } = cache;
  const d = dOut[0]?.length ?? 0;
  const root = Math.sqrt(Math.floor(d / H));
  const divide = (m: Matrix) => m.map((row) => row.map((value) => value / root));

  const wo = matmul(transpose(O), dOut);
  const bo = sumRows(dOut);
  const dOh = splitHeads(matmul(dOut, transpose(params.wo)), H);
  const dVh = A.map((head, h) => matmul(transpose(head), dOh[h]!));
  const dA = dOh.map((head, h) => matmul(head, transpose(Vh[h]!)));
  const dS = A.map((head, h) => softmaxBackwardRows(head, dA[h]!));
  const dQh = dS.map((head, h) => divide(matmul(head, Kh[h]!)));
  const dKh = dS.map((head, h) => divide(matmul(transpose(head), Qh[h]!)));
  const dQ = mergeHeads(dQh);
  const dK = mergeHeads(dKh);
  const dV = mergeHeads(dVh);

  const dXn: Matrix = Xn.map((row) => row.map(() => 0));
  const fromInput = (dM: Matrix, W: Matrix) => {
    matmul(dM, transpose(W)).forEach((row, t) =>
      row.forEach((value, c) => {
        dXn[t]![c]! += value;
      }),
    );
    return { w: matmul(transpose(Xn), dM), b: sumRows(dM) };
  };
  const q = fromInput(dQ, params.wq);
  const k = fromInput(dK, params.wk);
  const v = fromInput(dV, params.wv);

  return {
    dOh,
    dVh,
    dA,
    dS,
    dQh,
    dKh,
    dQ,
    dK,
    dV,
    dXn,
    grads: { wo, bo, wq: q.w, bq: q.b, wk: k.w, bk: k.b, wv: v.w, bv: v.b },
  };
}

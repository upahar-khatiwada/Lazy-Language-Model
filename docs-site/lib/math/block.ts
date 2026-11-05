import { attentionBackward, attentionForward, type AttentionParams } from "./attention";
import {
  layerNormBackward,
  layerNormForward,
  linearBackward,
  linearForward,
  relu,
  reluBackward,
  type LayerNormCache,
} from "./ops";
import type { Matrix, Tensor3, Vector } from "./tensor";

// Keys as init_params names them, with the same "b{l}." prefix.
export type FlatParams = Record<string, Matrix | Vector>;
export type Gradients = Record<string, Matrix | Vector>;

// Mirrors the params[...] lookups of a matrix in transformer.ffn_forward.
export function matrixOf(params: FlatParams, key: string): Matrix {
  const value = params[key];
  if (!value) throw new Error(`No parameter named ${key}`);
  return value as Matrix;
}

// Mirrors the params[...] lookups of a vector in transformer.ffn_forward.
export function vectorOf(params: FlatParams, key: string): Vector {
  const value = params[key];
  if (!value) throw new Error(`No parameter named ${key}`);
  return value as Vector;
}

const add = (a: Matrix, b: Matrix): Matrix =>
  a.map((row, t) => row.map((value, c) => value + b[t]![c]!));

export type FfnCache = { Xn: Matrix; Z: Matrix; A: Matrix };

// Mirrors transformer.ffn_forward. Z = Xn @ ff1_w + ff1_b, A = ReLU(Z), out = A @ ff2_w + ff2_b
export function ffnForward(Xn: Matrix, params: FlatParams, p: string) {
  const Z = linearForward(Xn, matrixOf(params, `${p}ff1_w`), vectorOf(params, `${p}ff1_b`));
  const A = relu(Z);
  const out = linearForward(A, matrixOf(params, `${p}ff2_w`), vectorOf(params, `${p}ff2_b`));
  return { out, cache: { Xn, Z, A } satisfies FfnCache };
}

// Mirrors transformer.ffn_backward. Python fills grads in place; here they are returned.
export function ffnBackward(dOut: Matrix, cache: FfnCache, params: FlatParams, p: string) {
  const second = linearBackward(cache.A, matrixOf(params, `${p}ff2_w`), dOut);
  const dZ = reluBackward(cache.Z, second.dX);
  const first = linearBackward(cache.Xn, matrixOf(params, `${p}ff1_w`), dZ);
  return {
    dA: second.dX,
    dZ,
    dXn: first.dX,
    grads: {
      [`${p}ff2_w`]: second.dW,
      [`${p}ff2_b`]: second.db,
      [`${p}ff1_w`]: first.dW,
      [`${p}ff1_b`]: first.db,
    } as Gradients,
  };
}

// The eight attention parameters of one block, without the prefix attentionForward leaves out.
function attentionParams(params: FlatParams, p: string): AttentionParams {
  return {
    wq: matrixOf(params, `${p}wq`),
    bq: vectorOf(params, `${p}bq`),
    wk: matrixOf(params, `${p}wk`),
    bk: vectorOf(params, `${p}bk`),
    wv: matrixOf(params, `${p}wv`),
    bv: vectorOf(params, `${p}bv`),
    wo: matrixOf(params, `${p}wo`),
    bo: vectorOf(params, `${p}bo`),
  };
}

export type AttentionCache = {
  Xn: Matrix;
  Qh: Tensor3;
  Kh: Tensor3;
  Vh: Tensor3;
  A: Tensor3;
  O: Matrix;
};

export type BlockCache = {
  ln1: LayerNormCache;
  att: AttentionCache;
  ln2: LayerNormCache;
  ff: FfnCache;
};

// Mirrors transformer.block_forward. The locals are returned too, because a page shows them.
export function blockForward(X: Matrix, params: FlatParams, l: number, H: number, epsilon: number) {
  const p = `b${l}.`;
  const first = layerNormForward(
    X,
    vectorOf(params, `${p}ln1_g`),
    vectorOf(params, `${p}ln1_b`),
    epsilon,
  );
  const Xn = first.Y;
  const attention = attentionForward(Xn, attentionParams(params, p), H);
  const att = attention.out;
  const X1 = add(X, att);
  const second = layerNormForward(
    X1,
    vectorOf(params, `${p}ln2_g`),
    vectorOf(params, `${p}ln2_b`),
    epsilon,
  );
  const Xn2 = second.Y;
  const feedForward = ffnForward(Xn2, params, p);
  const ff = feedForward.out;
  const { Qh, Kh, Vh, A, O } = attention;
  return {
    Xn,
    att,
    X1,
    Xn2,
    ff,
    Y: add(X1, ff),
    cache: {
      ln1: first.cache,
      att: { Xn, Qh, Kh, Vh, A, O },
      ln2: second.cache,
      ff: feedForward.cache,
    } satisfies BlockCache,
  };
}

// Mirrors transformer.block_backward, in the same order. Python fills grads in place.
export function blockBackward(
  dY: Matrix,
  cache: BlockCache,
  params: FlatParams,
  l: number,
  H: number,
) {
  const p = `b${l}.`;
  const feedForward = ffnBackward(dY, cache.ff, params, p);
  const dXn2 = feedForward.dXn;
  const second = layerNormBackward(dXn2, cache.ln2, vectorOf(params, `${p}ln2_g`));
  const dX1_branch = second.dX;
  const dX1 = add(dY, dX1_branch);
  const attention = attentionBackward(dX1, cache.att, attentionParams(params, p), H);
  const dXn = attention.dXn;
  const first = layerNormBackward(dXn, cache.ln1, vectorOf(params, `${p}ln1_g`));
  const dX_branch = first.dX;
  const grads: Gradients = {
    ...feedForward.grads,
    [`${p}ln2_g`]: second.dgamma,
    [`${p}ln2_b`]: second.dbeta,
    ...Object.fromEntries(
      Object.entries(attention.grads).map(([name, value]) => [`${p}${name}`, value]),
    ),
    [`${p}ln1_g`]: first.dgamma,
    [`${p}ln1_b`]: first.dbeta,
  };
  return { dXn2, dX1_branch, dX1, dXn, dX_branch, dX: add(dX1, dX_branch), grads };
}

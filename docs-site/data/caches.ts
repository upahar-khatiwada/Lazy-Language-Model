import type { CodeTarget } from "@/lib/code-targets";

// What the forward pass leaves for the backward pass. lib/site-verify.ts fails the build when a field
// is not in the line that builds it, or when the reader does not call the named function there.

export type CacheRow = {
  // The expression the reader passes on, as the code writes it.
  name: string;
  holds: { field: string; meaning: string }[];
  built: CodeTarget;
  reader: string;
  read: CodeTarget;
};

const BLOCK_FORWARD = { file: "transformer.py", symbol: "transformer.block_forward" };
const BLOCK_BACKWARD = { file: "transformer.py", symbol: "transformer.block_backward" };
const LOSS = { file: "transformer.py", symbol: "transformer.loss_and_grads" };
const LN_BUILT = {
  file: "ops.py",
  symbol: "ops.layer_norm_forward",
  match: "return gamma * Xhat + beta, (Xhat, s)",
};
const LN_FIELDS = [
  { field: "Xhat", meaning: "The normalised rows." },
  { field: "s", meaning: "One over the root of variance plus epsilon, per row." },
];

export const BLOCK_CACHES: CacheRow[] = [
  {
    name: 'c["ff"]',
    holds: [
      { field: "Xn", meaning: "The input of the feed-forward layer, the output of LayerNorm 2." },
      { field: "Z", meaning: "The output of the first linear layer, before ReLU." },
      { field: "A", meaning: "The output of ReLU." },
    ],
    built: {
      file: "transformer.py",
      symbol: "transformer.ffn_forward",
      match: "dict(Xn=Xn, Z=Z, A=A)",
    },
    reader: "transformer.ffn_backward",
    read: { ...BLOCK_BACKWARD, match: 'dXn2 = ffn_backward(dY, c["ff"], params, p, grads)' },
  },
  {
    name: 'c["ln2"]',
    holds: LN_FIELDS,
    built: LN_BUILT,
    reader: "ops.layer_norm_backward",
    read: {
      ...BLOCK_BACKWARD,
      match: 'dX1_branch, dg, db = layer_norm_backward(dXn2, c["ln2"], params[p + "ln2_g"])',
    },
  },
  {
    name: 'c["att"]',
    holds: [
      { field: "Xn", meaning: "The input of attention, the output of LayerNorm 1." },
      { field: "Qh", meaning: "Queries, per head." },
      { field: "Kh", meaning: "Keys, per head." },
      { field: "Vh", meaning: "Values, per head." },
      { field: "A", meaning: "Attention weights. A different array from the ReLU output above." },
      { field: "O", meaning: "The merged values, before the output projection." },
    ],
    built: {
      file: "transformer.py",
      symbol: "transformer.attention_forward",
      match: "dict(Xn=Xn, Qh=Qh, Kh=Kh, Vh=Vh, A=A, O=O)",
    },
    reader: "transformer.attention_backward",
    read: {
      ...BLOCK_BACKWARD,
      match: 'dXn = attention_backward(dX1, c["att"], params, p, cfg, grads)',
    },
  },
  {
    name: 'c["ln1"]',
    holds: LN_FIELDS,
    built: LN_BUILT,
    reader: "ops.layer_norm_backward",
    read: {
      ...BLOCK_BACKWARD,
      match: 'dX_branch, dg, db = layer_norm_backward(dXn, c["ln1"], params[p + "ln1_g"])',
    },
  },
];

export const MODEL_CACHES: CacheRow[] = [
  {
    name: 'cache["Xf"]',
    holds: [{ field: "Xf", meaning: "The output of the final LayerNorm, the input of the head." }],
    built: {
      file: "transformer.py",
      symbol: "transformer.forward",
      match: "dict(blocks=block_caches, lnf=c_lnf, Xf=Xf)",
    },
    reader: "ops.linear_backward",
    read: { ...LOSS, match: 'dXf, dW, db = linear_backward(cache["Xf"], params["head_w"], dZ)' },
  },
  {
    name: 'cache["lnf"]',
    holds: LN_FIELDS,
    built: LN_BUILT,
    reader: "ops.layer_norm_backward",
    read: {
      ...LOSS,
      match: 'dX, dg, db = layer_norm_backward(dXf, cache["lnf"], params["lnf_g"])',
    },
  },
  {
    name: 'cache["blocks"][l]',
    holds: [
      { field: "ln1", meaning: "The cache of LayerNorm 1." },
      { field: "att", meaning: "The cache of attention." },
      { field: "ln2", meaning: "The cache of LayerNorm 2." },
      { field: "ff", meaning: "The cache of the feed-forward layer." },
    ],
    built: { ...BLOCK_FORWARD, match: "dict(ln1=c_ln1, att=c_att, ln2=c_ln2, ff=c_ff)" },
    reader: "transformer.block_backward",
    read: { ...LOSS, match: 'dX = block_backward(dX, cache["blocks"][l], params, l, cfg, grads)' },
  },
];

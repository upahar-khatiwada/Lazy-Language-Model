import type { CodeTarget } from "@/lib/code-targets";
import type { DimensionName } from "@/lib/dimensions";

// The gradient-flow diagrams, in code order. No line numbers here: each target is resolved from
// data/generated/code-index.json. lib/site-verify.ts fails the build when a target, a call, an inline
// expression, the order or the list of parameters no longer matches the code, and tests compare both
// flows with a traced run of the real functions.

export type FlowStep = {
  id: string;
  // What this step produces, as the code names it.
  output: string;
  label: string;
  dims?: DimensionName[];
  // Exactly one of these: a function called at the target, or an expression written in place.
  call?: string;
  inline?: string;
  target: CodeTarget;
  // Entries of grads this step fills, as data/parameters.ts names them, and the lines that write them.
  writes?: string[];
  writeTargets?: CodeTarget[];
  // The block loop: this step runs once per block, last block first.
  perBlock?: boolean;
  // The whole group of entries a block fills, for the loop in loss_and_grads.
  writesBlock?: boolean;
  note?: string;
};

const LOSS = { file: "transformer.py", symbol: "transformer.loss_and_grads" };
const BLOCK = { file: "transformer.py", symbol: "transformer.block_backward" };
const FFN = { file: "transformer.py", symbol: "transformer.ffn_backward" };
const ATTENTION = { file: "transformer.py", symbol: "transformer.attention_backward" };
const EMBED = { file: "transformer.py", symbol: "transformer.embed_backward" };

// Every statement of loss_and_grads that makes a gradient, or starts the dictionary.
export const MODEL_FLOW: FlowStep[] = [
  {
    id: "forward",
    output: "logits",
    label: "Forward pass",
    dims: ["T", "V"],
    call: "transformer.forward",
    target: { ...LOSS, match: "logits, cache = forward(params, ids, cfg)" },
    note: "Also returns the cache that the backward steps read.",
  },
  {
    id: "loss",
    output: "loss, probs",
    label: "Loss",
    dims: ["T", "V"],
    call: "ops.cross_entropy_forward",
    target: { ...LOSS, match: "loss, probs = cross_entropy_forward(logits, targets)" },
    note: "The shape is that of probs. loss is one number.",
  },
  {
    id: "grads",
    output: "grads",
    label: "Empty gradient dictionary",
    inline: "grads = {k: np.zeros_like(v) for k, v in params.items()}",
    target: { ...LOSS, match: "grads = {k: np.zeros_like(v) for k, v in params.items()}" },
    note: "Not a function call. One array of zeros per parameter, under the same key.",
  },
  {
    id: "dZ",
    output: "dZ",
    label: "Gradient on the logits",
    dims: ["T", "V"],
    call: "ops.cross_entropy_backward",
    target: { ...LOSS, match: "dZ = cross_entropy_backward(probs, targets)" },
  },
  {
    id: "dXf",
    output: "dXf",
    label: "Through the head",
    dims: ["T", "d"],
    call: "ops.linear_backward",
    target: { ...LOSS, match: 'dXf, dW, db = linear_backward(cache["Xf"], params["head_w"], dZ)' },
    writes: ["head_w", "head_b"],
    writeTargets: [
      { ...LOSS, match: 'grads["head_w"] += dW' },
      { ...LOSS, match: 'grads["head_b"] += db' },
    ],
  },
  {
    id: "dX-final",
    output: "dX",
    label: "Through the final LayerNorm",
    dims: ["T", "d"],
    call: "ops.layer_norm_backward",
    target: {
      ...LOSS,
      match: 'dX, dg, db = layer_norm_backward(dXf, cache["lnf"], params["lnf_g"])',
    },
    writes: ["lnf_g", "lnf_b"],
    writeTargets: [
      { ...LOSS, match: 'grads["lnf_g"] += dg' },
      { ...LOSS, match: 'grads["lnf_b"] += db' },
    ],
  },
  {
    id: "dX-block",
    output: "dX",
    label: "Through block",
    dims: ["T", "d"],
    call: "transformer.block_backward",
    target: {
      ...LOSS,
      match: 'dX = block_backward(dX, cache["blocks"][l], params, l, cfg, grads)',
    },
    perBlock: true,
    writesBlock: true,
  },
  {
    id: "embed",
    output: "grads",
    label: "Into the embedding tables",
    call: "transformer.embed_backward",
    target: { ...LOSS, match: "embed_backward(dX, ids, grads)" },
    writes: ["tok_emb", "pos_emb"],
    writeTargets: [
      { ...EMBED, match: 'np.add.at(grads["tok_emb"], ids, dX0)' },
      { ...EMBED, match: 'grads["pos_emb"][: len(ids)] += dX0' },
    ],
  },
];

// Every statement of block_backward that makes a gradient.
export const BLOCK_FLOW: FlowStep[] = [
  {
    id: "dXn2",
    output: "dXn2",
    label: "Through the feed-forward layer",
    dims: ["T", "d"],
    call: "transformer.ffn_backward",
    target: { ...BLOCK, match: 'dXn2 = ffn_backward(dY, c["ff"], params, p, grads)' },
    writes: ["b{l}.ff2_w", "b{l}.ff2_b", "b{l}.ff1_w", "b{l}.ff1_b"],
    writeTargets: [
      { ...FFN, match: 'grads[p + "ff2_w"] += dW2' },
      { ...FFN, match: 'grads[p + "ff2_b"] += db2' },
      { ...FFN, match: 'grads[p + "ff1_w"] += dW1' },
      { ...FFN, match: 'grads[p + "ff1_b"] += db1' },
    ],
  },
  {
    id: "dX1-branch",
    output: "dX1_branch",
    label: "Through LayerNorm 2",
    dims: ["T", "d"],
    call: "ops.layer_norm_backward",
    target: {
      ...BLOCK,
      match: 'dX1_branch, dg, db = layer_norm_backward(dXn2, c["ln2"], params[p + "ln2_g"])',
    },
    writes: ["b{l}.ln2_g", "b{l}.ln2_b"],
    writeTargets: [
      { ...BLOCK, match: 'grads[p + "ln2_g"] += dg' },
      { ...BLOCK, match: 'grads[p + "ln2_b"] += db' },
    ],
  },
  {
    id: "dX1",
    output: "dX1",
    label: "Residual add",
    dims: ["T", "d"],
    inline: "dY + dX1_branch",
    target: { ...BLOCK, match: "dX1 = dY + dX1_branch" },
    note: "Not a function call. dY comes along the identity path, unchanged.",
  },
  {
    id: "dXn",
    output: "dXn",
    label: "Through attention",
    dims: ["T", "d"],
    call: "transformer.attention_backward",
    target: { ...BLOCK, match: 'dXn = attention_backward(dX1, c["att"], params, p, cfg, grads)' },
    writes: [
      "b{l}.wo",
      "b{l}.bo",
      "b{l}.wq",
      "b{l}.bq",
      "b{l}.wk",
      "b{l}.bk",
      "b{l}.wv",
      "b{l}.bv",
    ],
    writeTargets: [
      { ...ATTENTION, match: 'grads[p + "wo"] += c["O"].T @ dOut' },
      { ...ATTENTION, match: 'grads[p + "bo"] += dOut.sum(axis=0)' },
      { ...ATTENTION, match: 'grads[p + "w" + n] += c["Xn"].T @ dM' },
      { ...ATTENTION, match: 'grads[p + "b" + n] += dM.sum(axis=0)' },
    ],
  },
  {
    id: "dX-branch",
    output: "dX_branch",
    label: "Through LayerNorm 1",
    dims: ["T", "d"],
    call: "ops.layer_norm_backward",
    target: {
      ...BLOCK,
      match: 'dX_branch, dg, db = layer_norm_backward(dXn, c["ln1"], params[p + "ln1_g"])',
    },
    writes: ["b{l}.ln1_g", "b{l}.ln1_b"],
    writeTargets: [
      { ...BLOCK, match: 'grads[p + "ln1_g"] += dg' },
      { ...BLOCK, match: 'grads[p + "ln1_b"] += db' },
    ],
  },
  {
    id: "dX",
    output: "dX",
    label: "Residual add",
    dims: ["T", "d"],
    inline: "dX1 + dX_branch",
    target: { ...BLOCK, match: "return dX1 + dX_branch" },
    note: "Not a function call. dX1 comes along the identity path, unchanged.",
  },
];

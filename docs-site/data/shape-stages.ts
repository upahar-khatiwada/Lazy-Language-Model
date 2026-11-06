// Check 5 of verify:content looks up each shape stage by its name, as a local of the function its
// shape targets, or of the concept's source functions. A stage named for display has no such local,
// so it is mapped here: "<concept>/<shape>/<stage>" to a function and the name data/golden/shapes.json
// records there. A stage with neither is listed in the report as not checked.

type Recorded = { fn: string; name: string };

const ATTENTION = "transformer.attention_forward";
const ATTENTION_BACK = "transformer.attention_backward";
const FFN_BACK = "transformer.ffn_backward";
const LOSS = "transformer.loss_and_grads";
const BATCH = "train.batch_loss_and_grads";
const ADAM = "optim.Adam.step";
const CLIP = "optim.clip_grad_norm";
const INIT = "transformer.init_params";
const SAMPLE = "generate.sample_next";

const adamStages = (concept: string, shape: string, key: string): Record<string, Recorded> => ({
  [`${concept}/${shape}/grads[k]`]: { fn: ADAM, name: `grads["${key}"]` },
  [`${concept}/${shape}/self.m[k]`]: { fn: ADAM, name: `self.m["${key}"]` },
  [`${concept}/${shape}/self.v[k]`]: { fn: ADAM, name: `self.v["${key}"]` },
  [`${concept}/${shape}/update`]: { fn: ADAM, name: "update" },
  [`${concept}/${shape}/params[k]`]: { fn: ADAM, name: `params["${key}"]` },
});

export const SHAPE_STAGE_ALIASES: Record<string, Recorded> = {
  // Attention
  "attention.mask/attention-mask/S, scaled": { fn: ATTENTION, name: "S" },
  "attention.mask/attention-mask/S, masked": { fn: ATTENTION, name: "S" },
  "attention.merge/attention-merge/A @ Vh": { fn: "transformer.merge_heads", name: "Mh" },
  "attention.merge/attention-merge/O": { fn: ATTENTION, name: "O" },
  "attention.output/attention-output/wo": { fn: ATTENTION, name: 'params["b0.wo"]' },
  "attention.projections/attention-projection/wq": { fn: ATTENTION, name: 'params["b0.wq"]' },
  "attention.scale/attention-scale/Qh @ Kh^T": { fn: ATTENTION, name: "S" },
  "attention.scores/attention-scores/Qh @ Kh^T": { fn: ATTENTION, name: "S" },
  "attention.split/attention-split/Q": { fn: ATTENTION, name: "Q" },
  "attention.split/attention-split/transpose": { fn: ATTENTION, name: "Qh" },
  "attention.weighted/attention-weighted/A @ Vh": { fn: "transformer.merge_heads", name: "Mh" },
  "notation.heads/heads/Q": { fn: ATTENTION, name: "Q" },
  "notation.heads/heads/Qh": { fn: ATTENTION, name: "Qh" },
  "notation.heads/merged/A @ Vh": { fn: "transformer.merge_heads", name: "Mh" },
  "notation.heads/merged/O": { fn: ATTENTION, name: "O" },
  "ops.softmax_forward/attention/S": { fn: ATTENTION, name: "S" },
  "ops.softmax_forward/attention/A": { fn: ATTENTION, name: "A" },
  "ops.softmax_forward/sampling/p": { fn: SAMPLE, name: "p" },
  "ops.softmax_forward/sampling/logits / temperature": { fn: SAMPLE, name: "logits" },
  "ops.softmax_backward/softmax-backward-attention/dA": { fn: ATTENTION_BACK, name: "dA" },
  "ops.softmax_backward/softmax-backward-attention/dS": { fn: ATTENTION_BACK, name: "dS" },

  // Linear pieces, named after the place they are used
  "ops.linear_forward/attention-output/O": { fn: ATTENTION, name: "O" },
  "ops.linear_forward/attention-output/out": { fn: ATTENTION, name: "out" },
  "ops.linear_forward/first-feed-forward/Xn": { fn: "transformer.ffn_forward", name: "Xn" },
  "ops.linear_forward/first-feed-forward/Z": { fn: "transformer.ffn_forward", name: "Z" },
  "ops.linear_forward/second-feed-forward/A": { fn: "transformer.ffn_forward", name: "A" },
  "ops.linear_forward/second-feed-forward/output": { fn: "transformer.block_forward", name: "ff" },
  "ops.linear_forward/linear-head/Xf": { fn: "transformer.forward", name: "Xf" },
  "ops.linear_forward/linear-head/logits": { fn: "transformer.forward", name: "logits" },
  "notation.matmul/query/Xn": { fn: ATTENTION, name: "Xn" },
  "notation.matmul/query/Q": { fn: ATTENTION, name: "Q" },
  "notation.matmul/feed-forward/Xn": { fn: "transformer.ffn_forward", name: "Xn" },
  "notation.matmul/feed-forward/Z": { fn: "transformer.ffn_forward", name: "Z" },
  "notation.matmul/head/Xf": { fn: "transformer.forward", name: "Xf" },
  "notation.matmul/head/logits": { fn: "transformer.forward", name: "logits" },
  "ops.linear_backward/backward-head/dZ": { fn: LOSS, name: "dZ" },
  "ops.linear_backward/backward-head/dW": { fn: LOSS, name: 'grads["head_w"]' },
  "ops.linear_backward/backward-head/db": { fn: LOSS, name: 'grads["head_b"]' },
  "ops.linear_backward/backward-head/dXf": { fn: LOSS, name: "dXf" },
  "ops.linear_backward/backward-second-feed-forward/dOut": { fn: FFN_BACK, name: "dOut" },
  "ops.linear_backward/backward-second-feed-forward/dW2": {
    fn: FFN_BACK,
    name: 'grads["b0.ff2_w"]',
  },
  "ops.linear_backward/backward-second-feed-forward/db2": {
    fn: FFN_BACK,
    name: 'grads["b0.ff2_b"]',
  },
  "ops.linear_backward/backward-second-feed-forward/dA": { fn: FFN_BACK, name: "dA" },
  "ops.linear_backward/backward-first-feed-forward/dZ": { fn: FFN_BACK, name: "dZ" },
  "ops.linear_backward/backward-first-feed-forward/dW1": {
    fn: FFN_BACK,
    name: 'grads["b0.ff1_w"]',
  },
  "ops.linear_backward/backward-first-feed-forward/db1": {
    fn: FFN_BACK,
    name: 'grads["b0.ff1_b"]',
  },
  "ops.linear_backward/backward-first-feed-forward/dXn": { fn: FFN_BACK, name: "dXn" },
  "ops.relu_forward/relu-in-feed-forward/Z": { fn: "transformer.ffn_forward", name: "Z" },
  "ops.relu_forward/relu-in-feed-forward/A": { fn: "transformer.ffn_forward", name: "A" },
  "ops.relu_backward/relu-backward-in-feed-forward/dA": { fn: FFN_BACK, name: "dA" },
  "ops.relu_backward/relu-backward-in-feed-forward/dZ": { fn: FFN_BACK, name: "dZ" },
  "transformer.ffn_backward/ffn-backward-second/dW2": { fn: FFN_BACK, name: 'grads["b0.ff2_w"]' },
  "transformer.ffn_backward/ffn-backward-second/db2": { fn: FFN_BACK, name: 'grads["b0.ff2_b"]' },
  "transformer.ffn_backward/ffn-backward-first/dW1": { fn: FFN_BACK, name: 'grads["b0.ff1_w"]' },
  "transformer.ffn_backward/ffn-backward-first/db1": { fn: FFN_BACK, name: 'grads["b0.ff1_b"]' },

  // Layer norm, at each place it is used
  "ops.layer_norm_forward/ln-before-attention/X": { fn: "transformer.block_forward", name: "X" },
  "ops.layer_norm_forward/ln-before-attention/Xn": { fn: "transformer.block_forward", name: "Xn" },
  "ops.layer_norm_forward/ln-before-feed-forward/X1": {
    fn: "transformer.block_forward",
    name: "X1",
  },
  "ops.layer_norm_forward/ln-before-feed-forward/Xn2": {
    fn: "transformer.block_forward",
    name: "Xn2",
  },
  "ops.layer_norm_forward/ln-final/X": { fn: "transformer.forward", name: "X" },
  "ops.layer_norm_forward/ln-final/Xf": { fn: "transformer.forward", name: "Xf" },
  "ops.layer_norm_backward/ln-backward-final/dXf": { fn: LOSS, name: "dXf" },
  "ops.layer_norm_backward/ln-backward-final/dX": { fn: LOSS, name: "dX" },
  "ops.layer_norm_backward/ln-backward-final/dg": { fn: LOSS, name: 'grads["lnf_g"]' },
  "ops.layer_norm_backward/ln-backward-final/db": { fn: LOSS, name: 'grads["lnf_b"]' },
  "ops.layer_norm_backward/ln-backward-block/dXn": {
    fn: "transformer.block_backward",
    name: "dXn",
  },
  "ops.layer_norm_backward/ln-backward-block/dX_branch": {
    fn: "transformer.block_backward",
    name: "dX_branch",
  },
  "ops.layer_norm_backward/ln-backward-block/dg": {
    fn: "transformer.block_backward",
    name: 'grads["b0.ln1_g"]',
  },
  "ops.layer_norm_backward/ln-backward-block/db": {
    fn: "transformer.block_backward",
    name: 'grads["b0.ln1_b"]',
  },

  // Returned as an expression, so the function itself has no local of that name
  "transformer.block_forward/block-forward/Y": { fn: "transformer.forward", name: "X" },
  "transformer.block_backward/block-backward/dX": { fn: LOSS, name: "dX" },
  "transformer.ffn_forward/ffn-forward/out": { fn: "transformer.block_forward", name: "ff" },

  // Backward through attention
  "transformer.attention_backward/attention-back-input/dM": { fn: ATTENTION_BACK, name: "dQ" },
  "transformer.attention_backward/attention-back-input/grads wn": {
    fn: ATTENTION_BACK,
    name: 'grads["b0.wq"]',
  },
  "transformer.attention_backward/attention-back-values/dOut": { fn: ATTENTION_BACK, name: "dOut" },

  // Embeddings and parameters
  "embedding.position/position-slice/pos_emb": { fn: LOSS, name: 'grads["pos_emb"]' },
  "embedding.position/position-slice/pos_emb[:T]": {
    fn: "transformer.embed_backward",
    name: "dX0",
  },
  "embedding.token/token-lookup/ids": { fn: "transformer.embed_forward", name: "ids" },
  "embedding.token/token-lookup/tok_emb[ids]": { fn: "transformer.embed_backward", name: "dX0" },
  "embedding.token/token-table/tok_emb": { fn: LOSS, name: 'grads["tok_emb"]' },
  "transformer.embed_forward/embedding-sum/tok_emb[ids]": {
    fn: "transformer.embed_backward",
    name: "dX0",
  },
  "transformer.embed_forward/embedding-sum/pos_emb[:T]": {
    fn: "transformer.embed_backward",
    name: "dX0",
  },
  "transformer.embed_forward/embedding-sum/X0": { fn: "transformer.embed_backward", name: "dX0" },
  'transformer.embed_backward/embed-backward-token/grads["tok_emb"]': {
    fn: LOSS,
    name: 'grads["tok_emb"]',
  },
  'transformer.embed_backward/embed-backward-position/grads["pos_emb"]': {
    fn: LOSS,
    name: 'grads["pos_emb"]',
  },
  "transformer.init_params/init-head/head_w": { fn: LOSS, name: 'grads["head_w"]' },
  "transformer.init_params/init-head/head_b": { fn: LOSS, name: 'grads["head_b"]' },
  "transformer.count_params/feed-forward-weights/ff1_w": { fn: LOSS, name: 'grads["b0.ff1_w"]' },
  "transformer.count_params/feed-forward-weights/ff2_w": { fn: LOSS, name: 'grads["b0.ff2_w"]' },
  "transformer.loss_and_grads/model-backward-head/head_w": { fn: LOSS, name: 'grads["head_w"]' },
  "transformer.loss_and_grads/model-backward-head/head_b": { fn: LOSS, name: 'grads["head_b"]' },
  "transformer.loss_and_grads/model-backward-tables/tok_emb": {
    fn: LOSS,
    name: 'grads["tok_emb"]',
  },
  "transformer.loss_and_grads/model-backward-tables/pos_emb": {
    fn: LOSS,
    name: 'grads["pos_emb"]',
  },
  "ops.xavier_uniform/xavier-attention/w{q,k,v,o}": { fn: LOSS, name: 'grads["b0.wq"]' },
  "ops.xavier_uniform/xavier-feed-forward/ff1_w": { fn: LOSS, name: 'grads["b0.ff1_w"]' },
  "gradcheck.check_params/check-params-position/W = pos_emb": {
    fn: LOSS,
    name: 'grads["pos_emb"]',
  },
  'gradcheck.check_params/check-params-position/grads["pos_emb"]': {
    fn: LOSS,
    name: 'grads["pos_emb"]',
  },

  // Data and training
  "data.make_windows/inputs/train_ids": { fn: "data.make_windows", name: "ids" },
  "data.make_windows/inputs/tr_x": { fn: "data.make_windows", name: "inputs" },
  "data.make_windows/targets/train_ids": { fn: "data.make_windows", name: "ids" },
  "data.make_windows/targets/tr_y": { fn: "data.make_windows", name: "targets" },
  "data.iterate_batches/batches/tr_x": { fn: "data.iterate_batches", name: "inputs" },
  "data.iterate_batches/batches/xb": { fn: BATCH, name: "xb" },
  "data.iterate_batches/batches/x": { fn: BATCH, name: "x" },
  "tokenizer.Tokenizer.encode/encoded-corpus/tok.encode(text)": {
    fn: "data.split_ids",
    name: "ids",
  },
  "tokenizer.Tokenizer.encode/encoded-corpus/ids": { fn: "data.split_ids", name: "ids" },
  "train.batch_loss_and_grads/batch-windows/logits inside loss_and_grads": {
    fn: "transformer.forward",
    name: "logits",
  },
  'train.batch_loss_and_grads/batch-grads/g["head_w"]': { fn: BATCH, name: 'g["head_w"]' },
  'train.batch_loss_and_grads/batch-grads/grads["head_w"]': { fn: BATCH, name: 'grads["head_w"]' },
  "train.validation_loss/validation-windows/logits": {
    fn: "train.validation_loss",
    name: "logits",
  },
  ...adamStages("optim.Adam.step", "adam-matrix", "head_w"),
  ...adamStages("optim.Adam.step", "adam-vector", "head_b"),
  "optim.clip_grad_norm/clip-tensor/grads[k] before": { fn: CLIP, name: 'grads["head_w"]' },
  "optim.clip_grad_norm/clip-tensor/grads[k] after": { fn: CLIP, name: 'grads["head_w"]' },

  // Generation
  "generate.generate/generate-step/x": { fn: "transformer.forward", name: "ids" },
  "generate.generate/generate-step/logits": { fn: "transformer.forward", name: "logits" },
  "generate.generate/generate-step/logits[-1]": { fn: SAMPLE, name: "logits" },
  "sampling.greedy/greedy/logits": { fn: "transformer.forward", name: "logits" },
  "sampling.greedy/greedy/logits[-1]": { fn: SAMPLE, name: "logits" },
  "sampling.temperature/temperature/logits": { fn: SAMPLE, name: "logits" },
  "sampling.temperature/temperature/p": { fn: SAMPLE, name: "p" },
  "sampling.top_k/top-k/p before": { fn: SAMPLE, name: "p" },
  "sampling.top_k/top-k/p after": { fn: SAMPLE, name: "p" },
  "sampling.top_p/top-p/order": { fn: SAMPLE, name: "order" },
  "sampling.top_p/top-p/cum": { fn: SAMPLE, name: "cum" },
  "sampling.top_p/top-p/p": { fn: SAMPLE, name: "p" },
  "sampling.choice/choice/p after top-k": { fn: SAMPLE, name: "p" },
  "sampling.choice/choice/p after top-p": { fn: SAMPLE, name: "p" },
};

// A run without these functions is broken, whatever the mappings above find.
export const SHAPE_RUN_REQUIRED_FUNCTIONS = [INIT, ATTENTION, LOSS, ADAM];

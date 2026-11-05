// What each parameter is for, and the line in init_params that creates it. Names, shapes and initial
// values are not written here: they come from data/derived/init-params.json, which is produced by
// calling init_params. lib/site-verify.ts fails the build when the two sets of names differ.

const line = (match: string) => ({
  file: "transformer.py",
  symbol: "transformer.init_params",
  match,
});

const LN1 = line('p[b + "ln1_g"], p[b + "ln1_b"] = np.ones(d), np.zeros(d)');
const LN2 = line('p[b + "ln2_g"], p[b + "ln2_b"] = np.ones(d), np.zeros(d)');
const WEIGHTS = line('p[b + "w" + n] = xavier_uniform(d, d, rng)');
const BIASES = line('p[b + "b" + n] = np.zeros(d)');
const FF1 = line('p[b + "ff1_w"], p[b + "ff1_b"] = xavier_uniform(d, F, rng), np.zeros(F)');
const FF2 = line('p[b + "ff2_w"], p[b + "ff2_b"] = xavier_uniform(F, d, rng), np.zeros(d)');
const LNF = line('p["lnf_g"], p["lnf_b"] = np.ones(d), np.zeros(d)');
const HEAD = line('p["head_w"], p["head_b"] = xavier_uniform(d, V, rng), np.zeros(V)');

export const PARAMETER_ROLES = {
  tok_emb: {
    role: "Token table: one row per vocabulary id.",
    target: line('"tok_emb": xavier_uniform(V, d, rng)'),
  },
  pos_emb: {
    role: "Position table: one row per position.",
    target: line('"pos_emb": xavier_uniform(cfg.context_len, d, rng)'),
  },
  "b{l}.ln1_g": { role: "LayerNorm 1 scale, before attention.", target: LN1 },
  "b{l}.ln1_b": { role: "LayerNorm 1 shift, before attention.", target: LN1 },
  "b{l}.wq": { role: "Query projection weights.", target: WEIGHTS },
  "b{l}.bq": { role: "Query projection bias.", target: BIASES },
  "b{l}.wk": { role: "Key projection weights.", target: WEIGHTS },
  "b{l}.bk": { role: "Key projection bias.", target: BIASES },
  "b{l}.wv": { role: "Value projection weights.", target: WEIGHTS },
  "b{l}.bv": { role: "Value projection bias.", target: BIASES },
  "b{l}.wo": { role: "Attention output projection weights.", target: WEIGHTS },
  "b{l}.bo": { role: "Attention output projection bias.", target: BIASES },
  "b{l}.ln2_g": { role: "LayerNorm 2 scale, before feed-forward.", target: LN2 },
  "b{l}.ln2_b": { role: "LayerNorm 2 shift, before feed-forward.", target: LN2 },
  "b{l}.ff1_w": { role: "First feed-forward layer weights.", target: FF1 },
  "b{l}.ff1_b": { role: "First feed-forward layer bias.", target: FF1 },
  "b{l}.ff2_w": { role: "Second feed-forward layer weights.", target: FF2 },
  "b{l}.ff2_b": { role: "Second feed-forward layer bias.", target: FF2 },
  lnf_g: { role: "Final LayerNorm scale.", target: LNF },
  lnf_b: { role: "Final LayerNorm shift.", target: LNF },
  head_w: { role: "Linear head weights. A separate matrix, not tied to tok_emb.", target: HEAD },
  head_b: { role: "Linear head bias.", target: HEAD },
} as const satisfies Record<
  string,
  { role: string; target: { file: string; symbol: string; match: string } }
>;

// One line per file in data/generated/code-index.json. lib/site-verify.ts checks both directions.
export const REPO_MAP: Record<string, string> = {
  "config.py": "Hyperparameters as module-level constants, plus two import-time assertions.",
  "tokenizer.py": "BPE tokenizer that works on characters: pre-tokens, merges, encode and decode.",
  "data.py": "Corpus loading, encoding, the split, sliding windows and batches.",
  "ops.py": "Operations with hand-written backward functions.",
  "transformer.py": "Parameters, the forward pass and the backward pass.",
  "optim.py": "Adam with decoupled weight decay, the learning-rate schedule and gradient clipping.",
  "generate.py": "Sampling and the generation loop.",
  "checkpoint.py": "Saves and loads weights, model shape and tokenizer.",
  "train.py": "Training entry point.",
  "talk.py": "Interactive inference entry point.",
  "gradcheck.py": "Finite-difference gradient checker. Only tests use it.",
  "tests/test_checkpoint.py": "Checkpoint tests.",
  "tests/test_data.py": "Data pipeline tests.",
  "tests/test_generate.py": "Sampling tests.",
  "tests/test_gradcheck.py": "Gradient-check tests.",
  "tests/test_ops.py": "Operation tests.",
  "tests/test_optim.py": "Optimizer, schedule and clipping tests.",
  "tests/test_tokenizer.py": "Tokenizer tests.",
  "tests/test_transformer.py": "Model tests.",
};

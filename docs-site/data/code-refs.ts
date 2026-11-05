import type { CodeTarget } from "@/lib/code-targets";

// Lines that pages quote, by id. MDX refers to these ids through CodeStep and CodeLine. Line numbers
// are resolved from data/generated/code-index.json. Within a prefix (train., talk.) the order here
// is the order of the walkthrough; lib/site-verify.ts checks it against the line numbers.

const MAIN = { file: "train.py", symbol: "train.main" };
const TALK = { file: "talk.py", symbol: "talk.main" };
const FORWARD = { file: "transformer.py", symbol: "transformer.forward" };
const ATTENTION = { file: "transformer.py", symbol: "transformer.attention_forward" };
const ATTENTION_BACKWARD = { file: "transformer.py", symbol: "transformer.attention_backward" };
const FFN = { file: "transformer.py", symbol: "transformer.ffn_forward" };
const FFN_BACKWARD = { file: "transformer.py", symbol: "transformer.ffn_backward" };
const BLOCK = { file: "transformer.py", symbol: "transformer.block_forward" };
const INIT = { file: "transformer.py", symbol: "transformer.init_params" };

export const CODE_REFS: Record<string, CodeTarget> = {
  "train.seed": { ...MAIN, match: "rng = np.random.default_rng(config.seed)" },
  "train.corpus": {
    ...MAIN,
    match: "text = load_corpus(config.corpus_path, config.corpus_column, config.max_rows)",
  },
  "train.tokenizer": { ...MAIN, match: "tok = build_tokenizer(text, config.num_merges)" },
  "train.ids": { ...MAIN, match: "ids = encode_corpus(text, tok)" },
  "train.split": { ...MAIN, match: "train_ids, val_ids = split_ids(ids, config.val_fraction)" },
  "train.windows": {
    ...MAIN,
    match: "tr_x, tr_y = make_windows[\\s\\S]*?config\\.stride\\)[\\s\\S]*?config\\.stride\\)",
    regex: true,
  },
  "train.model": {
    ...MAIN,
    match: "cfg = ModelConfig\\([\\s\\S]*?config\\.weight_decay\\)",
    regex: true,
  },
  "train.steps": {
    ...MAIN,
    match:
      "steps_per_epoch = math\\.ceil[\\s\\S]*?total_steps = config\\.epochs \\* steps_per_epoch",
    regex: true,
  },
  "train.batches": {
    ...MAIN,
    match: "for epoch in range\\(config\\.epochs\\):[\\s\\S]*?config\\.batch_size, rng\\):",
    regex: true,
  },
  "train.gradient": { ...MAIN, match: "loss, grads = batch_loss_and_grads(params, xb, yb, cfg)" },
  "train.clip": { ...MAIN, match: "norm = clip_grad_norm(grads, config.grad_clip)" },
  "train.lr": {
    ...MAIN,
    match: "step \\+= 1[\\s\\S]*?config\\.lr_max, config\\.lr_min\\)",
    regex: true,
  },
  "train.adamw": { ...MAIN, match: "opt.step(params, grads, lr)" },
  "train.log": {
    ...MAIN,
    match: 'if step == 1 or step % 50 == 0:[\\s\\S]*?lr \\{lr:\\.5f\\}"\\)',
    regex: true,
  },
  "train.validation": {
    ...MAIN,
    match: 'v = validation_loss[\\s\\S]*?ppl \\{perplexity\\(v\\):\\.1f\\}"\\)',
    regex: true,
  },
  "train.sample": {
    ...MAIN,
    match: "seed_ids = \\[int[\\s\\S]*?repr\\(tok\\.decode\\(out\\)\\)\\)",
    regex: true,
  },
  "train.checkpoint": { ...MAIN, match: "save_checkpoint(prefix, params, cfg, tok)" },

  "talk.load": { ...TALK, match: "params, cfg, tok = load_checkpoint(prefix)" },
  "talk.encode": { ...TALK, match: "ids = tok.encode(prompt)" },
  "talk.generate": {
    ...TALK,
    match: "out = generate\\([\\s\\S]*?config\\.top_p, rng\\)",
    regex: true,
  },
  "talk.print": { ...TALK, match: "print(tok.decode(out[len(ids):]))" },

  "tokenizer.merges-append": {
    file: "tokenizer.py",
    symbol: "tokenizer.train_bpe",
    match: 'merges.append({"pair": best_pair, "merged": merged, "frequency": best_count})',
  },
  "tokenizer.stop": {
    file: "tokenizer.py",
    symbol: "tokenizer.train_bpe",
    match: "if len(pair_freq) == 0:",
  },
  "tokenizer.strict": {
    file: "tokenizer.py",
    symbol: "tokenizer.train_bpe",
    match: "if pair_freq[pair] > best_count:",
  },
  "tokenizer.chars": {
    file: "tokenizer.py",
    symbol: "tokenizer.train_bpe",
    match: "word_splits[word] = list(word)",
  },
  "tokenizer.unk": {
    file: "tokenizer.py",
    symbol: "tokenizer.Tokenizer.encode",
    match: "ids.append(self.stoi.get(token, self.stoi[UNK]))",
  },
  "tokenizer.dedupe": {
    file: "tokenizer.py",
    symbol: "tokenizer.build_tokenizer",
    match: 'if m["merged"] not in seen:',
  },
  "notation.batch-loop": {
    file: "train.py",
    symbol: "train.batch_loss_and_grads",
    match: "for x, y in zip(xb, yb):",
  },

  "linear.q": { ...ATTENTION, match: 'Q = Xn @ params[p+"wq"] + params[p+"bq"]' },
  "linear.k": { ...ATTENTION, match: 'K = Xn @ params[p+"wk"] + params[p+"bk"]' },
  "linear.v": { ...ATTENTION, match: 'V = Xn @ params[p+"wv"] + params[p+"bv"]' },
  "linear.out": { ...ATTENTION, match: 'out = O @ params[p + "wo"] + params[p + "bo"]' },
  "linear.ff1": { ...FFN, match: 'Z = Xn @ params[p + "ff1_w"] + params[p + "ff1_b"]' },
  "linear.ff2": { ...FFN, match: 'return A @ params[p + "ff2_w"] + params[p + "ff2_b"]' },
  "linear.head": { ...FORWARD, match: 'logits = Xf @ params["head_w"] + params["head_b"]' },
  "linear.back-head": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: 'dXf, dW, db = linear_backward(cache["Xf"], params["head_w"], dZ)',
  },
  "linear.back-ff2": {
    ...FFN_BACKWARD,
    match: 'dA, dW2, db2 = linear_backward(c["A"], params[p + "ff2_w"], dOut)',
  },
  "linear.back-ff1": {
    ...FFN_BACKWARD,
    match: 'dXn, dW1, db1 = linear_backward(c["Xn"], params[p + "ff1_w"], dZ)',
  },
  "linear.back-out": {
    ...ATTENTION_BACKWARD,
    match: 'grads\\[p \\+ "wo"\\][\\s\\S]*?params\\[p \\+ "wo"\\]\\.T, H\\)',
    regex: true,
  },
  "linear.back-qkv": {
    ...ATTENTION_BACKWARD,
    match: 'grads\\[p \\+ "w" \\+ n\\][\\s\\S]*?params\\[p \\+ "w" \\+ n\\]\\.T',
    regex: true,
  },
  "update.head": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: 'grads["head_w"] += dW',
  },
  "update.ff": { ...FFN_BACKWARD, match: 'grads[p + "ff1_w"] += dW1' },
  "update.lnf": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: 'grads["lnf_g"] += dg',
  },
  "update.ce-start": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: "dZ = cross_entropy_backward(probs, targets)",
  },
  "update.ff-relu": { ...FFN_BACKWARD, match: 'dZ = relu_backward(c["Z"], dA)' },
  "update.attention-softmax": { ...ATTENTION_BACKWARD, match: 'dS = softmax_backward(c["A"], dA)' },
  "update.attention-q": {
    ...ATTENTION_BACKWARD,
    match: 'dQh = dS @ c["Kh"] / np.sqrt(dh)',
  },

  "xavier.tok": { ...INIT, match: '"tok_emb": xavier_uniform(V, d, rng)' },
  "xavier.head": {
    ...INIT,
    match: 'p["head_w"], p["head_b"] = xavier_uniform(d, V, rng), np.zeros(V)',
  },

  "update.ln2": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: 'grads[p + "ln2_g"] += dg',
  },
  "update.residual": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: "return dX1 + dX_branch",
  },

  "ln.block-first": {
    ...BLOCK,
    match:
      'Xn, c_ln1 = layer_norm_forward(X, params[p + "ln1_g"], params[p + "ln1_b"], cfg.ln_eps)',
  },
  "ln.block-second": {
    ...BLOCK,
    match:
      'Xn2, c_ln2 = layer_norm_forward(X1, params[p + "ln2_g"], params[p + "ln2_b"], cfg.ln_eps)',
  },
  "ln.final": {
    ...FORWARD,
    match: 'Xf, c_lnf = layer_norm_forward(X, params["lnf_g"], params["lnf_b"], cfg.ln_eps)',
  },
  "ln.eps-field": {
    file: "transformer.py",
    symbol: "transformer.ModelConfig",
    match: "ln_eps: float = 1e-5",
  },
  "ln.eps-default": { file: "ops.py", symbol: "ops.layer_norm_forward", match: "epsilon = 1e-5" },
  "ln.eps-config": { file: "config.py", symbol: "config.ln_eps", match: "ln_eps      = 1e-5" },

  "forward.assert": { ...FORWARD, match: "assert len(ids) <= cfg.context_len" },
  "forward.return": { ...FORWARD, match: "return logits, dict(" },
  "outside.loss": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: "loss, probs = cross_entropy_forward(logits, targets)",
  },
  "outside.validation": {
    file: "train.py",
    symbol: "train.validation_loss",
    match: "loss, _ = cross_entropy_forward(logits, y)",
  },
  "outside.sampling": {
    file: "generate.py",
    symbol: "generate.sample_next",
    match: "p = softmax_forward(logits / temperature)",
  },
  "inside.attention-softmax": {
    file: "transformer.py",
    symbol: "transformer.attention_forward",
    match: "A = softmax_forward(S)",
  },
  "inside.head-separate": {
    file: "transformer.py",
    symbol: "transformer.init_params",
    match: 'p["head_w"], p["head_b"] = xavier_uniform(d, V, rng), np.zeros(V)',
  },
};

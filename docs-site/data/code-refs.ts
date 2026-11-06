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
const SAMPLE_NEXT = { file: "generate.py", symbol: "generate.sample_next" };

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

  "schedule.warmup-if": {
    file: "optim.py",
    symbol: "optim.lr_schedule",
    match: "if step <= warmup:",
  },
  "schedule.warmup-return": {
    file: "optim.py",
    symbol: "optim.lr_schedule",
    match: "return lr_max * step / warmup",
  },
  "schedule.config-comment": {
    file: "config.py",
    symbol: "config.lr_max",
    match: "lr_max       = 1e-3",
  },
  "adam.init-moments": {
    file: "optim.py",
    symbol: "optim.Adam.__init__",
    match: "self.m = {k: np.zeros_like(v) for k, v in params.items()}",
  },
  "adam.create": {
    ...MAIN,
    match: "opt = Adam(params, config.beta1, config.beta2, config.adam_eps, config.weight_decay)",
  },
  "adam.first-step": {
    file: "tests/test_optim.py",
    symbol: "tests.test_optim.test_adam_first_step_is_lr_times_sign_of_gradient",
    match: 'assert np.isclose(params["w"][0], -0.01 * np.sign(g), atol=1e-6)',
  },
  "adam.ndim": { file: "optim.py", symbol: "optim.Adam.step", match: "if params[k].ndim >= 2:" },
  "loop.step-count": { ...MAIN, match: "step += 1" },
  "loop.after-gradient": {
    ...MAIN,
    match: "norm = clip_grad_norm[\\s\\S]*?opt\\.step\\(params, grads, lr\\)",
    regex: true,
  },
  "loop.whole": {
    ...MAIN,
    match: 'step = 0[\\s\\S]*?saved checkpoint to \\{prefix\\}\\.\\*"\\)',
    regex: true,
  },
  "loop.shuffle-rng": {
    ...MAIN,
    match: "for xb, yb in iterate_batches(tr_x, tr_y, config.batch_size, rng):",
  },
  "log.vocab": {
    ...MAIN,
    match: 'print(f"V={cfg.vocab_size}  ln V={math.log(cfg.vocab_size):.3f}")',
  },
  "log.step": {
    ...MAIN,
    match: 'print(f"step {step:5d}  loss {loss:.3f}  |g| {norm:.2f}  lr {lr:.5f}")',
  },
  "log.epoch": {
    ...MAIN,
    match: 'print(f"epoch {epoch + 1}: val loss {v:.3f}  ppl {perplexity(v):.1f}")',
  },
  "log.sample": { ...MAIN, match: 'print("  sample:", repr(tok.decode(out)))' },
  "log.saved": { ...MAIN, match: 'print(f"saved checkpoint to {prefix}.*")' },
  "loop.body": {
    ...MAIN,
    match: '# 3\\. the loop[\\s\\S]*?lr \\{lr:\\.5f\\}"\\)',
    regex: true,
  },

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

  "attention.qkv": {
    ...ATTENTION,
    match:
      'Q = Xn @ params\\[p\\+"wq"\\][\\s\\S]*?V = Xn @ params\\[p\\+"wv"\\] \\+ params\\[p\\+"bv"\\]',
    regex: true,
  },
  "attention.split": {
    ...ATTENTION,
    match: "Qh, Kh, Vh = split_heads(Q, H), split_heads(K, H), split_heads(V, H)",
  },
  "attention.scores": { ...ATTENTION, match: "S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)" },
  "attention.mask": {
    ...ATTENTION,
    match: "S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, S)",
  },
  "attention.weighted": { ...ATTENTION, match: "O = merge_heads(A @ Vh)" },
  "attention.width": { ...ATTENTION, match: "dh = d // H" },
  "attention.call": { ...BLOCK, match: "att, c_att = attention_forward(Xn, params, p, cfg)" },
  "attention.residual": { ...BLOCK, match: "X1 = X + att" },
  "attention.back-call": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: 'dXn = attention_backward(dX1, c["att"], params, p, cfg, grads)',
  },
  "attention.back-next": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: 'dX_branch, dg, db = layer_norm_backward(dXn, c["ln1"], params[p + "ln1_g"])',
  },
  "attention.back-residual": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: "dX1 = dY + dX1_branch",
  },
  "attention.future-test": {
    file: "tests/test_transformer.py",
    symbol: "tests.test_transformer.test_future_tokens_do_not_change_earlier_logits",
    match: "assert np.allclose(la[:4], lb[:4])",
  },
  "attention.params": {
    ...INIT,
    match: 'for n in "qkvo":[\\s\\S]*?p\\[b \\+ "b" \\+ n\\] = np\\.zeros\\(d\\)',
    regex: true,
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

  "update.ln1": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: 'grads[p + "ln1_g"] += dg',
  },
  "update.ff2": { ...FFN_BACKWARD, match: 'grads[p + "ff2_w"] += dW2' },
  "embed.back-tok": {
    file: "transformer.py",
    symbol: "transformer.embed_backward",
    match: 'np.add.at(grads["tok_emb"], ids, dX0)',
  },
  "embed.back-pos": {
    file: "transformer.py",
    symbol: "transformer.embed_backward",
    match: 'grads["pos_emb"][: len(ids)] += dX0',
  },

  "embed.tok-table": { ...INIT, match: '"tok_emb": xavier_uniform(V, d, rng)' },
  "embed.pos-table": { ...INIT, match: '"pos_emb": xavier_uniform(cfg.context_len, d, rng)' },
  "embed.back-call": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: "embed_backward(dX, ids, grads)",
  },
  "embed.config-field": {
    file: "transformer.py",
    symbol: "transformer.ModelConfig",
    match: "context_len: int",
  },
  "embed.generate-window": {
    file: "generate.py",
    symbol: "generate.generate",
    match: "logits, _ = forward(params, np.array(ids[-window:]), cfg)",
  },

  "gradcheck.default-h": {
    file: "gradcheck.py",
    symbol: "gradcheck.numerical_grad",
    match: "def numerical_grad(f, x, h = 1e-5):",
  },
  "gradcheck.sample": {
    file: "gradcheck.py",
    symbol: "gradcheck.check_params",
    match: "idx = tuple(int(rng.integers(s)) for s in W.shape)",
  },
  "gradcheck.test-quadratic": {
    file: "tests/test_gradcheck.py",
    symbol: "tests.test_gradcheck.test_numerical_grad_on_quadratic",
    match: "assert np.allclose(g, 2 * x, atol=1e-8)",
  },
  "gradcheck.test-bk": {
    file: "tests/test_gradcheck.py",
    symbol: "tests.test_gradcheck.test_key_bias_gradient_is_zero",
    match: 'assert np.abs(grads[f"b{l}.bk"]).max() < 1e-12',
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

  "block.def": { ...BLOCK, match: "def block_forward(X, params, l: int, cfg: ModelConfig):" },
  "block.ffn-call": { ...BLOCK, match: "ff, c_ff = ffn_forward(Xn2, params, p)" },
  "block.return": { ...BLOCK, match: "return X1 + ff" },
  "block.back-call": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: 'dX = block_backward(dX, cache["blocks"][l], params, l, cfg, grads)',
  },
  "ffn.params-first": {
    ...INIT,
    match: 'p[b + "ff1_w"], p[b + "ff1_b"] = xavier_uniform(d, F, rng), np.zeros(F)',
  },
  "ffn.params-second": {
    ...INIT,
    match: 'p[b + "ff2_w"], p[b + "ff2_b"] = xavier_uniform(F, d, rng), np.zeros(d)',
  },
  "ffn.back-call": {
    file: "transformer.py",
    symbol: "transformer.block_backward",
    match: 'dXn2 = ffn_backward(dY, c["ff"], params, p, grads)',
  },
  "ffn.test-forward": {
    file: "tests/test_transformer.py",
    symbol: "tests.test_transformer.test_ffn_worked_example",
    match: "y, cache = ffn_forward(np.array([[1., -1.]]), params, p)",
  },
  "model.block-loop": {
    ...FORWARD,
    match: "for l in range(cfg.num_layers):",
  },
  "model.key": { ...INIT, match: 'b = f"b{l}." # key' },
  "model.key-block": { ...BLOCK, match: 'p = f"b{l}."' },
  "model.grads-zero": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: "grads = {k: np.zeros_like(v) for k, v in params.items()}",
  },
  "model.reverse": {
    file: "transformer.py",
    symbol: "transformer.loss_and_grads",
    match: "for l in reversed(range(cfg.num_layers)):",
  },
  "model.grads-shape-test": {
    file: "tests/test_transformer.py",
    symbol: "tests.test_transformer.test_grads_have_the_shape_of_params",
    match: "assert grads.keys() == params.keys()",
  },
  "model.overfit-test": {
    file: "tests/test_transformer.py",
    symbol: "tests.test_transformer.test_overfit_one_window",
    match: "assert loss < 0.05",
  },
  "model.batch-add": {
    file: "train.py",
    symbol: "train.batch_loss_and_grads",
    match: "grads[k] += g[k] / B",
  },
  "model.discard-cache-generate": {
    file: "generate.py",
    symbol: "generate.generate",
    match: "logits, _ = forward(params, np.array(ids[-window:]), cfg)",
  },
  "model.discard-cache-validation": {
    file: "train.py",
    symbol: "train.validation_loss",
    match: "logits, _ = forward(params, x, cfg)",
  },
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
  "generation.greedy": {
    ...SAMPLE_NEXT,
    match: "if temperature < 1e-6:[\\s\\S]*?return int\\(np\\.argmax\\(logits\\)\\)",
    regex: true,
  },
  "generation.top-k": {
    ...SAMPLE_NEXT,
    match: "if 0 < top_k < len\\(p\\):[\\s\\S]*?p = p / p\\.sum\\(\\)",
    regex: true,
  },
  "generation.cutoff": { ...SAMPLE_NEXT, match: "cutoff = np.sort(p)[-top_k]" },
  "generation.keep-ties": { ...SAMPLE_NEXT, match: "p = np.where(p >= cutoff, p, 0.0)" },
  "generation.top-p": {
    ...SAMPLE_NEXT,
    match: "if top_p < 1\\.0:[\\s\\S]*?p = q / q\\.sum\\(\\)",
    regex: true,
  },
  "generation.argsort": { ...SAMPLE_NEXT, match: "order = np.argsort(-p)" },
  "generation.prefix": { ...SAMPLE_NEXT, match: "keep = order[: np.searchsorted(cum, top_p) + 1]" },
  "generation.choice": { ...SAMPLE_NEXT, match: "return int(rng.choice(len(p), p=p))" },
  "generation.last-logits": {
    file: "generate.py",
    symbol: "generate.generate",
    match: "ids.append(sample_next(logits[-1], temperature, top_k, top_p, rng))",
  },
  "generation.train-seed": { ...MAIN, match: "seed_ids = [int(i) for i in va_x.iloc[0][:3]]" },
  "generation.train-call": {
    ...MAIN,
    match: "out = generate\\(params, cfg, seed_ids, 30, config\\.seq_len,[\\s\\S]*?rng\\)",
    regex: true,
  },
  "generation.train-rng": { ...MAIN, match: "rng = np.random.default_rng(config.seed)" },
  "generation.talk-rng": { ...TALK, match: "rng = np.random.default_rng(config.seed)" },
  "generation.talk-prompt": {
    ...TALK,
    match: 'print("Type the start of a text and press Enter. Empty line quits.")',
  },
  "generation.talk-quit": { ...TALK, match: 'if prompt.strip() == "":' },
  "generation.mask": {
    ...ATTENTION,
    match: "S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, S)",
  },
  "generation.position": {
    file: "transformer.py",
    symbol: "transformer.embed_forward",
    match: 'params["pos_emb"][: len(ids)]',
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

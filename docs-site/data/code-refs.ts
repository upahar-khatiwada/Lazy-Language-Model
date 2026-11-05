import type { CodeTarget } from "@/lib/code-targets";

// Lines that pages quote, by id. MDX refers to these ids through CodeStep and CodeLine. Line numbers
// are resolved from data/generated/code-index.json. Within a prefix (train., talk.) the order here
// is the order of the walkthrough; lib/site-verify.ts checks it against the line numbers.

const MAIN = { file: "train.py", symbol: "train.main" };
const TALK = { file: "talk.py", symbol: "talk.main" };
const FORWARD = { file: "transformer.py", symbol: "transformer.forward" };

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

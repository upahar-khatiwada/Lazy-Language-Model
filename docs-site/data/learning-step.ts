import type { CodeTarget } from "@/lib/code-targets";

// The stages of one learning step, in the order train.main runs them. Text only: every line number
// and quoted statement is resolved from data/generated, and lib/site-verify.ts checks each link
// against the call graph and the line order.

const MAIN = { file: "train.py", symbol: "train.main" };
const BATCH = { file: "train.py", symbol: "train.batch_loss_and_grads" };
const WHOLE = { file: "transformer.py", symbol: "transformer.loss_and_grads" };

const GRADIENT_LINE: CodeTarget = {
  ...MAIN,
  match: "loss, grads = batch_loss_and_grads(params, xb, yb, cfg)",
};
const ONE_WINDOW: CodeTarget = { ...BATCH, match: "loss, g = loss_and_grads(params, x, y, cfg)" };

// One hop from a call site to what runs there: a function, or an expression written in that line.
export type LearningLink = { target: CodeTarget; call?: string; inline?: string };

export type LearningStage = {
  id: string;
  label: string;
  page: string;
  summary: string;
  // First hop: the line of train.main. Later hops: the path down to this stage's own line.
  links: LearningLink[];
  // Another line of train.main that belongs to this stage.
  alsoCalls?: { target: CodeTarget; call: string; note: string };
  // The function whose source the detail panel shows.
  showSymbol: string;
};

export const LEARNING_STAGES: LearningStage[] = [
  {
    id: "text",
    label: "Input text",
    page: "mathematics/data-pipeline",
    summary:
      "`load_corpus` reads one column of the CSV file and joins its rows into a single string, once, before the loop starts.",
    links: [
      {
        target: {
          ...MAIN,
          match: "text = load_corpus(config.corpus_path, config.corpus_column, config.max_rows)",
        },
        call: "data.load_corpus",
      },
    ],
    showSymbol: "data.load_corpus",
  },
  {
    id: "tokens",
    label: "Tokenization",
    page: "mathematics/tokenization",
    summary:
      "`build_tokenizer` learns merges from the text, and `encode_corpus` turns the text into one long array of token ids.",
    links: [
      { target: { ...MAIN, match: "ids = encode_corpus(text, tok)" }, call: "data.encode_corpus" },
    ],
    alsoCalls: {
      target: { ...MAIN, match: "tok = build_tokenizer(text, config.num_merges)" },
      call: "tokenizer.build_tokenizer",
      note: "Builds the tokenizer on the line before.",
    },
    showSymbol: "data.encode_corpus",
  },
  {
    id: "windows",
    label: "Input and target windows",
    page: "mathematics/data-pipeline",
    summary:
      "`make_windows` cuts the ids into windows. The targets are the inputs moved one token to the left, so each position is asked to predict the next token.",
    links: [
      {
        target: {
          ...MAIN,
          match: "tr_x, tr_y = make_windows(train_ids, config.seq_len, config.stride)",
        },
        call: "data.make_windows",
      },
    ],
    alsoCalls: {
      target: {
        ...MAIN,
        match: "for xb, yb in iterate_batches(tr_x, tr_y, config.batch_size, rng):",
      },
      call: "data.iterate_batches",
      note: "Inside the loop, hands out batch_size windows at a time.",
    },
    showSymbol: "data.make_windows",
  },
  {
    id: "forward",
    label: "Forward pass",
    page: "mathematics/full-model",
    summary:
      "The model has no batch dimension, so `batch_loss_and_grads` calls `loss_and_grads` once per window, and that calls `forward`. `forward` returns logits, one row of scores per position.",
    links: [
      { target: GRADIENT_LINE, call: "train.batch_loss_and_grads" },
      { target: ONE_WINDOW, call: "transformer.loss_and_grads" },
      {
        target: { ...WHOLE, match: "logits, cache = forward(params, ids, cfg)" },
        call: "transformer.forward",
      },
    ],
    showSymbol: "transformer.forward",
  },
  {
    id: "probabilities",
    label: "Predicted probabilities",
    page: "mathematics/softmax-cross-entropy",
    summary:
      "`cross_entropy_forward` computes log probabilities from the logits and returns their exponential as `probs`, one distribution over the vocabulary per position.",
    links: [
      { target: GRADIENT_LINE, call: "train.batch_loss_and_grads" },
      { target: ONE_WINDOW, call: "transformer.loss_and_grads" },
      {
        target: { ...WHOLE, match: "loss, probs = cross_entropy_forward(logits, targets)" },
        call: "ops.cross_entropy_forward",
      },
    ],
    showSymbol: "ops.cross_entropy_forward",
  },
  {
    id: "loss",
    label: "Loss",
    page: "mathematics/softmax-cross-entropy",
    summary:
      "The same call also returns the loss: the mean of -log `p` over the window's positions, where `p` is the probability given to the target token. `batch_loss_and_grads` divides each window's loss by `B` and adds it up.",
    links: [
      { target: GRADIENT_LINE, call: "train.batch_loss_and_grads" },
      { target: { ...BATCH, match: "total_loss += loss / B" }, inline: "total_loss += loss / B" },
    ],
    showSymbol: "train.batch_loss_and_grads",
  },
  {
    id: "backward",
    label: "Backward pass",
    page: "mathematics/full-model",
    summary:
      "`loss_and_grads` runs the hand-written backward functions: cross-entropy, the head, the final LayerNorm, the blocks from last to first, and the embeddings.",
    links: [
      { target: GRADIENT_LINE, call: "train.batch_loss_and_grads" },
      { target: ONE_WINDOW, call: "transformer.loss_and_grads" },
      {
        target: { ...WHOLE, match: "dZ = cross_entropy_backward(probs, targets)" },
        call: "ops.cross_entropy_backward",
      },
    ],
    showSymbol: "transformer.loss_and_grads",
  },
  {
    id: "gradients",
    label: "Gradients",
    page: "training/training-loop",
    summary:
      "Each window gives one gradient per parameter. `batch_loss_and_grads` divides each by `B` and adds them, so `train.main` receives the mean gradient of the batch, under the same keys as `params`.",
    links: [
      { target: GRADIENT_LINE, call: "train.batch_loss_and_grads" },
      { target: { ...BATCH, match: "grads[k] += g[k] / B" }, inline: "grads[k] += g[k] / B" },
    ],
    showSymbol: "train.batch_loss_and_grads",
  },
  {
    id: "clipping",
    label: "Gradient clipping",
    page: "training/adamw-learning-rate-clipping",
    summary:
      "`clip_grad_norm` measures the length of all gradients together. If it is above `grad_clip`, every tensor is scaled by the same factor. It returns the length from before clipping.",
    links: [
      {
        target: { ...MAIN, match: "norm = clip_grad_norm(grads, config.grad_clip)" },
        call: "optim.clip_grad_norm",
      },
    ],
    showSymbol: "optim.clip_grad_norm",
  },
  {
    id: "schedule",
    label: "Learning-rate schedule",
    page: "training/adamw-learning-rate-clipping",
    summary:
      "`step` goes up by one first, so the schedule starts from step 1. `lr_schedule` gives the learning rate for this step: a linear warmup, then a cosine down to `lr_min`.",
    links: [
      {
        target: { ...MAIN, match: "lr = lr_schedule(step, config.warmup_steps, total_steps," },
        call: "optim.lr_schedule",
      },
    ],
    showSymbol: "optim.lr_schedule",
  },
  {
    id: "adamw",
    label: "AdamW",
    page: "training/adamw-learning-rate-clipping",
    summary:
      "`opt.step` updates the moving averages `m` and `v` and computes one update per parameter. Parameters with two or more dimensions also get weight decay.",
    links: [{ target: { ...MAIN, match: "opt.step(params, grads, lr)" }, call: "optim.Adam.step" }],
    showSymbol: "optim.Adam.step",
  },
  {
    id: "updated",
    label: "Updated parameters",
    page: "training/adamw-learning-rate-clipping",
    summary:
      "The last line of `opt.step` changes each array in place and returns nothing. The next batch in `train.main` reads the new values from the same `params` dictionary.",
    links: [
      { target: { ...MAIN, match: "opt.step(params, grads, lr)" }, call: "optim.Adam.step" },
      {
        target: { file: "optim.py", symbol: "optim.Adam.step", match: "params[k] -= lr * update" },
        inline: "params[k] -= lr * update",
      },
    ],
    showSymbol: "optim.Adam.step",
  },
  {
    id: "repeat",
    label: "Repeat",
    page: "training/training-loop",
    summary:
      "The batch loop takes the next batch and runs from the forward pass again, with the updated parameters. The epoch loop repeats that for `epochs` passes. Text, tokenization and windows are not repeated.",
    links: [
      {
        target: {
          ...MAIN,
          match: "for xb, yb in iterate_batches(tr_x, tr_y, config.batch_size, rng):",
        },
        call: "data.iterate_batches",
      },
    ],
    showSymbol: "train.main",
  },
];

// Every print in train.main, in file order. Output text comes from the golden case, by index.
export const LOG_LINES: { ref: string; when: string; numbers: string }[] = [
  {
    ref: "log.vocab",
    when: "Once, before training",
    numbers:
      "`V` is the vocabulary size. `ln V` is the loss of a model that gives every token the same probability, which is a first number to compare `loss` with.",
  },
  {
    ref: "log.step",
    when: "At step 1, and at every step divisible by 50",
    numbers:
      "`step` counts updates from 1. `loss` is this batch's mean cross-entropy, before this step's update. `|g|` is the gradient's length before clipping. `lr` is this step's learning rate.",
  },
  {
    ref: "log.epoch",
    when: "After each epoch",
    numbers:
      "The epoch counts from 1, because the code prints `epoch + 1`. `val loss` is the mean cross-entropy over all validation windows. `ppl` is `exp` of that value.",
  },
  {
    ref: "log.sample",
    when: "After each epoch, on the line after validation",
    numbers: "No numbers: the decoded seed and generated tokens.",
  },
  {
    ref: "log.saved",
    when: "Once, after the last epoch",
    numbers: "No numbers: the prefix of the three files.",
  },
];

// The numbers the walk-through prints. Meaning text only: values come from a real step (golden).
export type LogRow = {
  printed: string;
  code: string;
  meaning: string;
};

export const LOG_ROWS: Record<"loss" | "perplexity" | "norm" | "lr", LogRow> = {
  loss: {
    printed: "loss",
    code: "loss",
    meaning:
      "The mean cross-entropy of this batch, computed before this step's update. It is one batch, so it moves up and down from step to step. A lower value means the model gave the target tokens higher probabilities.",
  },
  perplexity: {
    printed: "ppl",
    code: "perplexity(v)",
    meaning:
      "exp of the validation loss, printed once per epoch and read by nothing else. In general, it can be read as the number of equally likely tokens that would give the same loss. A model that gives every token the same probability has `ppl` equal to `V`.",
  },
  norm: {
    printed: "|g|",
    code: "norm",
    meaning:
      "The length of the mean gradient, measured over all parameters at once, before clipping. So it can be above `grad_clip`. When it is, the update used the clipped gradient, not the printed length.",
  },
  lr: {
    printed: "lr",
    code: "lr",
    meaning:
      "The learning rate used for this step, from `lr_schedule`. It is printed with five decimals, so very small values lose digits.",
  },
};

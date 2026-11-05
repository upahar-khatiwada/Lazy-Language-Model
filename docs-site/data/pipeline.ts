import type { CodeTarget } from "@/lib/code-targets";
import type { DimensionName } from "@/lib/dimensions";

// The real pipeline, in code order: contract/ARCHITECTURE.md sections 3 to 6. No line numbers here:
// each target is resolved from data/generated/code-index.json. lib/site-verify.ts fails the build
// when a target, a call, an inline expression, a page or the order no longer matches the code.

export type PipelineNode = {
  id: string;
  label: string;
  // Route of the page that explains this step. Shown as disabled until that page is written.
  page: string;
  dims?: DimensionName[];
  target: CodeTarget;
  // Exactly one is set in forward phases: what this box is in the code.
  call?: string; // a function called at the target, as named in call-graph.json
  inline?: string; // an expression written in place, which appears in the target's lines
  note?: string;
};

export type PipelinePhase = {
  id: string;
  title: string;
  // True for the phases inside transformer.forward.
  forward: boolean;
  // The nodes run once per value of this dimension, through this call.
  repeat?: { dimension: DimensionName; call: string; target: CodeTarget };
  nodes: PipelineNode[];
};

const MAIN = { file: "train.py", symbol: "train.main" };
const BLOCK = { file: "transformer.py", symbol: "transformer.block_forward" };
const FORWARD = { file: "transformer.py", symbol: "transformer.forward" };
const LOSS = { file: "transformer.py", symbol: "transformer.loss_and_grads" };

export const PIPELINE: PipelinePhase[] = [
  {
    id: "data",
    title: "Text to training windows",
    forward: false,
    nodes: [
      {
        id: "text",
        label: "Text",
        page: "mathematics/data-pipeline",
        call: "data.load_corpus",
        target: { ...MAIN, match: "text = load_corpus(" },
      },
      {
        id: "tokenizer",
        label: "Tokenizer",
        page: "mathematics/tokenization",
        call: "tokenizer.build_tokenizer",
        target: { ...MAIN, match: "tok = build_tokenizer(" },
      },
      {
        id: "ids",
        label: "Ids",
        page: "mathematics/tokenization",
        call: "data.encode_corpus",
        target: { ...MAIN, match: "ids = encode_corpus(" },
      },
      {
        id: "windows",
        label: "Windows",
        page: "mathematics/data-pipeline",
        call: "data.make_windows",
        target: { ...MAIN, match: "tr_x, tr_y = make_windows(" },
      },
    ],
  },
  {
    id: "embeddings",
    title: "Embeddings",
    forward: true,
    nodes: [
      {
        id: "embeddings",
        label: "Token plus position embeddings",
        page: "mathematics/embeddings",
        dims: ["T", "d"],
        call: "transformer.embed_forward",
        target: { ...FORWARD, match: "X = embed_forward(params, ids)" },
      },
    ],
  },
  {
    id: "blocks",
    title: "Blocks",
    forward: true,
    repeat: {
      dimension: "L",
      call: "transformer.block_forward",
      target: { ...FORWARD, match: "X, c = block_forward(X, params, l, cfg)" },
    },
    nodes: [
      {
        id: "ln1",
        label: "LayerNorm 1",
        page: "mathematics/layernorm",
        dims: ["T", "d"],
        call: "ops.layer_norm_forward",
        target: { ...BLOCK, match: "Xn, c_ln1 = layer_norm_forward(" },
      },
      {
        id: "attention",
        label: "Attention",
        page: "mathematics/attention",
        dims: ["T", "d"],
        call: "transformer.attention_forward",
        target: { ...BLOCK, match: "att, c_att = attention_forward(Xn, params, p, cfg)" },
      },
      {
        id: "residual1",
        label: "Residual add",
        page: "mathematics/transformer-block",
        dims: ["T", "d"],
        inline: "X + att",
        target: { ...BLOCK, match: "X1 = X + att" },
      },
      {
        id: "ln2",
        label: "LayerNorm 2",
        page: "mathematics/layernorm",
        dims: ["T", "d"],
        call: "ops.layer_norm_forward",
        target: { ...BLOCK, match: "Xn2, c_ln2 = layer_norm_forward(" },
      },
      {
        id: "ffn",
        label: "Feed-forward",
        page: "mathematics/transformer-block",
        dims: ["T", "d"],
        call: "transformer.ffn_forward",
        target: { ...BLOCK, match: "ff, c_ff = ffn_forward(Xn2, params, p)" },
      },
      {
        id: "residual2",
        label: "Residual add",
        page: "mathematics/transformer-block",
        dims: ["T", "d"],
        inline: "X1 + ff",
        target: { ...BLOCK, match: "return X1 + ff" },
      },
    ],
  },
  {
    id: "output",
    title: "Output",
    forward: true,
    nodes: [
      {
        id: "final-ln",
        label: "Final LayerNorm",
        page: "mathematics/full-model",
        dims: ["T", "d"],
        call: "ops.layer_norm_forward",
        target: { ...FORWARD, match: "Xf, c_lnf = layer_norm_forward(" },
      },
      {
        id: "head",
        label: "Linear head",
        page: "mathematics/full-model",
        dims: ["T", "V"],
        inline: 'Xf @ params["head_w"] + params["head_b"]',
        target: { ...FORWARD, match: 'logits = Xf @ params["head_w"] + params["head_b"]' },
      },
      {
        id: "logits",
        label: "Logits",
        page: "mathematics/full-model",
        dims: ["T", "V"],
        inline: "return logits",
        target: { ...FORWARD, match: "return logits, dict(" },
      },
    ],
  },
  {
    id: "update",
    title: "Loss and update",
    forward: false,
    nodes: [
      {
        id: "loss",
        label: "Softmax and cross-entropy",
        page: "mathematics/softmax-cross-entropy",
        call: "ops.cross_entropy_forward",
        target: { ...LOSS, match: "loss, probs = cross_entropy_forward(logits, targets)" },
      },
      {
        id: "backward",
        label: "Backward",
        page: "training/how-the-model-learns",
        call: "ops.cross_entropy_backward",
        target: {
          ...LOSS,
          match: "dZ = cross_entropy_backward[\\s\\S]*embed_backward\\(dX, ids, grads\\)",
          regex: true,
        },
      },
      {
        id: "clipping",
        label: "Clipping",
        page: "training/adamw-learning-rate-clipping",
        call: "optim.clip_grad_norm",
        target: { ...MAIN, match: "norm = clip_grad_norm(grads, config.grad_clip)" },
      },
      {
        id: "adamw",
        label: "AdamW with schedule",
        page: "training/adamw-learning-rate-clipping",
        call: "optim.lr_schedule",
        target: {
          ...MAIN,
          match: "lr = lr_schedule\\([\\s\\S]*?opt\\.step\\(params, grads, lr\\)",
          regex: true,
        },
      },
    ],
  },
  {
    id: "generation",
    title: "Generation",
    forward: false,
    nodes: [
      {
        id: "generation",
        label: "Generation",
        page: "generation/autoregressive-generation",
        dims: ["V"],
        call: "generate.sample_next",
        target: {
          file: "generate.py",
          symbol: "generate.generate",
          match: "ids.append(sample_next(logits[-1]",
        },
        note: "Samples from the last row of logits.",
      },
    ],
  },
];

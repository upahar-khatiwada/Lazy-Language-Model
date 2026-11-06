// Which config.py constants the homepage shows. Values are never written here: lib/model-glance.ts
// reads them from data/generated/config.json.
export type GlanceRow = { label: string; config: string; note?: string };
export type GlanceGroup = { title: string; rows: GlanceRow[] };

export const GLANCE: GlanceGroup[] = [
  {
    title: "Model",
    rows: [
      { label: "Layers", config: "num_layers" },
      { label: "Heads", config: "num_heads" },
      { label: "Width", config: "emb_dim" },
      { label: "Head width", config: "head_dim" },
      { label: "FFN width", config: "ff_dim" },
      { label: "Window length", config: "seq_len" },
      { label: "Context length", config: "context_len" },
    ],
  },
  {
    title: "Tokenizer",
    rows: [{ label: "BPE merges", config: "num_merges" }],
  },
  {
    title: "Optimizer",
    rows: [
      { label: "Largest learning rate", config: "lr_max" },
      { label: "Smallest learning rate", config: "lr_min" },
      { label: "Warmup steps", config: "warmup_steps" },
      { label: "Beta 1", config: "beta1" },
      { label: "Beta 2", config: "beta2" },
      { label: "Adam epsilon", config: "adam_eps" },
      { label: "Weight decay", config: "weight_decay" },
      { label: "Gradient clip", config: "grad_clip" },
    ],
  },
  {
    title: "Sampling defaults",
    rows: [
      { label: "Temperature", config: "temperature" },
      { label: "Top-k", config: "top_k", note: "0 turns it off." },
      { label: "Top-p", config: "top_p" },
      { label: "New tokens in talk.py", config: "max_new_tokens" },
    ],
  },
];

export const QUESTIONS: { question: string; page: string }[] = [
  { question: "What does this LLM do?", page: "getting-started/how-this-llm-works" },
  { question: "How does attention work?", page: "mathematics/attention" },
  { question: "What is the equation?", page: "reference/mathematics" },
  {
    question: "Can I calculate it with a tiny example?",
    page: "mathematics/softmax-cross-entropy",
  },
  { question: "Where is it implemented?", page: "code/math-code" },
  { question: "Show me the actual Python code.", page: "code" },
  { question: "How does the gradient flow through it?", page: "training/how-the-model-learns" },
  { question: "How does this fit into the whole model?", page: "getting-started/architecture" },
];

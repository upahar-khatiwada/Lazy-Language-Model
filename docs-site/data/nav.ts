export type NavPage = {
  title: string;
  // Route and content path: /<slug>/ and content/<slug>.mdx. A page is disabled until that file exists.
  slug: string;
  // Path in the repository. tests/nav.test.ts checks it against data/generated/code-index.json.
  sourceFile?: string;
};

export type NavGroup = {
  title: string;
  pages: NavPage[];
};

export const NAV: NavGroup[] = [
  {
    title: "Getting Started",
    pages: [
      { title: "Introduction", slug: "getting-started/introduction" },
      { title: "How This LLM Works", slug: "getting-started/how-this-llm-works" },
      { title: "Architecture", slug: "getting-started/architecture" },
    ],
  },
  {
    title: "Mathematics",
    pages: [
      { title: "Notation & Shapes", slug: "mathematics/notation-and-shapes" },
      { title: "Tokenization", slug: "mathematics/tokenization", sourceFile: "tokenizer.py" },
      { title: "Data Pipeline", slug: "mathematics/data-pipeline", sourceFile: "data.py" },
      {
        title: "Linear / ReLU / Initialisation",
        slug: "mathematics/linear-relu-initialisation",
        sourceFile: "ops.py",
      },
      {
        title: "Softmax & Cross-Entropy",
        slug: "mathematics/softmax-cross-entropy",
        sourceFile: "ops.py",
      },
      { title: "LayerNorm", slug: "mathematics/layernorm", sourceFile: "ops.py" },
      { title: "Embeddings", slug: "mathematics/embeddings", sourceFile: "transformer.py" },
      { title: "Attention", slug: "mathematics/attention", sourceFile: "transformer.py" },
      {
        title: "Transformer Block",
        slug: "mathematics/transformer-block",
        sourceFile: "transformer.py",
      },
      { title: "Full Model", slug: "mathematics/full-model", sourceFile: "transformer.py" },
    ],
  },
  {
    title: "Training",
    pages: [
      {
        title: "How the Model Learns",
        slug: "training/how-the-model-learns",
        sourceFile: "transformer.py",
      },
      {
        title: "AdamW / Learning Rate / Clipping",
        slug: "training/adamw-learning-rate-clipping",
        sourceFile: "optim.py",
      },
      { title: "Training Loop", slug: "training/training-loop", sourceFile: "train.py" },
    ],
  },
  {
    title: "Generation",
    pages: [
      { title: "Sampling", slug: "generation/sampling", sourceFile: "generate.py" },
      {
        title: "Autoregressive Generation",
        slug: "generation/autoregressive-generation",
        sourceFile: "generate.py",
      },
    ],
  },
  {
    title: "Code",
    pages: [
      { title: "Repository", slug: "code/repository" },
      { title: "Code Explorer", slug: "code" },
      { title: "Math ↔ Code", slug: "code/math-code" },
      { title: "Checkpoints & Tests", slug: "code/checkpoints-tests", sourceFile: "checkpoint.py" },
    ],
  },
  {
    // Decision D2 (contract/DISCREPANCIES.md): test-only mathematics stays out of the pipeline flow.
    title: "Testing",
    pages: [
      { title: "Gradient Checking", slug: "testing/gradient-checking", sourceFile: "gradcheck.py" },
    ],
  },
  {
    title: "Reference",
    pages: [
      { title: "Mathematics Reference", slug: "reference/mathematics" },
      { title: "Not in This Model", slug: "reference/not-in-this-model" },
      { title: "Sources & Discrepancies", slug: "reference/sources-discrepancies" },
    ],
  },
];

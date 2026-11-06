// The words verify:content scans for. Check 7 fails if a confirmed-absent row of
// contract/PDF_VS_CODE.md section 3 (data/derived/absent-techniques.json) has no entry here, or if an
// entry here has no such row, so this list cannot drift from the contract. Keys are the contract's
// own technique names. The words are chosen to name the technique and not ordinary code: "padding"
// alone would match CSS.

export const ABSENT_TERMS: Record<string, string[]> = {
  "Post-LN": ["post-LN", "post-layer-norm", "post-norm"],
  BatchNorm: ["BatchNorm", "batch norm", "batch normalization", "batch normalisation"],
  Dropout: ["dropout"],
  "Label smoothing": ["label smoothing"],
  "Sinusoidal positions": ["sinusoidal"],
  "Weight tying": ["weight tying", "tied weights", "weight-tied"],
  GELU: ["GELU"],
  "Upstream epsilon loss": ["upstream epsilon"],
  "N(0, 0.02^2) embedding init": ["N(0, 0.02", "0.02^2"],
  "Lower-casing": ["lower-casing", "lower-cases", "lower-cased", "lowercasing"],
  "Pre-token encoding cache": ["encoding cache", "pre-token cache"],
  "Pad, BOS, EOS tokens": ["pad token", "padding token", "padding, start", "BOS", "EOS"],
  "L2 penalty in the loss": ["L2 penalty", "L2 regularization", "L2 regularisation"],
  "Plain gradient descent": ["gradient descent"],
  "Beam search, repetition penalty": ["beam search", "repetition penalty"],
  "Six-rung ladder of checks": ["six-rung"],
  "Stop when validation rises": ["early stopping", "stop when validation"],
  "ELIZA chat (`chat.py`)": ["ELIZA", "chat.py"],
  "XOR network (`xor_net.py`)": ["XOR", "xor_net"],
  "Skip-gram (`skipgram.py`)": ["skip-gram", "skipgram"],
  "Instruction following": ["instruction following"],
};

// Task 16 names these as examples of what the scan must catch, but contract/PDF_VS_CODE.md has no
// row for them, so they are scanned and do not appear on the Not in This Model page. Check 7 still
// proves them absent from the repository's Python files.
export const TASK_NAMED_TERMS: Record<string, string[]> = {
  "Rotary positions": ["rotary"],
  RMSNorm: ["RMSNorm"],
  "KV cache": ["KV cache", "key-value cache"],
};

// Contract notes addressed to the documentation's author, reworded for a reader. An empty note shows
// nothing. Any other row shows the contract's note unchanged.
export const READER_NOTES: Record<string, string> = {
  Dropout: "",
  "Weight tying": "`head_w` is a separate matrix from `tok_emb`.",
};

// Where the scan does not look, and why. Every other hand-written source of the site is scanned.
export const SCAN_EXEMPT: { path: string; reason: string }[] = [
  {
    path: "content/reference/not-in-this-model.mdx",
    reason: "This is the page that lists them.",
  },
  {
    path: "data/not-in-model.ts",
    reason: "This list of words itself.",
  },
  {
    path: "data/derived/",
    reason:
      "Snapshots of the contract. Not in This Model renders absent-techniques.json. Sources & Discrepancies renders discrepancies.json, which has to quote what the PDF says the code does not do.",
  },
];

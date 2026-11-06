// The files of one checkpoint. tests/checkpoint-page.test.ts compares the extensions with every
// `prefix + "..."` in save_checkpoint and load_checkpoint, so a file cannot be missing from this list.
export const CHECKPOINT_FILES = [
  {
    extension: ".npz",
    format: "NumPy archive",
    holds:
      "Every array in `params`, one entry per key. Keys such as `b0.wq` are kept as they are, and `np.load` gives the same keys back.",
    write: "checkpoint.save-weights",
    read: "checkpoint.load-weights",
  },
  {
    extension: ".json",
    format: "JSON",
    holds:
      "One object, the fields of `ModelConfig` from `asdict(cfg)`. `load_checkpoint` passes them back to `ModelConfig`, which rebuilds the shapes.",
    write: "checkpoint.save-config",
    read: "checkpoint.load-config",
  },
  {
    extension: ".tok.json",
    format: "JSON",
    holds:
      "One object with two keys, `merges` and `itos`. `merges` is the list of merge records and `itos` is the vocabulary list, where a token's position is its id.",
    write: "checkpoint.save-tokenizer",
    read: "checkpoint.load-stoi",
  },
] as const;

// Every symbol of a repository file, and where its page shows it or why it does not. site-verify fails
// the build when a symbol is missing from this list, and tests check that each `section` shows it.

export type CoverageRow = { symbol: string } & ({ section: string } | { skip: string });

export const SYMBOL_COVERAGE: Record<string, CoverageRow[]> = {
  "tokenizer.py": [
    { symbol: "tokenizer.UNK", section: "Encode and decode" },
    { symbol: "tokenizer.PRE_TOKEN_RE", section: "Pre-tokens and word counts" },
    { symbol: "tokenizer.count_words", section: "Pre-tokens and word counts" },
    { symbol: "tokenizer.merge_tokens", section: "Applying a merge" },
    { symbol: "tokenizer.train_bpe", section: "Pair frequency and the best pair" },
    { symbol: "tokenizer.apply_merges", section: "Encode and decode" },
    { symbol: "tokenizer.train_bpe_on_text", section: "Pair frequency and the best pair" },
    {
      symbol: "tokenizer.Tokenizer",
      skip: "Holds no logic of its own. Its encode and decode methods are shown separately.",
    },
    {
      symbol: "tokenizer.Tokenizer.__init__",
      skip: "Stores merges, stoi and itos, and sets vocab_size to len(itos). There is nothing to compute.",
    },
    { symbol: "tokenizer.Tokenizer.encode", section: "Encode and decode" },
    { symbol: "tokenizer.Tokenizer.decode", section: "Encode and decode" },
    { symbol: "tokenizer.build_tokenizer", section: "The vocabulary" },
  ],
  "data.py": [
    { symbol: "data.load_corpus", section: "Text to ids" },
    { symbol: "data.encode_corpus", section: "Text to ids" },
    { symbol: "data.split_ids", section: "The split" },
    { symbol: "data.make_windows", section: "Sliding windows" },
    { symbol: "data.iterate_batches", section: "Shuffling and batching" },
  ],
};

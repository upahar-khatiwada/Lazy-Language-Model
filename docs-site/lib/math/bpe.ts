// Words and pairs are kept in arrays and Maps, in first-seen order, because that order decides ties.
// A plain object would move numeric-looking keys such as "10" to the front.

export type Pair = [string, string];
export type PairCount = { pair: Pair; count: number };
export type Merge = { pair: Pair; merged: string; frequency: number };
export type WordFrequency = [word: string, frequency: number];

// Mirrors tokenizer.merge_tokens. Scans left to right, so "aaa" with ("a", "a") gives ["aa", "a"].
export function mergeTokens(tokens: string[], pair: Pair, merged: string): string[] {
  const result: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
      result.push(merged);
      i += 2;
    } else {
      result.push(tokens[i]!);
      i += 1;
    }
  }
  return result;
}

// Mirrors tokenizer.train_bpe, the pair_freq loop. Each word adds its frequency per occurrence.
export function countPairs(
  splits: [word: string, tokens: string[], weight: number][],
): PairCount[] {
  const counts = new Map<string, PairCount>();
  for (const [, tokens, weight] of splits) {
    for (let i = 0; i < tokens.length - 1; i++) {
      const pair: Pair = [tokens[i]!, tokens[i + 1]!];
      const key = JSON.stringify(pair);
      const entry = counts.get(key);
      if (entry) entry.count += weight;
      else counts.set(key, { pair, count: weight });
    }
  }
  return [...counts.values()];
}

// Mirrors tokenizer.train_bpe, the best_pair loop. Strict >, starting at 0: a tie goes to the pair
// seen first, and a pair with count 0 is never chosen.
export function bestPair(counts: PairCount[]): PairCount | null {
  let best: PairCount | null = null;
  let bestCount = 0;
  for (const entry of counts) {
    if (entry.count > bestCount) {
      bestCount = entry.count;
      best = entry;
    }
  }
  return best;
}

// Python's \w and \s are Unicode-aware and JavaScript's are not, and Unicode property classes do not
// match them exactly. The pages only run this on fixed ASCII text, where the two languages agree. The
// control characters \x1c-\x1f count as whitespace in Python only, so they are refused too.
const COMPARABLE = /^[\t\n\v\f\r\x20-\x7e]*$/;

// Mirrors tokenizer.count_words, with tokenizer.PRE_TOKEN_RE. First-seen order, as a Python dict keeps.
export function countWords(text: string): WordFrequency[] {
  if (!COMPARABLE.test(text)) throw new Error(`countWords only accepts ASCII text: ${text}`);
  const counts = new Map<string, number>();
  for (const [word] of text.matchAll(/\w+|\s|[^\w\s]/g)) {
    counts.set(word, (counts.get(word) ?? 0) + 1);
  }
  return [...counts.entries()];
}

// Mirrors tokenizer.train_bpe. Array.from splits by code point, as Python's list(word) does.
// `pairCounts` holds pair_freq at each step, which Python computes but does not return.
export function trainBpe(wordFreqs: WordFrequency[], maxMerges = 1000) {
  let splits = wordFreqs.map(
    ([word, weight]) => [word, Array.from(word), weight] as [string, string[], number],
  );
  const merges: Merge[] = [];
  const pairCounts: PairCount[][] = [];

  for (let step = 0; step < maxMerges; step++) {
    const counts = countPairs(splits);
    pairCounts.push(counts);
    const best = bestPair(counts);
    if (best === null) break;

    const merged = best.pair[0] + best.pair[1];
    splits = splits.map(([word, tokens, weight]) => [
      word,
      mergeTokens(tokens, best.pair, merged),
      weight,
    ]);
    merges.push({ pair: best.pair, merged, frequency: best.count });
  }
  return { merges, pairCounts };
}

export type WordSplit = [word: string, tokens: string[]];

export type BpeStep = {
  counts: PairCount[];
  best: PairCount;
  // Pairs with the same count as the chosen pair. They come later in pair_freq, so they lose.
  tied: PairCount[];
  merged: string;
  before: WordSplit[];
  after: WordSplit[];
};

// Mirrors tokenizer.train_bpe_on_text, keeping what tokenizer.train_bpe computes at each step for the
// stepper. `stopped` says why it ended: no adjacent pair is left, or max_merges was reached.
export function bpeSteps(text: string, maxMerges = 1000) {
  const wordFreqs = countWords(text);
  const weights = wordFreqs.map(([, weight]) => weight);
  const initial: WordSplit[] = wordFreqs.map(([word]) => [word, Array.from(word)]);
  const steps: BpeStep[] = [];
  let splits = initial;

  while (steps.length < maxMerges) {
    const counts = countPairs(splits.map(([word, tokens], i) => [word, tokens, weights[i]!]));
    const best = bestPair(counts);
    if (best === null) return { wordFreqs, initial, steps, stopped: "no-pairs" as const };

    const merged = best.pair[0] + best.pair[1];
    const after: WordSplit[] = splits.map(([word, tokens]) => [
      word,
      mergeTokens(tokens, best.pair, merged),
    ]);
    const tied = counts.filter((entry) => entry !== best && entry.count === best.count);
    steps.push({ counts, best, tied, merged, before: splits, after });
    splits = after;
  }
  return { wordFreqs, initial, steps, stopped: "limit" as const };
}

// Mirrors tokenizer.apply_merges. Each pre-token starts as characters and gets every merge, in order.
export function applyMerges(text: string, merges: Merge[]): string[] {
  return countWordsInOrder(text).flatMap((preToken) =>
    merges.reduce(
      (tokens, { pair, merged }) => mergeTokens(tokens, pair, merged),
      Array.from(preToken),
    ),
  );
}

function countWordsInOrder(text: string): string[] {
  if (!COMPARABLE.test(text)) throw new Error(`applyMerges only accepts ASCII text: ${text}`);
  return [...text.matchAll(/\w+|\s|[^\w\s]/g)].map(([preToken]) => preToken);
}

export type BpeTokenizer = {
  merges: Merge[];
  itos: string[];
  stoi: Map<string, number>;
  vocabSize: number;
};

export const UNK = "<unk>";

// Mirrors tokenizer.build_tokenizer. The unknown token, then characters in order of first appearance,
// then each merged string that is not already there.
export function buildTokenizer(text: string, maxMerges = 1000): BpeTokenizer {
  const { merges } = trainBpe(countWords(text), maxMerges);
  const itos = [...new Set([UNK, ...text, ...merges.map(({ merged }) => merged)])];
  const stoi = new Map(itos.map((token, i) => [token, i]));
  return { merges, itos, stoi, vocabSize: itos.length };
}

// Mirrors tokenizer.Tokenizer.encode. A token missing from stoi gets the id of the unknown token.
export function encode(tok: BpeTokenizer, text: string): number[] {
  return applyMerges(text, tok.merges).map((token) => tok.stoi.get(token) ?? tok.stoi.get(UNK)!);
}

// Mirrors tokenizer.Tokenizer.decode. Joins the strings of the ids, so an unknown id gives "<unk>".
export function decode(tok: BpeTokenizer, ids: number[]): string {
  return ids.map((id) => tok.itos[id]!).join("");
}

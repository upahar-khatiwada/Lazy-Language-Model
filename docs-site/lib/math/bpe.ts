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

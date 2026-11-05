// No shuffling here: its order comes from pandas and NumPy's generator, so pages show the stored result
// of the real iterate_batches for that, and only the sizes are computed.

// Mirrors the np.arange and range calls in data.make_windows. A plain counting helper.
export function range(start: number, stop: number, step = 1): number[] {
  const values: number[] = [];
  for (let value = start; value < stop; value += step) values.push(value);
  return values;
}

// Mirrors data.load_corpus, after pandas has read the rows. max_rows applies before missing rows are
// dropped. Takes the rows directly, because CSV parsing belongs to pandas.
export function joinRows(rows: (string | null)[], maxRows: number): string {
  const kept = rows.slice(0, maxRows).filter((row): row is string => row !== null);
  return `${kept.map((row) => row.trim()).join("\n")}\n`;
}

// Mirrors data.split_ids. The cut truncates, and nothing is shuffled.
export function splitIds<T>(ids: T[], valFraction: number) {
  const cut = Math.trunc((1 - valFraction) * ids.length);
  return { cut, train: ids.slice(0, cut), val: ids.slice(cut) };
}

// Mirrors data.make_windows, the np.arange(0, len(ids) - seq_len, stride) line.
export function windowStarts(n: number, seqLen: number, stride = 1): number[] {
  return range(0, n - seqLen, stride);
}

// Mirrors data.make_windows, as a formula for the length of its starts array.
export function windowCount(n: number, seqLen: number, stride = 1): number {
  return Math.ceil((n - seqLen) / stride);
}

// Mirrors data.make_windows, with the same error when there are too few ids.
export function makeWindows(ids: number[], seqLen: number, stride = 1) {
  if (ids.length <= seqLen) {
    throw new Error(`need more than seq_len=${seqLen} tokens, got ${ids.length}`);
  }
  const starts = windowStarts(ids.length, seqLen, stride);
  return {
    starts,
    inputs: starts.map((s) => ids.slice(s, s + seqLen)),
    targets: starts.map((s) => ids.slice(s + 1, s + seqLen + 1)),
  };
}

// Mirrors data.iterate_batches, the loop over range(0, len(order), batch_size). A short last batch is
// kept.
export function batchSizes(count: number, batchSize: number): number[] {
  return range(0, count, batchSize).map((start) => Math.min(batchSize, count - start));
}

// Mirrors train.main, the line that computes steps_per_epoch.
export function batchesPerEpoch(count: number, batchSize: number): number {
  return Math.ceil(count / batchSize);
}

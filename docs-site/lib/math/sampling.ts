import { softmax } from "./ops";
import type { Vector } from "./tensor";

// generate.sample_next ends with rng.choice(len(p), p=p). That generator is not reproduced here, so
// the engine stops at the distribution the choice would be made from.

function renormalise(p: Vector): Vector {
  const total = p.reduce((sum, value) => sum + value, 0);
  return p.map((value) => value / total);
}

// Mirrors generate.sample_next, the np.argmax branch. A tie goes to the first index.
export function greedyIndex(logits: Vector): number {
  let best = 0;
  logits.forEach((value, i) => {
    if (value > logits[best]!) best = i;
  });
  return best;
}

// Mirrors generate.sample_next, p = softmax_forward(logits / temperature).
export function applyTemperature(logits: Vector, temperature: number): Vector {
  return softmax(logits.map((value) => value / temperature));
}

// Mirrors generate.sample_next, the top-k branch. Runs only if 0 < k < len(p). Uses >= the k-th
// largest, so every token tied with it stays and more than k tokens can remain.
export function applyTopK(p: Vector, topK: number): Vector {
  if (!(0 < topK && topK < p.length)) return p;
  const cutoff = [...p].sort((a, b) => a - b)[p.length - topK]!;
  return renormalise(p.map((value) => (value >= cutoff ? value : 0)));
}

// Mirrors generate.sample_next, np.argsort(-p), but with a stable order. NumPy's default sort is
// not stable, so Python may order equal probabilities differently.
export function stableDescending(p: Vector): number[] {
  return p.map((_, i) => i).sort((a, b) => p[b]! - p[a]! || a - b);
}

export type TopPResult = {
  applied: boolean;
  p: Vector;
  order: number[];
  cum: Vector;
  keepCount: number;
  // True when the cutoff falls inside a group of equal non-zero probabilities. Which of those
  // tokens stay then depends on the sort's tie order, and only then. keepCount, cum and the sorted
  // kept probabilities do not depend on it. Pass Python's order to reproduce its exact tokens.
  tieDependent: boolean;
};

// Mirrors generate.sample_next, the top-p branch. Runs only if top_p < 1. The token that crosses
// top_p is kept: np.searchsorted (side left) finds the first cum >= top_p, and the slice adds 1.
export function applyTopP(p: Vector, topP: number, order = stableDescending(p)): TopPResult {
  if (!(topP < 1)) {
    return { applied: false, p, order: [], cum: [], keepCount: p.length, tieDependent: false };
  }
  let running = 0;
  const cum = order.map((index) => (running += p[index]!));
  const found = cum.findIndex((value) => value >= topP);
  const keepCount = found === -1 ? p.length : found + 1;

  const q = new Array<number>(p.length).fill(0);
  for (const index of order.slice(0, keepCount)) q[index] = p[index]!;

  const last = p[order[keepCount - 1]!]!;
  const next = keepCount < p.length ? p[order[keepCount]!]! : undefined;
  return {
    applied: true,
    p: renormalise(q),
    order,
    cum,
    keepCount,
    tieDependent: last === next && last > 0,
  };
}

export type SampleSettings = { temperature: number; topK: number; topP: number };

export type SampleResult =
  | { kind: "greedy"; index: number }
  | {
      kind: "distribution";
      p: Vector;
      afterTemperature: Vector;
      afterTopK: Vector;
      topP: TopPResult;
    };

// Mirrors generate.sample_next, up to the rng.choice call. `order` replaces the top-p sort order;
// leave it out to use a stable order.
export function sampleNextDistribution(
  logits: Vector,
  { temperature, topK, topP }: SampleSettings,
  order?: number[],
): SampleResult {
  if (temperature < 1e-6) return { kind: "greedy", index: greedyIndex(logits) };
  const afterTemperature = applyTemperature(logits, temperature);
  const afterTopK = applyTopK(afterTemperature, topK);
  const top = applyTopP(afterTopK, topP, order);
  return { kind: "distribution", p: top.p, afterTemperature, afterTopK, topP: top };
}

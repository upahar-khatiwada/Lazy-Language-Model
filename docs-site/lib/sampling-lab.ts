import { sampleNextDistribution, topKCutoff, topKRuns, type SampleSettings } from "./math/sampling";
import type { Vector } from "./math/tensor";

// What the probability lab shows for one set of inputs. Every number comes from the engine mirror of
// generate.sample_next. No file access, so client components can use it.

export type LabRow = {
  index: number;
  logit: number;
  afterTemperature: number;
  afterTopK: number;
  // Position in the top-p sort, and the running sum there. Null when top-p did not run.
  rank: number | null;
  cum: number | null;
  final: number;
  kept: boolean;
};

export type LabView =
  | { kind: "greedy"; index: number }
  | {
      kind: "distribution";
      rows: LabRow[];
      cutoff: number | null;
      topKRan: boolean;
      topPRan: boolean;
      keepCount: number;
      tieDependent: boolean;
    }
  | { kind: "not-finite" };

export function labView(logits: Vector, settings: SampleSettings, order?: number[]): LabView {
  const result = sampleNextDistribution(logits, settings, order);
  if (result.kind === "greedy") return { kind: "greedy", index: result.index };
  if (!result.p.every(Number.isFinite)) return { kind: "not-finite" };

  const { afterTemperature, afterTopK, topP, p } = result;
  const topKRan = topKRuns(afterTemperature, settings.topK);
  const rows = logits.map((logit, index): LabRow => {
    const rank = topP.applied ? topP.order.indexOf(index) : null;
    return {
      index,
      logit,
      afterTemperature: afterTemperature[index]!,
      afterTopK: afterTopK[index]!,
      rank,
      cum: rank === null ? null : topP.cum[rank]!,
      final: p[index]!,
      kept: p[index]! > 0,
    };
  });
  return {
    kind: "distribution",
    rows,
    cutoff: topKRan ? topKCutoff(afterTemperature, settings.topK) : null,
    topKRan,
    topPRan: topP.applied,
    keepCount: rows.filter((row) => row.kept).length,
    tieDependent: topP.tieDependent,
  };
}

const four = (value: number) => String(Number(value.toFixed(4)));

// One sentence per step of sample_next, for the settings given. The wording follows the code's order.
export function labSentences(view: LabView, settings: SampleSettings): string[] {
  if (view.kind === "not-finite") {
    return ["These inputs do not give finite probabilities, so the lab shows no result."];
  }
  if (view.kind === "greedy") {
    return [
      `temperature is below 1e-6, so sample_next returns token ${view.index}, the np.argmax of the logits. top_k, top_p and rng are not used.`,
    ];
  }
  const sentences = [`p = softmax_forward(logits / ${settings.temperature}).`];
  if (view.cutoff === null) {
    sentences.push(`top_k = ${settings.topK}: 0 < top_k < len(p) is false, so top-k is skipped.`);
  } else {
    const left = view.rows.filter((row) => row.afterTopK > 0).length;
    sentences.push(
      `top_k = ${settings.topK}: the cut-off is ${four(view.cutoff)}, and ${left} token${left === 1 ? "" : "s"} are at or above it.` +
        (left > settings.topK ? " Ties at the cut-off all stay, so more than k remain." : ""),
    );
  }
  if (!view.topPRan) {
    sentences.push(`top_p = ${settings.topP}: top_p < 1.0 is false, so top-p is skipped.`);
  } else {
    sentences.push(
      `top_p = ${settings.topP}: the sum first reaches it within the first ${view.keepCount} sorted token${view.keepCount === 1 ? "" : "s"}, and the token that crosses it is kept.`,
    );
  }
  if (view.tieDependent) {
    sentences.push(
      "The cut falls inside a group of equal probabilities, so the sort order decides which of them stay. This lab sorts ties by lowest index. np.argsort does not promise an order for ties, so Python may keep different tokens. The number kept and their values are the same.",
    );
  }
  sentences.push(
    `rng.choice would then draw from the ${view.keepCount} kept token${view.keepCount === 1 ? "" : "s"}.`,
  );
  return sentences;
}

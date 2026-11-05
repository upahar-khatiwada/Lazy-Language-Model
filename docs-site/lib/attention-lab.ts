import { attentionForward, type AttentionParams } from "./math/attention";
import type { Matrix, Vector } from "./math/tensor";

// The data behind the attention lab. No React and no file reads, so tests can compare every matrix
// with the golden results of the real attention_forward, and the browser can run it.

// One row of the running example per word. The numbers are hand-picked and do not encode these words.
export const LAB_TOKENS = ["The", "cat", "sat", "on", "the", "mat"];

export type LabInputs = { Xn: Matrix; H: number } & AttentionParams;
export type LabRun = ReturnType<typeof attentionForward>;

export const LAB_STEPS = [
  {
    id: "scores",
    title: "Scores",
    // The equations whose link records give the lines this step highlights.
    equations: ["attention.scores"],
    section: "scores",
    text: "Cell (i, j) is the dot product of row i's query with row j's key. Python computes this and the next step on one line.",
  },
  {
    id: "scale",
    title: "Scale",
    equations: ["attention.scale"],
    section: "scaling-by-sqrtd_h",
    text: "Every score is divided by the same number, the square root of the head width.",
  },
  {
    id: "mask",
    title: "Mask",
    equations: ["attention.mask"],
    section: "causal-mask",
    text: "Cells above the diagonal become minus infinity. A position keeps its own cell and the cells of earlier positions.",
  },
  {
    id: "softmax",
    title: "Softmax",
    equations: ["attention.softmax"],
    section: "softmax-per-row",
    text: "Each row becomes weights that add up to 1. A masked cell is exactly 0.",
  },
  {
    id: "output",
    title: "Output",
    equations: ["attention.weighted", "attention.merge", "attention.output"],
    section: "weighted-values",
    text: "The weights multiply the value rows in each head. The heads are merged, and the output projection gives the result.",
  },
] as const;

export type LabStepId = (typeof LAB_STEPS)[number]["id"];

export function runLab({ Xn, H, ...params }: LabInputs): LabRun {
  return attentionForward(Xn, params, H);
}

// The matrix a step shows, for one head. The Output step shows the weights it multiplies with.
export function stepMatrix(run: LabRun, step: LabStepId, head: number): Matrix {
  const source = {
    scores: run.raw,
    scale: run.scaled,
    mask: run.masked,
    softmax: run.A,
    output: run.A,
  }[step];
  return source[head]!;
}

// Cells the mask has set. Read from the masked scores, so this cannot disagree with the engine.
export function maskedCells(run: LabRun, head: number): boolean[][] {
  return run.masked[head]!.map((row) => row.map((value) => value === -Infinity));
}

export type WeightedTerm = { position: number; weight: number; value: Vector };

// The non-masked terms of (A Vh)[row], and their sum as the merged O holds it.
export function weightedTerms(run: LabRun, head: number, row: number) {
  const width = run.Vh[head]![0]!.length;
  const terms: WeightedTerm[] = run.A[head]![row]!.flatMap((weight, position) =>
    run.masked[head]![row]![position] === -Infinity
      ? []
      : [{ position, weight, value: run.Vh[head]![position]! }],
  );
  return { terms, result: run.O[row]!.slice(head * width, (head + 1) * width) };
}

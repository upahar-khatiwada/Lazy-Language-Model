import type { Matrix, Vector } from "./math/tensor";

// The steps of the residual-stream stepper. No Node imports, so the browser can load this.
// `ref` names a line in data/code-refs.ts.

export type StreamRun = {
  X: Matrix;
  Xn: Matrix;
  att: Matrix;
  X1: Matrix;
  Xn2: Matrix;
  ff: Matrix;
  Y: Matrix;
};

export type StreamField = keyof StreamRun;

export type StreamStep = {
  id: string;
  title: string;
  ref: string;
  text: string;
  // "add" shows two rows and their sum. "branch" shows a sub-layer's input and output.
  kind: "input" | "branch" | "add";
  rows: StreamField[];
};

export const STREAM_STEPS: StreamStep[] = [
  {
    id: "input",
    title: "Input",
    ref: "block.def",
    text: "X is the residual stream as it enters the block.",
    kind: "input",
    rows: ["X"],
  },
  {
    id: "ln1",
    title: "LayerNorm 1",
    ref: "ln.block-first",
    text: "LayerNorm 1 makes a normalised copy for attention to read. The stream itself does not change.",
    kind: "branch",
    rows: ["X", "Xn"],
  },
  {
    id: "attention",
    title: "Attention",
    ref: "attention.call",
    text: "Attention reads the normalised copy of this row and of earlier rows.",
    kind: "branch",
    rows: ["Xn", "att"],
  },
  {
    id: "add1",
    title: "First add",
    ref: "attention.residual",
    text: "The first add. Attention's output goes onto X, which has not been normalised.",
    kind: "add",
    rows: ["X", "att", "X1"],
  },
  {
    id: "ln2",
    title: "LayerNorm 2",
    ref: "ln.block-second",
    text: "LayerNorm 2 makes a normalised copy for the feed-forward layer. The stream does not change.",
    kind: "branch",
    rows: ["X1", "Xn2"],
  },
  {
    id: "ffn",
    title: "Feed-forward",
    ref: "block.ffn-call",
    text: "The feed-forward layer reads the normalised copy of this row only.",
    kind: "branch",
    rows: ["Xn2", "ff"],
  },
  {
    id: "add2",
    title: "Second add",
    ref: "block.return",
    text: "The second add. The feed-forward output goes onto X1, and the result leaves the block.",
    kind: "add",
    rows: ["X1", "ff", "Y"],
  },
];

// One row of one array, as the stepper shows it.
export function streamVector(run: StreamRun, field: StreamField, row: number): Vector {
  const values = run[field][row];
  if (!values) throw new Error(`The stream has no row ${row} in ${field}`);
  return values;
}

import type { FlatParams, Gradients } from "./block";
import { forward, lossAndGrads, type ModelSizes } from "./model";
import { crossEntropy } from "./ops";
import {
  adamStep,
  clipGradNorm,
  lrSchedule,
  type AdamState,
  type OptimTensor,
  type TensorMap,
} from "./optim";
import type { Matrix, Vector } from "./tensor";

function isMatrix(tensor: OptimTensor): tensor is Matrix {
  return Array.isArray(tensor[0]);
}

function addDivided(total: OptimTensor, value: OptimTensor, B: number): OptimTensor {
  if (isMatrix(total)) {
    const rows = value as Matrix;
    return total.map((row, r) => row.map((item, c) => item + rows[r]![c]! / B));
  }
  const items = value as Vector;
  return total.map((item, i) => item + items[i]! / B);
}

function zerosLike(tensor: OptimTensor): OptimTensor {
  return isMatrix(tensor) ? tensor.map((row) => row.map(() => 0)) : tensor.map(() => 0);
}

// Mirrors train.batch_loss_and_grads. Each window's loss and gradients are divided by B as they are
// added. B is the length of this batch, so a short last batch divides by its own size.
export function batchLossAndGrads(
  params: FlatParams,
  xb: number[][],
  yb: number[][],
  cfg: ModelSizes,
) {
  const B = xb.length;
  const grads: Gradients = {};
  for (const [name, tensor] of Object.entries(params)) grads[name] = zerosLike(tensor);
  let totalLoss = 0;
  const losses: number[] = [];

  xb.forEach((x, index) => {
    const { loss, grads: g } = lossAndGrads(params, x, yb[index]!, cfg);
    losses.push(loss);
    totalLoss += loss / B;
    for (const name of Object.keys(grads)) grads[name] = addDivided(grads[name]!, g[name]!, B);
  });

  return { loss: totalLoss, grads, losses };
}

// Mirrors train.validation_loss. The mean of each window's mean cross-entropy, one window at a time.
export function validationLoss(
  params: FlatParams,
  cfg: ModelSizes,
  inputs: number[][],
  targets: number[][],
) {
  const losses = inputs.map((x, index) => {
    const { logits } = forward(params, x, cfg);
    return crossEntropy(logits, targets[index]!).loss;
  });
  return { loss: losses.reduce((total, loss) => total + loss, 0) / losses.length, losses };
}

// Mirrors train.main, the two lines that compute steps_per_epoch and total_steps. lastBatch is the size
// of the last batch data.iterate_batches hands out.
export function stepCounts(windows: number, batchSize: number, epochs: number) {
  const stepsPerEpoch = Math.ceil(windows / batchSize);
  return {
    stepsPerEpoch,
    totalSteps: epochs * stepsPerEpoch,
    lastBatch: windows % batchSize === 0 ? batchSize : windows % batchSize,
  };
}

export type LoopSettings = {
  maxNorm: number;
  warmup: number;
  total: number;
  lrMax: number;
  lrMin: number;
};

// The calls and the one statement of the batch loop in train.main that come after the gradient.
export const LOOP_STEPS = ["clip_grad_norm", "step += 1", "lr_schedule", "opt.step"] as const;

// Mirrors train.main, the batch loop after the gradient: clip, count the step, schedule, Adam.
export function loopStep(
  state: AdamState,
  params: TensorMap,
  grads: TensorMap,
  stepBefore: number,
  { maxNorm, warmup, total, lrMax, lrMin }: LoopSettings,
) {
  const clipped = clipGradNorm(grads, maxNorm);
  const step = stepBefore + 1;
  const lr = lrSchedule(step, warmup, total, lrMax, lrMin);
  const next = adamStep(state, params, clipped.grads, lr);
  return { norm: clipped.norm, clipped: clipped.grads, step, lr, ...next };
}

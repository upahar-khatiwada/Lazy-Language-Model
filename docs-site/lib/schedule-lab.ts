import { lrSchedule } from "./math/optim";

// The chart draws straight lines between these steps. Values always come from lrSchedule.
const MAX_POINTS = 240;

// Steps to draw for a schedule: every step if there are few, otherwise an even spread. The start, the
// warmup end, the first step after it and the last step are always included, so the corner is exact.
export function chartSteps(total: number, warmup: number): number[] {
  const steps = new Set([0, 1, warmup, warmup + 1, total].filter((step) => step <= total));
  const stride = Math.max(1, Math.ceil(total / MAX_POINTS));
  for (let step = 0; step <= total; step += stride) steps.add(step);
  return [...steps].sort((a, b) => a - b);
}

export type ScheduleSettings = { total: number; warmup: number; lrMax: number; lrMin: number };

export function schedulePoints({ total, warmup, lrMax, lrMin }: ScheduleSettings) {
  return chartSteps(total, warmup).map((step) => ({
    x: step,
    y: lrSchedule(step, warmup, total, lrMax, lrMin),
  }));
}

// The rows the page lists by name. Step 0 is what the function returns; train.main never calls it.
export function scheduleKeyPoints({ total, warmup, lrMax, lrMin }: ScheduleSettings) {
  const at = (step: number) => lrSchedule(step, warmup, total, lrMax, lrMin);
  return [
    { id: "start", label: "Step 0, which train.main never uses", step: 0, lr: at(0) },
    { id: "first", label: "Step 1, the first one used", step: 1, lr: at(1) },
    ...(warmup <= total
      ? [{ id: "warmup", label: "The last warmup step", step: warmup, lr: at(warmup) }]
      : []),
    { id: "final", label: "The last step", step: total, lr: at(total) },
  ];
}

"use client";

import { useId, useState } from "react";
import { adamStep, createAdamState, type OptimTensor } from "@/lib/math/optim";
import { WorkedRows } from "./WorkedRows";
import examples from "./Examples.module.css";
import { VizFigure } from "./VizFigure";

type Settings = { beta1: number; beta2: number; eps: number; weightDecay: number };

export type AdamStart = { w: number; g: number; m: number; v: number; t: number; lr: number };

const entry = (tensor: OptimTensor) => tensor.flat()[0] as number;

const FIELDS = [
  { key: "w", label: "params[k]" },
  { key: "g", label: "grads[k]" },
  { key: "m", label: "self.m[k]" },
  { key: "v", label: "self.v[k]" },
  { key: "t", label: "self.t" },
  { key: "lr", label: "lr" },
] as const;

// One Adam.step on one number. Results come from lib/math/optim.ts, which tests compare with the
// real function. The betas, eps and weight_decay stay at their config.py values.
export function AdamCalculator({ start, settings }: { start: AdamStart; settings: Settings }) {
  const [texts, setTexts] = useState(
    Object.fromEntries(FIELDS.map(({ key }) => [key, String(start[key])])) as Record<
      keyof AdamStart,
      string
    >,
  );
  const [matrix, setMatrix] = useState(true);
  const baseId = useId();

  const values = Object.fromEntries(
    FIELDS.map(({ key }) => [key, Number(texts[key])]),
  ) as AdamStart;
  const problem = FIELDS.some(
    ({ key }) => texts[key].trim() === "" || !Number.isFinite(values[key]),
  )
    ? "Every field must be a number."
    : !Number.isInteger(values.t) || values.t < 0
      ? "self.t must be a whole number of at least 0."
      : values.v < 0
        ? "self.v[k] is an average of squares, so it cannot be negative."
        : null;

  let rows = null;
  if (problem === null) {
    const wrap = (value: number): OptimTensor => (matrix ? [[value]] : [value]);
    const params = { w: wrap(values.w) };
    const state = {
      ...createAdamState(params, settings),
      m: { w: wrap(values.m) },
      v: { w: wrap(values.v) },
      t: values.t,
    };
    const next = adamStep(state, params, { w: wrap(values.g) }, values.lr);
    const trace = next.trace.w!;
    rows = [
      { code: "self.t += 1", value: next.state.t, testId: "calc-t" },
      {
        code: "self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g",
        value: entry(next.state.m.w!),
        testId: "calc-m",
      },
      {
        code: "self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g",
        value: entry(next.state.v.w!),
        testId: "calc-v",
      },
      {
        code: "mhat = self.m[k] / (1 - self.b1 ** self.t)",
        value: entry(trace.mhat),
        testId: "calc-mhat",
      },
      {
        code: "vhat = self.v[k] / (1 - self.b2 ** self.t)",
        value: entry(trace.vhat),
        testId: "calc-vhat",
      },
      {
        code: "update = mhat / (np.sqrt(vhat) + self.eps)",
        value: entry(trace.raw),
        testId: "calc-raw",
      },
      ...(matrix
        ? [
            {
              code: "update = update + self.wd * params[k]",
              value: entry(trace.update),
              testId: "calc-update",
            },
          ]
        : []),
      { code: "params[k] -= lr * update", value: entry(next.params.w!), testId: "calc-w" },
    ];
  }

  return (
    <VizFigure
      title="One step for one number"
      kind="illustrative"
      caption="The fields start at the two-step example's first step, and you can edit them. To see that example's second step, copy self.m[k], self.v[k], self.t and params[k] from the result back into the fields, and set grads[k] to its second gradient. The betas, eps and weight_decay stay at their config.py values."
    >
      <div className={examples.controls} data-testid="adam-calculator">
        {FIELDS.map(({ key, label }) => (
          <div className={examples.control} key={key}>
            <label htmlFor={`${baseId}-${key}`}>{label}</label>
            <input
              id={`${baseId}-${key}`}
              data-testid={`calc-input-${key}`}
              type="number"
              step="any"
              value={texts[key]}
              onChange={(event) => setTexts({ ...texts, [key]: event.target.value })}
            />
          </div>
        ))}
        <div className={examples.control}>
          <label htmlFor={`${baseId}-ndim`}>Dimensions</label>
          <select
            id={`${baseId}-ndim`}
            data-testid="calc-ndim"
            value={matrix ? "2" : "1"}
            onChange={(event) => setMatrix(event.target.value === "2")}
          >
            <option value="2">2, so decay applies</option>
            <option value="1">1, so no decay</option>
          </select>
        </div>
      </div>
      <p className={examples.status} role="status" data-testid="calc-status">
        {problem ?? ""}
      </p>
      {rows ? <WorkedRows label="Result of one step" rows={rows} /> : null}
    </VizFigure>
  );
}

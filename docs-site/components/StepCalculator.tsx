"use client";

import { useId, useState } from "react";
import { stepCounts } from "@/lib/math/train";
import styles from "./Examples.module.css";
import { VizFigure } from "./VizFigure";

const whole = (text: string) => (/^[1-9]\d{0,11}$/.test(text.trim()) ? Number(text) : null);

const FIELDS = [
  { key: "windows", label: "Windows, len(tr_x)" },
  { key: "batchSize", label: "batch_size" },
  { key: "epochs", label: "epochs" },
] as const;

// Results come from lib/math/train.ts, which tests compare with the real lines of train.main.
export function StepCalculator({
  windows,
  batchSize,
  epochs,
}: {
  windows: number;
  batchSize: number;
  epochs: number;
}) {
  const [texts, setTexts] = useState({
    windows: String(windows),
    batchSize: String(batchSize),
    epochs: String(epochs),
  });
  const baseId = useId();

  const parsed = {
    windows: whole(texts.windows),
    batchSize: whole(texts.batchSize),
    epochs: whole(texts.epochs),
  };
  const ready = parsed.windows !== null && parsed.batchSize !== null && parsed.epochs !== null;
  const counts = ready ? stepCounts(parsed.windows!, parsed.batchSize!, parsed.epochs!) : null;

  return (
    <VizFigure
      title="Steps from windows, batch size and epochs"
      kind="illustrative"
      caption="The number of windows is hand-picked, because the real number depends on the corpus. batch_size and epochs start at their config.py values."
    >
      <div className={styles.controls} data-testid="step-calculator">
        {FIELDS.map(({ key, label }) => (
          <div className={styles.control} key={key}>
            <label htmlFor={`${baseId}-${key}`}>{label}</label>
            <input
              id={`${baseId}-${key}`}
              data-testid={`steps-${key}`}
              type="number"
              min={1}
              value={texts[key]}
              onChange={(event) => setTexts({ ...texts, [key]: event.target.value })}
            />
          </div>
        ))}
      </div>
      <p className={styles.status} role="status" data-testid="steps-status">
        {ready ? "" : "Every field must be a whole number of at least 1."}
      </p>
      {counts ? (
        <ul className={styles.results}>
          <li>
            <code data-testid="steps-per-epoch">
              steps_per_epoch = ceil({parsed.windows} / {parsed.batchSize}) = {counts.stepsPerEpoch}
            </code>
          </li>
          <li>
            <code data-testid="steps-total">
              total_steps = {parsed.epochs} * {counts.stepsPerEpoch} = {counts.totalSteps}
            </code>
          </li>
          <li>
            <span data-testid="steps-last">
              The last batch of each epoch has <code>{counts.lastBatch}</code>{" "}
              {counts.lastBatch === 1 ? "window" : "windows"}.
            </span>
          </li>
        </ul>
      ) : null}
    </VizFigure>
  );
}

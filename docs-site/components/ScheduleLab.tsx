"use client";

import { useId, useState } from "react";
import { scheduleKeyPoints, schedulePoints, type ScheduleSettings } from "@/lib/schedule-lab";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { LineChart } from "./LineChart";

const whole = (text: string) => (/^[1-9]\d{0,8}$/.test(text.trim()) ? Number(text) : null);
const rate = (text: string) => {
  const value = Number(text);
  return text.trim() !== "" && Number.isFinite(value) && value >= 0 ? value : null;
};

function problemWith(
  total: number | null,
  warmup: number | null,
  lrMax: number | null,
  lrMin: number | null,
) {
  if (total === null) return "total_steps must be a whole number of at least 1.";
  if (warmup === null) return "warmup_steps must be a whole number of at least 1 here.";
  if (lrMax === null || lrMin === null) return "lr_max and lr_min must be numbers of at least 0.";
  if (lrMin > lrMax) return "lr_min above lr_max is not shown here. The code does not check this.";
  return null;
}

// Values come from lib/math/optim.ts, which tests compare with the real lr_schedule.
export function ScheduleLab({ start }: { start: ScheduleSettings }) {
  const [totalText, setTotalText] = useState(String(start.total));
  const [warmupText, setWarmupText] = useState(String(start.warmup));
  const [maxText, setMaxText] = useState(String(start.lrMax));
  const [minText, setMinText] = useState(String(start.lrMin));
  const totalField = useId();
  const warmupField = useId();
  const maxField = useId();
  const minField = useId();

  const total = whole(totalText);
  const warmup = whole(warmupText);
  const lrMax = rate(maxText);
  const lrMin = rate(minText);
  const problem = problemWith(total, warmup, lrMax, lrMin);
  const settings = problem === null ? ({ total, warmup, lrMax, lrMin } as ScheduleSettings) : null;
  const points = settings ? schedulePoints(settings) : [];
  const keys = settings ? scheduleKeyPoints(settings) : [];
  const labelled = settings
    ? keys
        .filter((key) => key.id !== "first")
        .map((key) => points.findIndex((point) => point.x === key.step))
    : [];

  return (
    <div className={styles.stack} data-testid="schedule-lab">
      <div className={styles.controls}>
        <div className={styles.control}>
          <label htmlFor={totalField}>total_steps</label>
          <input
            id={totalField}
            data-testid="schedule-total"
            type="number"
            min={1}
            value={totalText}
            onChange={(event) => setTotalText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={warmupField}>warmup_steps</label>
          <input
            id={warmupField}
            data-testid="schedule-warmup"
            type="number"
            min={1}
            value={warmupText}
            onChange={(event) => setWarmupText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={maxField}>lr_max</label>
          <input
            id={maxField}
            data-testid="schedule-max"
            type="number"
            step="any"
            min={0}
            value={maxText}
            onChange={(event) => setMaxText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={minField}>lr_min</label>
          <input
            id={minField}
            data-testid="schedule-min"
            type="number"
            step="any"
            min={0}
            value={minText}
            onChange={(event) => setMinText(event.target.value)}
          />
        </div>
      </div>
      <p className={styles.status} role="status" data-testid="schedule-status">
        {problem ??
          (settings && settings.total < settings.warmup
            ? "total_steps is below warmup_steps, so the schedule never reaches lr_max."
            : "")}
      </p>
      {settings ? (
        <>
          <LineChart
            title="lr against step"
            kind="illustrative"
            caption="warmup_steps, lr_max and lr_min start at their config.py values. total_steps depends on the corpus, so its starting value is hand-picked. Both axes are linear, and the y axis starts at 0."
            points={points}
            xLabel="step"
            yLabel="lr"
            labelled={labelled}
            yDomain={[0, Math.max(settings.lrMax, settings.lrMin)]}
          />
          <DataTable label="Values at the key steps" data-testid="schedule-keys">
            <thead>
              <tr>
                <th scope="col">Where</th>
                <th scope="col">step</th>
                <th scope="col">lr</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.id}>
                  <td>{key.label}</td>
                  <td>
                    <code data-testid={`schedule-step-${key.id}`}>{key.step}</code>
                  </td>
                  <td>
                    <code data-testid={`schedule-lr-${key.id}`} data-exact={JSON.stringify(key.lr)}>
                      {Number(key.lr.toPrecision(6))}
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        </>
      ) : null}
    </div>
  );
}

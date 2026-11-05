"use client";

import { useState } from "react";
import {
  STREAM_STEPS,
  streamVector,
  type StreamField,
  type StreamRun,
  type StreamStep,
} from "@/lib/residual-stream";
import styles from "./AttentionLab.module.css";
import examples from "./Examples.module.css";
import flow from "./GradientFlow.module.css";
import { showNumber } from "./ResultMatrix";
import { StepPlayer } from "./StepPlayer";

export type StreamLine = { label: string; href: string; text: string };

function VectorRows({ run, step, row }: { run: StreamRun; step: StreamStep; row: number }) {
  const sign = (index: number) => {
    if (step.kind !== "add") return "";
    return index === 1 ? "+" : index === 2 ? "=" : "";
  };
  const name = (field: StreamField, index: number) => {
    if (step.kind === "branch") return index === 0 ? "reads" : "gives";
    return "";
  };
  return (
    <table className={flow.vectors} data-testid="stream-vectors">
      <caption className="visually-hidden">
        Row {row} of each array at this step. Each array has shape (T, d).
      </caption>
      <tbody>
        {step.rows.map((field, index) => {
          const values = streamVector(run, field, row);
          return (
            <tr key={field} data-field={field}>
              <td className={flow.sign}>{sign(index) || name(field, index)}</td>
              <th scope="row">
                <code>
                  {field}[{row}]
                </code>
              </th>
              {values.map((value, c) => (
                <td key={c} className={flow.number}>
                  {showNumber(value)}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// Every number comes from lib/math/block.ts, which tests compare with the real block_forward.
export function ResidualStream({
  run,
  lines,
  defaultRow,
}: {
  run: StreamRun;
  lines: Record<string, StreamLine>;
  defaultRow: number;
}) {
  const [row, setRow] = useState(defaultRow);
  return (
    <div className={styles.lab} data-testid="residual-stream">
      <div className={styles.row} role="group" aria-label="Selected row">
        <span className={styles.rowLabel}>Row</span>
        {run.X.map((_, t) => (
          <button
            type="button"
            className={styles.token}
            key={t}
            aria-pressed={t === row}
            data-testid={`stream-row-${t}`}
            onClick={() => setRow(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <StepPlayer
        label="Residual stream steps"
        steps={STREAM_STEPS.map((step) => {
          const line = lines[step.ref];
          if (!line) throw new Error(`No resolved line for ${step.ref}`);
          return {
            id: step.id,
            title: step.title,
            content: (
              <div
                className={styles.panel}
                data-testid={`stream-panel-${step.id}`}
                data-exact={JSON.stringify(step.rows.map((field) => streamVector(run, field, row)))}
              >
                <p className={styles.stepText}>{step.text}</p>
                <div className={styles.scroll} tabIndex={0} role="group" aria-label="Vectors">
                  <VectorRows run={run} step={step} row={row} />
                </div>
                <p className={examples.status}>
                  <a className="code-link" href={line.href} data-testid="stream-line">
                    {line.label}
                  </a>{" "}
                  <code data-testid="stream-code">{line.text}</code>
                </p>
              </div>
            ),
          };
        })}
      />
    </div>
  );
}

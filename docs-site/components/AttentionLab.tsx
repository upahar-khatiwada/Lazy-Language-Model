"use client";

import { useId, useState } from "react";
import {
  LAB_STEPS,
  LAB_TOKENS,
  maskedCells,
  runLab,
  stepMatrix,
  weightedTerms,
  type LabInputs,
  type LabStepId,
} from "@/lib/attention-lab";
import styles from "./AttentionLab.module.css";
import { BarChart } from "./BarChart";
import { CodeFrame, type CodeFrameProps } from "./CodeFrame";
import examples from "./Examples.module.css";
import { Heatmap } from "./Heatmap";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { StepPlayer } from "./StepPlayer";

const vector = (values: number[]) => `[${values.map(showNumber).join(", ")}]`;

const ACTUAL_NOTE =
  "Actual Model Output is not available yet. No attention weights from a real run have been stored for this site.";

// Every number comes from lib/attention-lab, which tests compare with the real attention_forward.
export function AttentionLab({
  inputs,
  code,
  stepLines,
}: {
  inputs: LabInputs;
  code: Omit<CodeFrameProps, "highlightLines">;
  // Real line numbers per step, resolved from the link records on the server.
  stepLines: Record<LabStepId, number[]>;
}) {
  const id = useId();
  const [row, setRow] = useState(2);
  const [head, setHead] = useState(0);
  const run = runLab(inputs);
  const tokens = LAB_TOKENS;
  const masked = maskedCells(run, head);
  const weights = run.A[head]![row]!;
  const { terms, result } = weightedTerms(run, head, row);
  const sum = weights.reduce((total, value) => total + value, 0);
  const dh = run.Vh[head]![0]!.length;

  const panel = (step: (typeof LAB_STEPS)[number]) => {
    const matrix = stepMatrix(run, step.id, head);
    const hasWeights = step.id === "softmax" || step.id === "output";
    return (
      <div className={styles.panel} data-testid={`lab-panel-${step.id}`}>
        <p className={styles.stepText} data-testid="lab-step-text">
          {step.text}
          {step.id === "scale" ? ` In this example d_h is ${dh}.` : ""}{" "}
          <a href={`#${step.section}`}>Read this step</a>
        </p>
        {/* JSON has no infinity, so a masked score appears in data-exact as null. */}
        <div data-testid="lab-matrix" data-exact={JSON.stringify(matrix)}>
          <Heatmap
            title={`${hasWeights ? "Weights A" : step.title}, head ${head}`}
            kind="illustrative"
            caption="Hand-picked input. These are not learned values, and they say nothing about the trained model."
            rowLabels={tokens}
            colLabels={tokens}
            xLabel="reads"
            yLabel="row"
            values={matrix}
            masked={hasWeights ? masked : undefined}
            selectedRow={row}
          />
        </div>
        {hasWeights ? (
          <div data-testid="lab-weights" data-exact={JSON.stringify(weights)}>
            <BarChart
              title={`Weights of "${tokens[row]}", head ${head}`}
              kind="illustrative"
              caption={`This row adds up to ${sum.toFixed(4)}. Positions after this token are masked, so their weight is exactly 0.`}
              labels={tokens}
              values={weights}
              valueHeader="Weight"
              domain={[0, 1]}
            />
          </div>
        ) : (
          <p className={styles.stepText}>Weights for the selected token appear at Softmax.</p>
        )}
        {step.id === "output" ? (
          <>
            <p className={styles.formula} data-testid="lab-formula">
              <code>
                (A @ Vh)[{row}] ={" "}
                {terms.map((t) => `${showNumber(t.weight)} * ${vector(t.value)}`).join(" + ")} ={" "}
                {vector(result)}
              </code>
            </p>
            <div
              className={styles.scroll}
              tabIndex={0}
              role="group"
              aria-label="Merged values and output"
            >
              <div className={examples.matrices}>
                <ResultMatrix label="O, both heads merged" testId="lab-o" values={run.O} />
                <ResultMatrix label="out" testId="lab-out" values={run.out} />
              </div>
            </div>
          </>
        ) : null}
        <div data-testid="lab-code">
          <CodeFrame {...code} highlightLines={new Set(stepLines[step.id])} />
        </div>
      </div>
    );
  };

  return (
    <div className={styles.lab} data-testid="attention-lab">
      <fieldset className={styles.source} data-testid="lab-source">
        <legend>Data source</legend>
        <label className={styles.option}>
          <input type="radio" name={`${id}-source`} defaultChecked />
          Illustrative Example
        </label>
        {/* Disabled until the site has attention weights from a real run. */}
        <label className={`${styles.option} ${styles.optionOff}`} title={ACTUAL_NOTE}>
          <input type="radio" name={`${id}-source`} disabled aria-describedby={`${id}-note`} />
          Actual Model Output
        </label>
        <p className={styles.sourceNote} id={`${id}-note`}>
          {ACTUAL_NOTE}
        </p>
      </fieldset>

      <div className={styles.row} role="group" aria-label="Selected token">
        <span className={styles.rowLabel}>Token</span>
        {tokens.map((token, t) => (
          <button
            type="button"
            className={styles.token}
            key={token}
            aria-pressed={t === row}
            data-testid={`lab-token-${t}`}
            onClick={() => setRow(t)}
          >
            {token}
          </button>
        ))}
      </div>

      <div className={styles.row} role="group" aria-label="Selected head">
        <span className={styles.rowLabel}>Head</span>
        {run.A.map((_, h) => (
          <button
            type="button"
            className={styles.token}
            key={h}
            aria-pressed={h === head}
            data-testid={`lab-head-${h}`}
            onClick={() => setHead(h)}
          >
            {h}
          </button>
        ))}
      </div>

      <StepPlayer
        label="Attention steps"
        steps={LAB_STEPS.map((step) => ({ id: step.id, title: step.title, content: panel(step) }))}
      />
    </div>
  );
}

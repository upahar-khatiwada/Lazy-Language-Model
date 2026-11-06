"use client";

import { useId, useState } from "react";
import type { SampleSettings } from "@/lib/math/sampling";
import type { Vector } from "@/lib/math/tensor";
import { labSentences, labView } from "@/lib/sampling-lab";
import { BarChart } from "./BarChart";
import { buttonClass } from "./Button";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { MatrixInput } from "./MatrixInput";
import { ProbabilityBars } from "./ProbabilityBars";
import { showNumber } from "./ResultMatrix";

export type LabPreset = { id: string; label: string; logits: Vector; settings: SampleSettings };

const number = (text: string) => (text.trim() === "" ? NaN : Number(text));

function Field({
  label,
  value,
  onChange,
  testId,
}: {
  label: string;
  value: string;
  onChange: (text: string) => void;
  testId: string;
}) {
  const id = useId();
  return (
    <div className={styles.control}>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        data-testid={testId}
        type="number"
        step="any"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}

// Results come from lib/sampling-lab.ts, the mirror of generate.sample_next, which tests compare with
// the real function. The lab stops before rng.choice, because that generator has no mirror.
export function ProbabilityLab({ presets }: { presets: LabPreset[] }) {
  const first = presets[0]!;
  const [logits, setLogits] = useState(first.logits);
  const [temperature, setTemperature] = useState(String(first.settings.temperature));
  const [topK, setTopK] = useState(String(first.settings.topK));
  const [topP, setTopP] = useState(String(first.settings.topP));
  // A new key resets what the logit cells hold while being typed in.
  const [version, setVersion] = useState(0);

  const load = (preset: LabPreset) => {
    setLogits(preset.logits);
    setTemperature(String(preset.settings.temperature));
    setTopK(String(preset.settings.topK));
    setTopP(String(preset.settings.topP));
    setVersion((value) => value + 1);
  };

  const settings = {
    temperature: number(temperature),
    topK: number(topK),
    topP: number(topP),
  };
  const problem = !Number.isFinite(settings.temperature)
    ? "temperature must be a number."
    : !Number.isInteger(settings.topK)
      ? "top_k must be a whole number. The code uses it as a list index."
      : !Number.isFinite(settings.topP)
        ? "top_p must be a number."
        : null;
  const view = problem === null ? labView(logits, settings) : null;
  const tokens = logits.map((_, index) => `token ${index}`);

  return (
    <div className={styles.stack} data-testid="probability-lab">
      <div className={styles.controls}>
        {presets.map((preset) => (
          <button
            type="button"
            key={preset.id}
            className={buttonClass()}
            data-testid={`lab-preset-${preset.id}`}
            onClick={() => load(preset)}
          >
            {preset.label}
          </button>
        ))}
      </div>
      <MatrixInput
        key={version}
        label="logits"
        rowLabels={["logits"]}
        colLabels={tokens}
        value={[logits]}
        onChange={(next) => setLogits(next[0]!)}
      />
      <div className={styles.controls}>
        <Field
          label="temperature"
          testId="lab-temperature"
          value={temperature}
          onChange={setTemperature}
        />
        <Field label="top_k" testId="lab-top-k" value={topK} onChange={setTopK} />
        <Field label="top_p" testId="lab-top-p" value={topP} onChange={setTopP} />
      </div>
      <div role="status" data-testid="lab-status">
        {problem ? (
          <p className={styles.error}>{problem}</p>
        ) : (
          <ol>
            {labSentences(view!, settings).map((sentence) => (
              <li key={sentence}>{sentence}</li>
            ))}
          </ol>
        )}
      </div>
      {view?.kind === "distribution" ? (
        <>
          <BarChart
            title="Before truncation: softmax of logits / temperature"
            kind="illustrative"
            caption="Hand-picked logits, not model output. Accent bars are the kept set. Grey bars are the tokens the later steps remove."
            labels={tokens}
            values={view.rows.map((row) => row.afterTemperature)}
            valueHeader="Probability"
            domain={[0, 1]}
            removed={view.rows.map((row) => !row.kept)}
          />
          <ProbabilityBars
            title="After truncation: what rng.choice receives"
            kind="illustrative"
            caption="The kept probabilities, scaled to sum to 1. Both charts run from 0 to 1, so the bars can be compared."
            labels={tokens}
            probabilities={view.rows.map((row) => row.final)}
            showRemoved
          />
          <DataTable label="Each token at each step" data-testid="lab-table">
            <thead>
              <tr>
                <th scope="col">Token</th>
                <th scope="col">logits</th>
                <th scope="col">After temperature</th>
                <th scope="col">After top-k</th>
                <th scope="col">Sort position</th>
                <th scope="col">cum</th>
                <th scope="col">Final p</th>
                <th scope="col">Kept</th>
              </tr>
            </thead>
            <tbody>
              {view.rows.map((row) => (
                <tr key={row.index} className={row.kept ? styles.chosen : undefined}>
                  <td>{tokens[row.index]}</td>
                  <td>
                    <code>{showNumber(row.logit)}</code>
                  </td>
                  <td>
                    <code>{showNumber(row.afterTemperature)}</code>
                  </td>
                  <td>
                    <code>{showNumber(row.afterTopK)}</code>
                  </td>
                  <td>{row.rank === null ? "Not run" : <code>{row.rank}</code>}</td>
                  <td>{row.cum === null ? "Not run" : <code>{showNumber(row.cum)}</code>}</td>
                  <td>
                    <code
                      data-testid={`lab-final-${row.index}`}
                      data-exact={JSON.stringify(row.final)}
                    >
                      {showNumber(row.final)}
                    </code>
                  </td>
                  <td>{row.kept ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        </>
      ) : null}
    </div>
  );
}

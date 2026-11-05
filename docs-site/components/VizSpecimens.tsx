"use client";

import { useId, useMemo, useState, type ChangeEvent } from "react";
import { attentionForward, type AttentionParams } from "@/lib/math/attention";
import { lrSchedule } from "@/lib/math/optim";
import { sampleNextDistribution } from "@/lib/math/sampling";
import type { Matrix, Vector } from "@/lib/math/tensor";
import { shapeOf } from "@/lib/viz/shape";
import { BarChart } from "./BarChart";
import { Heatmap } from "./Heatmap";
import { LineChart } from "./LineChart";
import { MatrixInput } from "./MatrixInput";
import { ProbabilityBars } from "./ProbabilityBars";
import { ShapeFlow } from "./ShapeFlow";
import { StepPlayer } from "./StepPlayer";
import styles from "./VizSpecimens.module.css";

// Control limits. These bound the inputs; they are not example values.
const TEMPERATURE_MAX = 2;
const TEMPERATURE_STEP = 0.05;
const TOP_P_STEP = 0.01;

export type SpecimenInputs = {
  attention: { Xn: Matrix; H: number; params: AttentionParams };
  sampling: { logits: Vector; temperature: number; topK: number; topP: number };
  schedule: { steps: number[]; warmup: number; total: number; lrMax: number; lrMin: number };
};

const KIND = "illustrative";

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  return (
    <div className={styles.slider}>
      <label htmlFor={id}>
        <code>{label}</code>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(Number(event.target.value))}
      />
      <output htmlFor={id}>
        <code>{value}</code>
      </output>
    </div>
  );
}

function AttentionSpecimens({ attention }: { attention: SpecimenInputs["attention"] }) {
  const [Xn, setXn] = useState(attention.Xn);
  const [head, setHead] = useState(0);
  const [selected, setSelected] = useState(0);
  const result = useMemo(
    () => attentionForward(Xn, attention.params, attention.H),
    [Xn, attention],
  );

  const tokens = Xn.map((_, t) => `t${t}`);
  const features = Xn[0]!.map((_, i) => `d${i}`);
  const heads = attention.H;
  const dh = shapeOf(result.Qh)[2]!;
  const weightMask = result.masked[head]!.map((row) => row.map((value) => value === -Infinity));

  const scoreMap = (title: string, values: Matrix, caption: string, masked?: boolean[][]) => (
    <Heatmap
      title={`${title}, head ${head}`}
      kind={KIND}
      caption={caption}
      rowLabels={tokens}
      colLabels={tokens}
      xLabel="column"
      yLabel="row"
      values={values}
      masked={masked}
    />
  );

  return (
    <>
      <h2 id="matrix-input">MatrixInput and ShapeFlow</h2>
      <MatrixInput
        label="Xn, the normalized input"
        rowLabels={tokens}
        colLabels={features}
        value={Xn}
        onChange={setXn}
      />
      <ShapeFlow
        title="Shapes in attention_forward"
        kind={KIND}
        stages={[
          { name: "Xn", shape: shapeOf(Xn) },
          { name: "Q", shape: shapeOf(result.Q) },
          { name: "Qh", shape: shapeOf(result.Qh) },
          { name: "S", shape: shapeOf(result.masked) },
          { name: "A", shape: shapeOf(result.A) },
          { name: "O", shape: shapeOf(result.O) },
          { name: "out", shape: shapeOf(result.out) },
        ]}
      />

      <h2 id="step-player">StepPlayer and Heatmap</h2>
      <label className={styles.select}>
        Head
        <select value={head} onChange={(event) => setHead(Number(event.target.value))}>
          {Array.from({ length: heads }, (_, h) => (
            <option key={h} value={h}>
              {h}
            </option>
          ))}
        </select>
      </label>
      <StepPlayer
        label="Attention steps"
        steps={[
          {
            id: "raw",
            title: "Scores",
            content: scoreMap("Qh @ Kh^T", result.raw[head]!, "Before scaling."),
          },
          {
            id: "scaled",
            title: "Scale",
            content: scoreMap(
              "S",
              result.scaled[head]!,
              `Divided by sqrt(dh), with dh = ${dh} for these inputs.`,
            ),
          },
          {
            id: "masked",
            title: "Mask",
            content: scoreMap("S after the mask", result.masked[head]!, "-inf where j > i."),
          },
          {
            id: "weights",
            title: "Softmax",
            content: scoreMap(
              "A",
              result.A[head]!,
              "Each row sums to 1. Masked cells are exactly 0.",
              weightMask,
            ),
          },
          {
            id: "values",
            title: "Weighted values",
            content: (
              <Heatmap
                title="O = merge_heads(A @ Vh)"
                kind={KIND}
                rowLabels={tokens}
                colLabels={features}
                xLabel="column"
                yLabel="row"
                values={result.O}
              />
            ),
          },
        ]}
      />

      <h2 id="selected-row">Selected row and BarChart</h2>
      <Heatmap
        title={`A, head ${head}, with a selected row`}
        kind={KIND}
        rowLabels={tokens}
        colLabels={tokens}
        xLabel="column"
        yLabel="row"
        values={result.A[head]!}
        masked={weightMask}
        selectedRow={selected}
        onSelectRow={setSelected}
      />
      <BarChart
        title={`Row ${tokens[selected]} of A`}
        kind={KIND}
        labels={tokens}
        values={result.A[head]![selected]!}
        valueHeader="Weight"
      />
    </>
  );
}

function SamplingSpecimen({ sampling }: { sampling: SpecimenInputs["sampling"] }) {
  const { logits } = sampling;
  const [temperature, setTemperature] = useState(sampling.temperature);
  const [topK, setTopK] = useState(sampling.topK);
  const [topP, setTopP] = useState(sampling.topP);
  const result = sampleNextDistribution(logits, { temperature, topK, topP });
  const labels = logits.map((_, i) => `id ${i}`);

  return (
    <>
      <h2 id="probability-bars">ProbabilityBars</h2>
      <div className={styles.sliders}>
        <Slider
          label="temperature"
          value={temperature}
          min={0}
          max={TEMPERATURE_MAX}
          step={TEMPERATURE_STEP}
          onChange={setTemperature}
        />
        <Slider
          label="top_k"
          value={topK}
          min={0}
          max={logits.length}
          step={1}
          onChange={setTopK}
        />
        <Slider label="top_p" value={topP} min={0} max={1} step={TOP_P_STEP} onChange={setTopP} />
      </div>
      {result.kind === "greedy" ? (
        <p data-testid="sampling-greedy">
          Temperature is below 1e-6, so <code>sample_next</code> returns the argmax, id{" "}
          {result.index}, and no distribution exists.
        </p>
      ) : (
        <>
          <ProbabilityBars
            title="Distribution given to rng.choice"
            kind={KIND}
            labels={labels}
            probabilities={result.p}
            showRemoved
          />
          {result.topP.tieDependent ? (
            <p data-testid="sampling-tie">
              The top-p cutoff falls inside a group of equal probabilities. NumPy&apos;s sort does
              not fix which of them stay, so the token ids kept by Python may differ from these. The
              kept count and the probabilities do not.
            </p>
          ) : null}
        </>
      )}
    </>
  );
}

function ScheduleSpecimen({ schedule }: { schedule: SpecimenInputs["schedule"] }) {
  const { steps, warmup, total, lrMax, lrMin } = schedule;
  const points = steps.map((step) => ({
    x: step,
    y: lrSchedule(step, warmup, total, lrMax, lrMin),
  }));
  const peak = steps.indexOf(warmup);

  return (
    <>
      <h2 id="line-chart">LineChart</h2>
      <LineChart
        title="lr_schedule"
        kind={KIND}
        caption={`warmup, lr_max and lr_min are the config.py values. total = ${total} is hand-picked, because the real total depends on the corpus.`}
        points={points}
        xLabel="step"
        yLabel="learning rate"
        yDomain={[0, lrMax]}
        labelled={[peak, points.length - 1]}
      />
    </>
  );
}

export function VizSpecimens({ inputs }: { inputs: SpecimenInputs }) {
  return (
    <div className={styles.stack}>
      <AttentionSpecimens attention={inputs.attention} />
      <SamplingSpecimen sampling={inputs.sampling} />
      <ScheduleSpecimen schedule={inputs.schedule} />
    </div>
  );
}

"use client";

import { useState } from "react";
import { layerNormForward, layerNormStats } from "@/lib/math/ops";
import type { Vector } from "@/lib/math/tensor";
import styles from "./Examples.module.css";
import { MatrixInput } from "./MatrixInput";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

function Stat({ label, testId, value }: { label: string; testId: string; value: number }) {
  return (
    <p data-testid={testId} data-exact={JSON.stringify(value)}>
      {label}: <code>{showNumber(value)}</code>
    </p>
  );
}

// One row, with gamma and beta. Every result comes from lib/math, which tests compare with the real
// layer_norm_forward. Epsilon is that function's default.
export function LayerNormLab({
  x: startX,
  gamma: startGamma,
  beta: startBeta,
}: {
  x: Vector;
  gamma: Vector;
  beta: Vector;
}) {
  const [x, setX] = useState(startX);
  const [gamma, setGamma] = useState(startGamma);
  const [beta, setBeta] = useState(startBeta);
  const features = x.map((_, i) => `i${i}`);
  const { mu, var: variance, s } = layerNormStats([x]);
  const { Y, cache } = layerNormForward([x], gamma, beta);

  return (
    <VizFigure
      title="One row through LayerNorm"
      kind="illustrative"
      caption="Hand-picked numbers from test_layer_norm_worked_example that you can edit. These are not learned values."
    >
      <div className={styles.matrices}>
        <MatrixInput
          label="X"
          rowLabels={["t0"]}
          colLabels={features}
          value={[x]}
          onChange={(next) => setX(next[0]!)}
        />
        <MatrixInput
          label="gamma"
          rowLabels={["gamma"]}
          colLabels={features}
          value={[gamma]}
          onChange={(next) => setGamma(next[0]!)}
        />
        <MatrixInput
          label="beta"
          rowLabels={["beta"]}
          colLabels={features}
          value={[beta]}
          onChange={(next) => setBeta(next[0]!)}
        />
      </div>
      <Stat label="Mean" testId="ln-mu" value={mu[0]![0]!} />
      <Stat label="Variance" testId="ln-var" value={variance[0]![0]!} />
      <Stat label="s" testId="ln-s" value={s[0]![0]!} />
      <p className={styles.status} role="status" data-testid="ln-status">
        {variance[0]![0] === 0
          ? "This row has no spread, so its variance is 0. Only epsilon keeps s finite, and the output is beta."
          : ""}
      </p>
      <div className={styles.matrices}>
        <ResultMatrix label="Xhat" testId="ln-xhat" values={cache.Xhat} />
        <ResultMatrix label="Output, gamma * Xhat + beta" testId="ln-output" values={Y} />
      </div>
    </VizFigure>
  );
}

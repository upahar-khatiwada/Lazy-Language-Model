"use client";

import { useState } from "react";
import { causalMask, tokenLabels, visibleToken, type ActualAttentionData } from "@/lib/actual-view";
import styles from "./AttentionLab.module.css";
import { BarChart } from "./BarChart";
import { Heatmap } from "./Heatmap";

// The Actual side of the lab. It holds its own selection and reads only exported data, so nothing
// from the Illustrative side can reach these panels.
export function ActualAttention({
  data,
  provenance,
}: {
  data: ActualAttentionData;
  provenance: string;
}) {
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [row, setRow] = useState(data.T - 1);
  const labels = tokenLabels(data.window.tokens);
  const A = data.layers[layer]!.heads[head]!.A;
  const weights = A[row]!;
  const sum = weights.reduce((total, value) => total + value, 0);
  const caption = `One real window, number ${data.window.index} of the ${data.window.split} split. These weights belong to this window only and say nothing about other text. Positions after the row are masked, so their weight is exactly 0.`;

  return (
    <div className={styles.panel} data-testid="actual-attention">
      <div className={styles.row} role="group" aria-label="Selected layer">
        <span className={styles.rowLabel}>Layer</span>
        {data.layers.map(({ layer: value }) => (
          <button
            type="button"
            className={styles.token}
            key={value}
            aria-pressed={value === layer}
            data-testid={`actual-layer-${value}`}
            onClick={() => setLayer(value)}
          >
            {value}
          </button>
        ))}
      </div>

      <div className={styles.row} role="group" aria-label="Selected head">
        <span className={styles.rowLabel}>Head</span>
        {data.layers[layer]!.heads.map(({ head: value }) => (
          <button
            type="button"
            className={styles.token}
            key={value}
            aria-pressed={value === head}
            data-testid={`actual-head-${value}`}
            onClick={() => setHead(value)}
          >
            {value}
          </button>
        ))}
      </div>

      <div className={styles.row} role="group" aria-label="Selected token">
        <span className={styles.rowLabel}>Token</span>
        {data.window.tokens.map((token, t) => (
          <button
            type="button"
            className={styles.token}
            key={t}
            aria-pressed={t === row}
            data-testid={`actual-token-${t}`}
            onClick={() => setRow(t)}
          >
            {visibleToken(token)}
          </button>
        ))}
      </div>

      <div data-testid="actual-matrix" data-exact={JSON.stringify(A)}>
        <Heatmap
          title={`Weights A, layer ${layer}, head ${head}`}
          kind="actual"
          provenance={provenance}
          caption={caption}
          rowLabels={labels}
          colLabels={labels}
          xLabel="reads"
          yLabel="row"
          values={A}
          masked={causalMask(data.T)}
          selectedRow={row}
        />
      </div>

      <div data-testid="actual-weights" data-exact={JSON.stringify(weights)}>
        <BarChart
          title={`Weights of position ${row}, layer ${layer}, head ${head}`}
          kind="actual"
          provenance={provenance}
          caption={`This row adds up to ${sum.toFixed(4)}.`}
          labels={labels}
          values={weights}
          valueHeader="Weight"
          domain={[0, 1]}
        />
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { crossEntropy, crossEntropyBackward } from "@/lib/math/ops";
import type { Vector } from "@/lib/math/tensor";
import styles from "./Examples.module.css";
import { MatrixInput } from "./MatrixInput";
import { ProbabilityBars } from "./ProbabilityBars";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

// One position of the loss: logits for three tokens and a chosen correct token. Every result comes
// from lib/math, which tests compare with the real cross_entropy_forward and cross_entropy_backward.
export function SoftmaxLab({ logits: startLogits }: { logits: Vector }) {
  const [logits, setLogits] = useState(startLogits);
  const [target, setTarget] = useState(0);
  const tokens = logits.map((_, v) => `token ${v}`);
  const { loss, probs } = crossEntropy([logits], [target]);
  const gradient = crossEntropyBackward(probs, [target]);

  return (
    <>
      <VizFigure
        title="Logits to probabilities, loss and gradient"
        kind="illustrative"
        caption="Hand-picked logits that you can edit. They are not model output. With one position, T is 1, so dZ is P minus the one-hot row."
      >
        <div className={styles.matrices}>
          <MatrixInput
            label="logits"
            rowLabels={["t0"]}
            colLabels={tokens}
            value={[logits]}
            onChange={(next) => setLogits(next[0]!)}
          />
          <label className={styles.control}>
            Correct token
            <select
              data-testid="lab-target"
              value={target}
              onChange={(event) => setTarget(Number(event.target.value))}
            >
              {tokens.map((token, v) => (
                <option key={token} value={v}>
                  {token}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p data-testid="lab-loss" data-exact={JSON.stringify(loss)}>
          Loss: <code>{showNumber(loss)}</code>
        </p>
        <div className={styles.matrices}>
          <ResultMatrix label="P = exp(logp)" testId="lab-probs" values={probs} />
          <ResultMatrix label="dZ = (P - one-hot) / T" testId="lab-gradient" values={gradient} />
        </div>
      </VizFigure>
      <ProbabilityBars
        title="Probabilities from these logits"
        kind="illustrative"
        caption={`The correct token is token ${target}. The loss falls as its bar grows.`}
        labels={tokens}
        probabilities={probs[0]!}
      />
    </>
  );
}

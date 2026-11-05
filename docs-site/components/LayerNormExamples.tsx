import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import { numericalGrad, numericalGradVector, outputDot } from "@/lib/math/gradcheck";
import { layerNormBackward, layerNormForward, layerNormJacobian } from "@/lib/math/ops";
import { matmul, transpose, type Matrix, type Vector } from "@/lib/math/tensor";
import styles from "./Examples.module.css";
import { LayerNormLab } from "./LayerNormLab";
import { GradientCheck, WorkedTable } from "./OpsExamples";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases. Results come from lib/math, which tests/math.golden.test.ts
// compares with the real functions at 1e-9.

const start = () => {
  const { X, gamma, beta } = getGoldenCase("ops", "layer_norm.test_file").inputs as {
    X: Matrix;
    gamma: Vector;
    beta: Vector;
  };
  const { dY } = getGoldenCase("ops", "layer_norm.test_file.backward").inputs as { dY: Matrix };
  return { X, gamma, beta, dY };
};

const mean = (values: Vector) => values.reduce((a, b) => a + b, 0) / values.length;

export function LayerNormLabExample({ concept }: { concept: string }) {
  const { X, gamma, beta } = start();
  return (
    <div id={exampleAnchor(concept)}>
      <LayerNormLab x={X[0]!} gamma={gamma} beta={beta} />
    </div>
  );
}

export function LayerNormBackwardExample({ concept }: { concept: string }) {
  const { X, gamma, beta, dY } = start();
  const { cache } = layerNormForward(X, gamma, beta);
  const { Xhat, s } = cache;
  const { dX, dgamma, dbeta } = layerNormBackward(dY, cache, gamma);
  const dXhat = dY.map((row) => row.map((value, i) => value * gamma[i]!));
  const firstMean = dXhat.map((row) => [mean(row)]);
  const secondMean = dXhat.map((row, t) => [mean(row.map((value, i) => value * Xhat[t]![i]!))]);
  // The first entry, written out from the numbers above.
  const parts = [dXhat[0]![0]!, firstMean[0]![0]!, Xhat[0]![0]!, secondMean[0]![0]!].map(
    showNumber,
  );

  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="A worked backward"
        kind="illustrative"
        caption="Hand-picked numbers from test_layer_norm_worked_example. dY is chosen by hand, not computed from a loss."
      >
        <WorkedTable
          label="LayerNorm backward, step by step"
          testId="ln-worked"
          rows={[
            { label: "Gradient on the output", code: "dY", value: dY },
            { label: "Cache", code: "Xhat", value: Xhat },
            { label: "Cache", code: "s", value: s },
            { label: "Gradient on the normalised row", code: "dXhat = dY * gamma", value: dXhat },
            { label: "First mean", code: "dXhat.mean(axis=-1, keepdims=True)", value: firstMean },
            {
              label: "Second mean",
              code: "(dXhat * Xhat).mean(axis=-1, keepdims=True)",
              value: secondMean,
            },
            { label: "Gradient on the input", code: "dX", value: dX, testId: "ln-dx" },
            {
              label: "Gradient on the scale",
              code: "dgamma = (dY * Xhat).sum(axis=0)",
              value: dgamma,
              testId: "ln-dgamma",
            },
            {
              label: "Gradient on the shift",
              code: "dbeta = dY.sum(axis=0)",
              value: dbeta,
              testId: "ln-dbeta",
            },
          ]}
        />
        <p data-testid="ln-entry">
          One entry:{" "}
          <code>
            dX[0][0] = {showNumber(s[0]![0]!)} × ({parts[0]} - {parts[1]} - ({parts[2]}) ×{" "}
            {parts[3]}) = {showNumber(dX[0]![0]!)}
          </code>
        </p>
      </VizFigure>
    </div>
  );
}

export function LayerNormDerivationExample({ concept }: { concept: string }) {
  const { X, gamma, beta, dY } = start();
  const { cache } = layerNormForward(X, gamma, beta);
  const { dX, dgamma, dbeta } = layerNormBackward(dY, cache, gamma);
  const x = X[0]!;
  const dXhat = dY[0]!.map((value, i) => value * gamma[i]!);
  const J = layerNormJacobian(cache.Xhat[0]!, cache.s[0]![0]!);
  const ones = x.map(() => 1);
  const zeros = x.map(() => 0);
  const numericJ = x.map((_, i) =>
    numericalGradVector((v) => layerNormForward([v], ones, zeros).cache.Xhat[0]![i]!, x),
  );
  const derived = matmul(transpose(J), transpose([dXhat]));
  const stored = getGoldenCase("ops", "layer_norm.test_file.backward").outputs.dX as Matrix;

  return (
    <div className={styles.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="The derived matrix against the code"
        kind="illustrative"
        caption="The same hand-picked row as the backward example. The matrix is step 5, multiplied through in step 6. The last result is stored from running the real layer_norm_backward."
      >
        <div className={styles.matrices}>
          <ResultMatrix label="Step 5, entry (i, j)" testId="ln-jacobian" values={J} />
          <ResultMatrix
            label="Central differences of the forward"
            testId="ln-jacobian-numeric"
            values={numericJ}
          />
          <ResultMatrix
            label="Step 6, as one row"
            testId="ln-derived-dx"
            values={transpose(derived)}
          />
          <ResultMatrix label="Real layer_norm_backward" testId="ln-real-dx" values={stored} />
        </div>
      </VizFigure>
      <GradientCheck
        title="Checking the whole backward numerically"
        testId="ln-check"
        scalar={
          <>
            <code>(Y * dY).sum()</code>, whose gradient on <code>Y</code> is exactly <code>dY</code>
            ,
          </>
        }
        rows={[
          {
            name: "dX",
            analytic: dX,
            numeric: numericalGrad((v) => outputDot(layerNormForward(v, gamma, beta).Y, dY), X),
          },
          {
            name: "dgamma",
            analytic: [dgamma],
            numeric: [
              numericalGradVector((v) => outputDot(layerNormForward(X, v, beta).Y, dY), gamma),
            ],
          },
          {
            name: "dbeta",
            analytic: [dbeta],
            numeric: [
              numericalGradVector((v) => outputDot(layerNormForward(X, gamma, v).Y, dY), beta),
            ],
          },
        ]}
      />
    </div>
  );
}

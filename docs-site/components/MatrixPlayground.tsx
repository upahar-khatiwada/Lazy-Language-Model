"use client";

import { useState } from "react";
import { linearBackward, linearForward } from "@/lib/math/ops";
import { matmul, type Matrix, type Vector } from "@/lib/math/tensor";
import { formatShape, shapeOf } from "@/lib/viz/shape";
import styles from "./Examples.module.css";
import { MatrixInput } from "./MatrixInput";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

const COLUMN_CHOICES = [2, 3];

// X and W are editable, and W can have two or three columns. With `dY`, the backward results for
// that dY are shown too. Every result comes from lib/math.
export function MatrixPlayground({
  X,
  W,
  b,
  dY,
  startColumns = COLUMN_CHOICES[0]!,
}: {
  X: Matrix;
  W: Matrix;
  b: Vector;
  dY?: Matrix;
  startColumns?: number;
}) {
  const [x, setX] = useState(X);
  const [w, setW] = useState(W);
  const [bias, setBias] = useState(b);
  const [gradient, setGradient] = useState(dY);
  const [columns, setColumns] = useState(startColumns);

  const shownW = w.map((row) => row.slice(0, columns));
  const shownB = bias.slice(0, columns);
  const shownDY = gradient?.map((row) => row.slice(0, columns));
  const product = matmul(x, shownW);
  const output = linearForward(x, shownW, shownB);
  const backward = shownDY ? linearBackward(x, shownW, shownDY) : null;
  const rowLabel = (i: number) => `t${i}`;
  const features = x[0]!.map((_, i) => `k${i}`);
  const outputs = shownW[0]!.map((_, i) => `j${i}`);
  const firstEntry = x[0]!.map((value, k) => `${value} × ${shownW[k]![0]}`).join(" + ");

  return (
    <VizFigure
      title={backward ? "X @ W + b, and its backward" : "X @ W and X @ W + b"}
      kind="illustrative"
      caption="Hand-picked numbers that you can edit. These are not learned weights. Results come from the math engine, and tests compare it with the real Python on the starting values."
    >
      <label className={styles.control}>
        Columns of W
        <select
          data-testid="matmul-columns"
          value={columns}
          onChange={(event) => setColumns(Number(event.target.value))}
        >
          {COLUMN_CHOICES.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      </label>
      <div className={styles.matrices} key={columns}>
        <MatrixInput
          label="X"
          rowLabels={x.map((_, i) => rowLabel(i))}
          colLabels={features}
          value={x}
          onChange={setX}
        />
        <MatrixInput
          label="W"
          rowLabels={features}
          colLabels={outputs}
          value={shownW}
          onChange={(next) => setW(w.map((row, r) => [...next[r]!, ...row.slice(next[r]!.length)]))}
        />
        <MatrixInput
          label="b"
          rowLabels={["b"]}
          colLabels={outputs}
          value={[shownB]}
          onChange={(next) => setBias([...next[0]!, ...bias.slice(next[0]!.length)])}
        />
        {gradient && shownDY ? (
          <MatrixInput
            label="dY"
            rowLabels={x.map((_, i) => rowLabel(i))}
            colLabels={outputs}
            value={shownDY}
            onChange={(next) =>
              setGradient(gradient.map((row, r) => [...next[r]!, ...row.slice(next[r]!.length)]))
            }
          />
        ) : null}
      </div>
      <p data-testid="matmul-rule">
        <code>
          {formatShape(shapeOf(x))} @ {formatShape(shapeOf(shownW))} →{" "}
          {formatShape(shapeOf(product))}
        </code>
        : columns of X must equal rows of W. The result has the rows of X and the columns of W.
      </p>
      <p data-testid="matmul-entry">
        Row 0, column 0 of the product:{" "}
        <code>
          {firstEntry} = {showNumber(product[0]![0]!)}
        </code>
      </p>
      <div className={styles.matrices}>
        <ResultMatrix label="X @ W" testId="matmul-product" values={product} />
        <ResultMatrix label="X @ W + b" testId="matmul-output" values={output} />
        {backward ? (
          <>
            <ResultMatrix label="dW = X.T @ dY" testId="linear-dw" values={backward.dW} />
            <ResultMatrix label="db = dY.sum(axis=0)" testId="linear-db" values={[backward.db]} />
            <ResultMatrix label="dX = dY @ W.T" testId="linear-dx" values={backward.dX} />
          </>
        ) : null}
      </div>
    </VizFigure>
  );
}

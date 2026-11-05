"use client";

import { useState } from "react";
import { linearForward } from "@/lib/math/ops";
import { matmul, type Matrix, type Vector } from "@/lib/math/tensor";
import { formatShape, shapeOf } from "@/lib/viz/shape";
import styles from "./Examples.module.css";
import { MatrixInput } from "./MatrixInput";
import { VizFigure } from "./VizFigure";

const COLUMN_CHOICES = [2, 3];

const showNumber = (value: number) => String(Number(value.toFixed(4)));

function Result({ label, testId, values }: { label: string; testId: string; values: Matrix }) {
  return (
    <table className={styles.compact} data-testid={testId}>
      <caption>
        {label}, shape {formatShape(shapeOf(values))}
      </caption>
      <tbody>
        {values.map((row, t) => (
          <tr key={t}>
            {row.map((value, c) => (
              <td key={c}>{showNumber(value)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// X and W are editable, and W can have two or three columns. Every result comes from lib/math.
export function MatrixPlayground({ X, W, b }: { X: Matrix; W: Matrix; b: Vector }) {
  const [x, setX] = useState(X);
  const [w, setW] = useState(W);
  const [bias, setBias] = useState(b);
  const [columns, setColumns] = useState(COLUMN_CHOICES[0]!);

  const shownW = w.map((row) => row.slice(0, columns));
  const shownB = bias.slice(0, columns);
  const product = matmul(x, shownW);
  const output = linearForward(x, shownW, shownB);
  const rowLabel = (i: number) => `t${i}`;
  const features = x[0]!.map((_, i) => `k${i}`);
  const outputs = shownW[0]!.map((_, i) => `j${i}`);
  const firstEntry = x[0]!.map((value, k) => `${value} × ${shownW[k]![0]}`).join(" + ");

  return (
    <VizFigure
      title="X @ W and X @ W + b"
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
        <Result label="X @ W" testId="matmul-product" values={product} />
        <Result label="X @ W + b" testId="matmul-output" values={output} />
      </div>
    </VizFigure>
  );
}

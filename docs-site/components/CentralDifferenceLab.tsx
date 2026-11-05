"use client";

import { useState } from "react";
import { numericalGradVector, relativeError } from "@/lib/math/gradcheck";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { VizFigure } from "./VizFigure";

const FUNCTIONS = [
  {
    name: "x * x",
    power: 2,
    f: (v: number[]) => v[0]! * v[0]!,
    analytic: (x: number) => 2 * x,
    formula: "2x",
  },
  {
    name: "x * x * x",
    power: 3,
    f: (v: number[]) => v[0]! * v[0]! * v[0]!,
    analytic: (x: number) => 3 * x * x,
    formula: "3x^2",
  },
];

const parse = (text: string) => (text.trim() === "" ? NaN : Number(text));

function problem(x: number, h: number): string {
  if (!Number.isFinite(x) || !Number.isFinite(h)) return "Enter a number in both fields.";
  if (h <= 0) return "h must be above 0. The code divides by 2 * h.";
  return "";
}

// Every result comes from lib/math, which tests compare with the real gradcheck functions.
export function CentralDifferenceLab({ x: startX, h: startH }: { x: number; h: number }) {
  // Text, not numbers, so that "1e-" can be typed on the way to "1e-5".
  const [xText, setXText] = useState(String(startX));
  const [hText, setHText] = useState(String(startH));
  const x = parse(xText);
  const h = parse(hText);
  const message = problem(x, h);

  return (
    <VizFigure
      title="Analytic against numerical"
      kind="illustrative"
      caption="Hand-picked functions and numbers that you can edit. Neither function is in the repository: x * x matches the test's quadratic for one entry, and x * x * x is here because h changes its result."
    >
      <div className={styles.controls}>
        <label className={styles.control}>
          x
          <input
            type="number"
            step="any"
            value={xText}
            onChange={(event) => setXText(event.target.value)}
          />
        </label>
        <label className={styles.control}>
          h
          <input
            type="number"
            step="any"
            value={hText}
            onChange={(event) => setHText(event.target.value)}
          />
        </label>
      </div>
      <p className={styles.status} role="status" data-testid="cd-status">
        {message}
      </p>
      <DataTable label="Analytic and numerical gradient" data-testid="cd-table">
        <thead>
          <tr>
            <th scope="col">f(x)</th>
            <th scope="col">Analytic</th>
            <th scope="col">Central difference</th>
            <th scope="col">Relative error</th>
          </tr>
        </thead>
        <tbody>
          {FUNCTIONS.map(({ name, power, f, analytic, formula }) => {
            const exact = message ? null : analytic(x);
            const numeric = message ? null : numericalGradVector(f, [x], h)[0]!;
            const error = exact === null || numeric === null ? null : relativeError(exact, numeric);
            return (
              <tr key={name} data-power={power}>
                <td>
                  <code className={styles.nowrap}>{name}</code>
                </td>
                <td>
                  <code className={styles.nowrap}>
                    {formula}
                    {exact === null ? "" : ` = ${exact}`}
                  </code>
                </td>
                <td data-testid={`cd-numeric-${power}`} data-exact={JSON.stringify(numeric)}>
                  <code>{numeric === null ? "not defined" : numeric}</code>
                </td>
                <td data-testid={`cd-error-${power}`} data-exact={JSON.stringify(error)}>
                  <code className={styles.nowrap}>
                    {error === null ? "not defined" : error.toExponential(2)}
                  </code>
                </td>
              </tr>
            );
          })}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

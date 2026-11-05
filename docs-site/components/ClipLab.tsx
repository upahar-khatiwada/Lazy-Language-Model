"use client";

import { useId, useState } from "react";
import { clipGradNorm } from "@/lib/math/optim";
import { BarChart } from "./BarChart";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";

function parseValues(text: string): number[] | null {
  const parts = text.split(",").map((part) => part.trim());
  const values = parts.map(Number);
  return parts.every((part) => part !== "") && values.every(Number.isFinite) ? values : null;
}

const parseLimit = (text: string) => {
  const value = Number(text);
  return text.trim() !== "" && Number.isFinite(value) && value > 0 ? value : null;
};

const show = (values: number[]) => values.map((value) => Number(value.toFixed(4))).join(", ");

// Results come from lib/math/optim.ts, which tests compare with the real clip_grad_norm.
export function ClipLab({ a, b, maxNorm }: { a: number[]; b: number[]; maxNorm: number }) {
  const [aText, setAText] = useState(a.join(", "));
  const [bText, setBText] = useState(b.join(", "));
  const [limitText, setLimitText] = useState(String(maxNorm));
  const aField = useId();
  const bField = useId();
  const limitField = useId();

  const first = parseValues(aText);
  const second = parseValues(bText);
  const limit = parseLimit(limitText);
  const problem =
    first === null || second === null
      ? "Enter each tensor as numbers separated by commas."
      : limit === null
        ? "max_norm must be a number above 0 here."
        : null;

  const result = problem === null ? clipGradNorm({ a: first!, b: second! }, limit!) : null;
  const after = result
    ? Math.sqrt(
        [...(result.grads.a as number[]), ...(result.grads.b as number[])].reduce(
          (total, value) => total + value * value,
          0,
        ),
      )
    : 0;
  const clipped = result !== null && result.norm > limit!;
  const factor = clipped ? limit! / result!.norm : null;

  return (
    <div className={styles.stack} data-testid="clip-lab">
      <div className={styles.controls}>
        <div className={styles.control}>
          <label htmlFor={aField}>Tensor a</label>
          <input
            id={aField}
            data-testid="clip-a"
            type="text"
            value={aText}
            onChange={(event) => setAText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={bField}>Tensor b</label>
          <input
            id={bField}
            data-testid="clip-b"
            type="text"
            value={bText}
            onChange={(event) => setBText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={limitField}>max_norm</label>
          <input
            id={limitField}
            data-testid="clip-limit"
            type="number"
            step="any"
            min={0}
            value={limitText}
            onChange={(event) => setLimitText(event.target.value)}
          />
        </div>
      </div>
      <p className={styles.status} role="status" data-testid="clip-status">
        {problem ??
          (clipped ? (
            <>
              <code>norm</code> is above <code>max_norm</code>, so every tensor is multiplied by{" "}
              <code className={styles.nowrap}>max_norm / norm = {Number(factor!.toFixed(6))}</code>.
            </>
          ) : (
            "norm is not above max_norm, so no tensor changes."
          ))}
      </p>
      {result ? (
        <>
          <BarChart
            title="Length of the whole gradient"
            kind="illustrative"
            caption="Hand-picked values that you can edit. max_norm starts at grad_clip from config.py."
            labels={["norm before", "max_norm", "norm after"]}
            values={[result.norm, limit!, after]}
            valueHeader="Length"
          />
          <DataTable label="Each tensor before and after" data-testid="clip-tensors">
            <thead>
              <tr>
                <th scope="col">Tensor</th>
                <th scope="col">Before</th>
                <th scope="col">After</th>
              </tr>
            </thead>
            <tbody>
              {(["a", "b"] as const).map((name) => (
                <tr key={name}>
                  <td>
                    <code>{name}</code>
                  </td>
                  <td>
                    <code>{show((name === "a" ? first : second)!)}</code>
                  </td>
                  <td>
                    <code
                      data-testid={`clip-after-${name}`}
                      data-exact={JSON.stringify(result.grads[name])}
                    >
                      {show(result.grads[name] as number[])}
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
          <ul className={styles.results}>
            <li>
              <code data-testid="clip-norm" data-exact={JSON.stringify(result.norm)}>
                norm = {Number(result.norm.toFixed(6))}
              </code>
              , the value <code>clip_grad_norm</code> returns
            </li>
            {factor === null ? null : (
              <li>
                <code data-testid="clip-factor" data-exact={JSON.stringify(factor)}>
                  max_norm / norm = {Number(factor.toFixed(6))}
                </code>
                , the factor for every tensor
              </li>
            )}
          </ul>
        </>
      ) : null}
    </div>
  );
}

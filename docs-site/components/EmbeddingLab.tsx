"use client";

import { useState } from "react";
import { embedBackward, embedForward, overwritingBackward } from "@/lib/math/embeddings";
import type { Matrix } from "@/lib/math/tensor";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

function sameId(ids: number[]): string {
  for (const id of new Set(ids)) {
    const places = ids.flatMap((value, t) => (value === id ? [t] : []));
    if (places.length > 1) {
      const sum = places.map((t) => `dX0[${t}]`).join(" + ");
      return `Id ${id} is at positions ${places.join(" and ")}, so row ${id} of the token gradient is ${sum}.`;
    }
  }
  return "No id repeats, so each token row receives at most one row of dX0.";
}

// Pick ids and a length. Every result comes from lib/math, which tests compare with the real
// embed_forward and embed_backward.
export function EmbeddingLab({
  tokEmb,
  posEmb,
  ids: startIds,
  dX0: fullDX0,
}: {
  tokEmb: Matrix;
  posEmb: Matrix;
  ids: number[];
  dX0: Matrix;
}) {
  const [allIds, setAllIds] = useState(startIds);
  const [T, setT] = useState(startIds.length);
  const ids = allIds.slice(0, T);
  const dX0 = fullDX0.slice(0, T);
  const { tokens, positions, X0 } = embedForward(tokEmb, posEmb, ids);
  const { tok, pos } = embedBackward(dX0, ids, tokEmb.length, posEmb.length);
  const overwritten = overwritingBackward(dX0, ids, tokEmb.length);

  return (
    <VizFigure
      title="Pick ids, and follow them both ways"
      kind="illustrative"
      caption="The tables and dX0 are the hand-picked numbers of test_embedding_worked_example and test_embedding_backward_sums_repeated_ids. They are not learned values, and dX0 is chosen by hand, not computed from a loss. The page starts with the id repeated, as those tests do."
    >
      <div className={styles.controls}>
        <label className={styles.control}>
          T
          <select
            data-testid="lab-length"
            value={T}
            onChange={(event) => setT(Number(event.target.value))}
          >
            {posEmb.map((_, t) => (
              <option key={t} value={t + 1}>
                {t + 1}
              </option>
            ))}
          </select>
        </label>
        {ids.map((id, t) => (
          <label className={styles.control} key={t}>
            ids[{t}]
            <select
              data-testid={`lab-id-${t}`}
              value={id}
              onChange={(event) => {
                const next = [...allIds];
                next[t] = Number(event.target.value);
                setAllIds(next);
              }}
            >
              {tokEmb.map((_, v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      <DataTable label="The token table, with the rows these ids select" data-testid="lab-table">
        <thead>
          <tr>
            <th scope="col">Id</th>
            <th scope="col">Row of tok_emb</th>
            <th scope="col">Selected at</th>
          </tr>
        </thead>
        <tbody>
          {tokEmb.map((row, v) => {
            const places = ids.flatMap((value, t) => (value === v ? [t] : []));
            return (
              <tr key={v} data-testid={`lab-row-${v}`} data-selected={places.length > 0}>
                <th scope="row">{v}</th>
                <td>
                  <code className={places.length > 0 ? styles.chosen : undefined}>
                    [{row.map(showNumber).join(", ")}]
                  </code>
                </td>
                <td>{places.length > 0 ? places.map((t) => `t = ${t}`).join(", ") : "not used"}</td>
              </tr>
            );
          })}
        </tbody>
      </DataTable>

      <div className={styles.matrices}>
        <ResultMatrix label="tok_emb[ids]" testId="lab-tokens" values={tokens} />
        <ResultMatrix label="pos_emb[:T]" testId="lab-positions" values={positions} />
        <ResultMatrix label="X0, their sum" testId="lab-x0" values={X0} />
      </div>

      <p className={styles.status} role="status" data-testid="lab-repeat">
        {sameId(ids)}
      </p>

      <div className={styles.matrices}>
        <ResultMatrix label="dX0" testId="lab-dx0" values={dX0} />
        <ResultMatrix label='grads["tok_emb"], np.add.at' testId="lab-dtok" values={tok} />
        <ResultMatrix label='grads["pos_emb"]' testId="lab-dpos" values={pos} />
        <ResultMatrix
          label="Wrong: grads[ids] += dX0"
          testId="lab-overwritten"
          values={overwritten}
        />
      </div>
    </VizFigure>
  );
}

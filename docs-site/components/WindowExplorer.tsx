"use client";

import { useId, useState } from "react";
import { makeWindows, windowCount } from "@/lib/math/data";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { VizFigure } from "./VizFigure";

// Keeps the table readable. It is a limit of this control, not of make_windows.
const MAX_IDS = 40;

function parseIds(text: string): number[] | null {
  const parts = text.split(",").map((part) => part.trim());
  return parts.every((part) => /^\d+$/.test(part)) ? parts.map(Number) : null;
}

function parseCount(text: string): number | null {
  return /^[1-9]\d*$/.test(text.trim()) ? Number(text) : null;
}

function problemWith(ids: number[] | null, seqLen: number | null, stride: number | null) {
  if (ids === null) return "Enter token ids as whole numbers separated by commas.";
  if (ids.length > MAX_IDS) return `This example takes up to ${MAX_IDS} ids.`;
  if (seqLen === null) return "Window length must be a whole number of at least 1.";
  if (stride === null) return "Stride must be a whole number of at least 1.";
  return null;
}

// Every value comes from lib/math/data.ts, which tests compare with the real make_windows.
export function WindowExplorer({
  ids: startIds,
  seqLen: startSeqLen,
  stride: startStride,
}: {
  ids: number[];
  seqLen: number;
  stride: number;
}) {
  const [idsText, setIdsText] = useState(startIds.join(", "));
  const [seqLenText, setSeqLenText] = useState(String(startSeqLen));
  const [strideText, setStrideText] = useState(String(startStride));
  const idsField = useId();
  const seqLenField = useId();
  const strideField = useId();

  const ids = parseIds(idsText);
  const seqLen = parseCount(seqLenText);
  const stride = parseCount(strideText);
  const problem = problemWith(ids, seqLen, stride);
  const ready = problem === null && ids !== null && seqLen !== null && stride !== null;
  // Python raises for this input too, so the example shows the same message.
  const tooFew =
    ready && ids.length <= seqLen
      ? `need more than seq_len=${seqLen} tokens, got ${ids.length}`
      : null;
  const windows = ready && !tooFew ? makeWindows(ids, seqLen, stride) : null;
  const shifted =
    windows?.inputs.every((row, w) =>
      row.slice(1).every((value, t) => value === windows.targets[w]![t]),
    ) ?? false;

  return (
    <VizFigure
      title="Sliding windows"
      kind="illustrative"
      caption="Hand-picked ids that you can edit. They are not ids from the corpus. Tests compare the engine behind this table with the real make_windows."
    >
      <div className={styles.controls}>
        <div className={`${styles.control} ${styles.fieldWide}`}>
          <label htmlFor={idsField}>Token ids</label>
          <input
            id={idsField}
            className={styles.fieldWide}
            data-testid="window-ids"
            type="text"
            inputMode="numeric"
            value={idsText}
            onChange={(event) => setIdsText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={seqLenField}>Window length T</label>
          <input
            id={seqLenField}
            data-testid="window-length"
            type="number"
            min={1}
            value={seqLenText}
            onChange={(event) => setSeqLenText(event.target.value)}
          />
        </div>
        <div className={styles.control}>
          <label htmlFor={strideField}>Stride k</label>
          <input
            id={strideField}
            data-testid="window-stride"
            type="number"
            min={1}
            value={strideText}
            onChange={(event) => setStrideText(event.target.value)}
          />
        </div>
      </div>

      <p className={styles.status} role="status" data-testid="window-status">
        {problem ?? (tooFew ? `make_windows raises ValueError: ${tooFew}` : "")}
      </p>

      {windows && ids && seqLen && stride ? (
        <>
          <p data-testid="window-count">
            <code>
              N = {ids.length}, W = ceil(({ids.length} - {seqLen}) / {stride}) ={" "}
              {windowCount(ids.length, seqLen, stride)}
            </code>
          </p>
          <DataTable label="Windows" data-testid="window-table">
            <thead>
              <tr>
                <th scope="col">Start s</th>
                <th scope="col">Inputs</th>
                <th scope="col">Targets</th>
              </tr>
            </thead>
            <tbody>
              {windows.starts.map((start, w) => (
                <tr key={start}>
                  <td>{start}</td>
                  <td>
                    <code>{windows.inputs[w]!.join(", ")}</code>
                  </td>
                  <td>
                    <code>{windows.targets[w]!.join(", ")}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
          <p data-testid="window-shift">
            {shifted
              ? "In every row, the targets are the inputs moved one place left, with one new id at the end."
              : "The targets differ from the shifted inputs."}
          </p>
        </>
      ) : null}
    </VizFigure>
  );
}

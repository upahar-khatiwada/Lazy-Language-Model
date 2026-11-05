"use client";

import { useId, useState } from "react";
import styles from "./MatrixInput.module.css";

const parse = (text: string) => (text.trim() === "" ? NaN : Number(text));

// Every cell reports on input; there is no submit button. A cell that does not hold a finite number
// (empty, or "-" while typing) is marked and the matrix keeps its last valid value.
export function MatrixInput({
  label,
  rowLabels,
  colLabels,
  value,
  onChange,
}: {
  label: string;
  rowLabels: string[];
  colLabels: string[];
  value: number[][];
  onChange: (next: number[][]) => void;
}) {
  const statusId = useId();
  // What the reader typed, per cell, while it differs from the number. "1." must stay "1.".
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const invalid = Object.values(drafts).filter((text) => !Number.isFinite(parse(text))).length;

  const edit = (r: number, c: number, text: string) => {
    setDrafts((current) => ({ ...current, [`${r},${c}`]: text }));
    const number = parse(text);
    if (!Number.isFinite(number)) return;
    onChange(value.map((row, i) => (i === r ? row.map((v, j) => (j === c ? number : v)) : row)));
  };

  // Leaving a cell shows its number in plain form; a bad entry stays visible and marked.
  const settle = (r: number, c: number) => {
    setDrafts((current) => {
      const key = `${r},${c}`;
      if (!(key in current) || !Number.isFinite(parse(current[key]!))) return current;
      const next = { ...current };
      delete next[key];
      return next;
    });
  };

  return (
    <div className={styles.wrapper}>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <caption className={styles.caption}>{label}</caption>
          <thead>
            <tr>
              <td className={styles.corner} />
              {colLabels.map((colLabel) => (
                <th scope="col" key={colLabel}>
                  {colLabel}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {value.map((row, r) => (
              <tr key={rowLabels[r]}>
                <th scope="row">{rowLabels[r]}</th>
                {row.map((cell, c) => {
                  const draft = drafts[`${r},${c}`];
                  const bad = draft !== undefined && !Number.isFinite(parse(draft));
                  return (
                    <td key={colLabels[c]}>
                      <input
                        className={styles.input}
                        type="number"
                        step="any"
                        aria-label={`${label}, row ${rowLabels[r]}, column ${colLabels[c]}`}
                        aria-invalid={bad || undefined}
                        aria-describedby={bad ? statusId : undefined}
                        value={draft ?? String(cell)}
                        onChange={(event) => edit(r, c, event.target.value)}
                        onBlur={() => settle(r, c)}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.status} id={statusId} role="status" data-testid="matrix-status">
        {invalid > 0 ? "Enter a number. Results still use the last valid value of that cell." : ""}
      </p>
    </div>
  );
}

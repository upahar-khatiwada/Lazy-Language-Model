import type { ReactNode } from "react";
import { Badge } from "./Badge";
import styles from "./VizFigure.module.css";

export type VizKind = "illustrative" | "actual";

// What every numeric chart needs from its page. `kind` is required, as on NumericExample: MDX is
// not type-checked, so a missing or wrong label also fails at render time.
export type VizLabel = {
  title: string;
  kind: VizKind;
  provenance?: string;
  caption?: ReactNode;
};

export type VizTable = { headers: string[]; rows: (string | number)[][] };

export function VizFigure({
  title,
  kind,
  provenance,
  caption,
  table,
  children,
}: VizLabel & { table?: VizTable; children: ReactNode }) {
  if (kind !== "illustrative" && kind !== "actual") {
    throw new Error(`Chart "${title}" needs kind="illustrative" or kind="actual"`);
  }
  if (kind === "actual" && !provenance) {
    throw new Error(`Chart "${title}" shows Actual Model Output, so it needs a provenance`);
  }

  return (
    <figure className={styles.figure}>
      <figcaption className={styles.caption}>
        <span className={styles.title}>{title}</span>
        <Badge kind={kind} />
        {provenance ? <span className={styles.note}>{provenance}</span> : null}
        {caption ? <span className={styles.note}>{caption}</span> : null}
      </figcaption>
      {children}
      {table ? (
        <details className={styles.details}>
          <summary className={styles.summary}>Data table</summary>
          <div className={styles.tableScroll}>
            <table>
              <thead>
                <tr>
                  {table.headers.map((header) => (
                    <th scope="col" key={header}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, r) => (
                  <tr key={r}>
                    {row.map((cell, c) => (
                      <td key={c}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </figure>
  );
}

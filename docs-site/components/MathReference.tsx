import { Fragment } from "react";
import Link from "next/link";
import { loadConcepts } from "@/lib/concepts";
import {
  buildReferenceRow,
  coverage,
  DIRECTION_LABELS,
  FILTER_LABELS,
  loadInventory,
  readPageSources,
  type ReferenceRow,
} from "@/lib/reference";
import { Badge } from "./Badge";
import { MathBlock } from "./MathBlock";
import { MathReferenceTable } from "./MathReferenceTable";
import styles from "./MathReference.module.css";

function Label({ children }: { children: string }) {
  return <span className={styles.cellLabel}>{children}</span>;
}

function Missing({ children }: { children: string }) {
  return <span className={styles.missing}>{children}</span>;
}

function Cells({ row }: { row: ReferenceRow }) {
  return (
    <>
      <td role="cell">
        <span className={styles.title}>{row.title}</span>
        <Badge kind={row.status} />
        <span className={styles.direction}>
          <Label>Direction</Label>
          {row.directions.map((direction) => DIRECTION_LABELS[direction]).join(", ")}
        </span>
      </td>
      <td role="cell" className={styles.equations}>
        <Label>Equation</Label>
        {row.equationIds.map((id) => (
          <MathBlock id={id} key={id} />
        ))}
      </td>
      <td role="cell">
        <Label>File and function</Label>
        {row.sources.map((source) => (
          <span className={styles.source} key={source.id}>
            <code>{source.file}</code>
            <code>{source.name}</code>
          </span>
        ))}
      </td>
      <td role="cell">
        <Label>Tensor shapes</Label>
        {row.shapes.length === 0 ? <Missing>None declared</Missing> : null}
        {row.shapes.map((shape) => (
          <span className={styles.shape} key={shape.id}>
            <a className={styles.shapeLabel} href={shape.permalink}>
              {shape.label}
            </a>
            {shape.stages.map((stage) => (
              <span className={styles.stage} key={stage.name}>
                <code>{stage.name}</code> <code>{stage.symbolic}</code>
                {stage.values ? <code className={styles.values}>{stage.values}</code> : null}
              </span>
            ))}
          </span>
        ))}
      </td>
      <td role="cell">
        <Label>Links</Label>
        <span className={styles.links}>
          {row.explanation ? (
            <Link prefetch={false} href={row.explanation}>
              Explanation
            </Link>
          ) : (
            <Missing>Explanation not written</Missing>
          )}
          {row.example ? (
            <Link prefetch={false} href={row.example}>
              Numerical example
            </Link>
          ) : (
            <Missing>
              {row.hasExample ? "Numerical example not on a page yet" : "No numerical example"}
            </Missing>
          )}
          {row.sources.map((source) => (
            <a className="code-link" href={source.permalink} key={source.id}>
              {source.file}:{source.lines}
            </a>
          ))}
        </span>
      </td>
    </>
  );
}

// Generated from data/concepts. Rows are rendered here, at build time, so no KaTeX reaches the
// browser; the client component only shows or hides them.
export function MathReference() {
  const concepts = loadConcepts();
  const rows = concepts.map((concept) => buildReferenceRow(concept, readPageSources()));
  const { documented, total } = coverage(concepts, loadInventory().rows);

  return (
    <div className={styles.reference}>
      <p className={styles.coverage} data-testid="coverage">
        {documented} of {total} operations documented
      </p>
      <MathReferenceTable
        filters={Object.entries(FILTER_LABELS).map(([id, label]) => ({
          id,
          label,
          count: rows.filter((row) => row.filter === id).length,
        }))}
        directions={Object.entries(DIRECTION_LABELS).map(([id, label]) => ({ id, label }))}
        rows={rows.map((row) => ({
          id: row.id,
          filter: row.filter,
          directions: row.directions,
          search: row.search,
          cells: (
            <Fragment>
              <Cells row={row} />
            </Fragment>
          ),
        }))}
      />
    </div>
  );
}

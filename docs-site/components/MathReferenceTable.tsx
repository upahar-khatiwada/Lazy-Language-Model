"use client";

import { useState, type ReactNode } from "react";
import styles from "./MathReference.module.css";

type Props = {
  filters: { id: string; label: string; count: number }[];
  directions: { id: string; label: string }[];
  rows: {
    id: string;
    filter: string;
    directions: string[];
    search: string;
    cells: ReactNode;
  }[];
};

const COLUMNS = ["Name and direction", "Equation", "File and function", "Tensor shapes", "Links"];

// Every row is in the page without JavaScript. Filtering only hides rows, so each equation keeps
// its anchor. Roles are explicit because the table is laid out as cards on a phone, and changing
// `display` would otherwise remove its table semantics.
export function MathReferenceTable({ filters, directions, rows }: Props) {
  const [text, setText] = useState("");
  const [filter, setFilter] = useState("");
  const [direction, setDirection] = useState("");

  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const matches = (row: Props["rows"][number]) =>
    (!filter || row.filter === filter) &&
    (!direction || row.directions.includes(direction)) &&
    words.every((word) => row.search.includes(word));
  const visible = rows.filter(matches).length;

  return (
    <>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Filter by text</span>
          <input
            type="search"
            value={text}
            onChange={(event) => setText(event.target.value)}
            data-testid="reference-text"
          />
        </label>
        <label className={styles.control}>
          <span>Category</span>
          <select
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            data-testid="reference-category"
          >
            <option value="">All categories</option>
            {filters.map(({ id, label, count }) => (
              <option value={id} key={id}>
                {label} ({count})
              </option>
            ))}
          </select>
        </label>
        <label className={styles.control}>
          <span>Direction</span>
          <select
            value={direction}
            onChange={(event) => setDirection(event.target.value)}
            data-testid="reference-direction"
          >
            <option value="">Any direction</option>
            {directions.map(({ id, label }) => (
              <option value={id} key={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className={styles.count} role="status" data-testid="reference-count">
        Showing {visible} of {rows.length}
      </p>
      <table role="table" className={styles.table} data-testid="reference-table">
        <thead role="rowgroup">
          <tr role="row">
            {COLUMNS.map((column) => (
              <th role="columnheader" scope="col" key={column}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody role="rowgroup">
          {rows.map((row) => (
            <tr role="row" hidden={!matches(row)} data-concept={row.id} key={row.id}>
              {row.cells}
            </tr>
          ))}
        </tbody>
      </table>
      {visible === 0 ? <p className={styles.empty}>No concept matches these filters.</p> : null}
    </>
  );
}

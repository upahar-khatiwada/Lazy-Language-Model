import { useId } from "react";
import { sequentialStep, SEQUENTIAL_STEPS } from "@/lib/viz/scale";
import styles from "./Heatmap.module.css";
import { VizFigure, type VizLabel } from "./VizFigure";

const CELL_W = 56;
const CELL_H = 40;
const LABEL_W = 56;
const HEADER_H = 44;

export type HeatmapProps = VizLabel & {
  rowLabels: string[];
  colLabels: string[];
  xLabel: string;
  yLabel: string;
  values: number[][];
  // Cells to draw as masked although they hold a number, such as attention weights of 0. A value
  // of -Infinity is always masked.
  masked?: boolean[][];
  precision?: number;
  selectedRow?: number;
  onSelectRow?: (row: number) => void;
};

// Fixed cell sizes, so labels keep their size. A wide matrix scrolls inside its own container.
export function Heatmap({
  rowLabels,
  colLabels,
  xLabel,
  yLabel,
  values,
  masked,
  precision = 2,
  selectedRow,
  onSelectRow,
  ...label
}: HeatmapProps) {
  const id = useId();
  const hatch = `${id}-hatch`;
  const isMasked = (r: number, c: number) =>
    values[r]![c] === -Infinity || masked?.[r]?.[c] === true;
  const format = (r: number, c: number) => {
    const value = values[r]![c]!;
    return value === -Infinity ? "-inf" : value.toFixed(precision);
  };

  const visible = values.flatMap((row, r) => row.filter((_, c) => !isMasked(r, c)));
  const low = visible.length ? Math.min(...visible) : 0;
  const high = visible.length ? Math.max(...visible) : 0;
  const width = LABEL_W + colLabels.length * CELL_W;
  const height = HEADER_H + rowLabels.length * CELL_H;

  return (
    <VizFigure
      {...label}
      table={{
        headers: [`${yLabel} \\ ${xLabel}`, ...colLabels],
        rows: rowLabels.map((rowLabel, r) => [
          rowLabel,
          ...colLabels.map((_, c) => format(r, c) + (isMasked(r, c) ? " (masked)" : "")),
        ]),
      }}
    >
      <div
        className={styles.scroll}
        tabIndex={0}
        role="group"
        aria-label={`${label.title}, matrix`}
      >
        <svg
          className={styles.chart}
          width={width}
          height={height}
          role="img"
          aria-label={`${label.title}: ${rowLabels.length} rows by ${colLabels.length} columns. The values are in the data table.`}
        >
          <defs>
            <pattern
              id={hatch}
              width="8"
              height="8"
              patternUnits="userSpaceOnUse"
              patternTransform="rotate(45)"
            >
              <line className={styles.hatchLine} x1="0" y1="0" x2="0" y2="8" />
            </pattern>
          </defs>
          <text className={styles.axis} x={LABEL_W} y="14">
            {xLabel} →
          </text>
          <text className={styles.axis} x="0" y={HEADER_H - 8}>
            {yLabel} ↓
          </text>
          {colLabels.map((colLabel, c) => (
            <text
              className={styles.label}
              key={colLabel}
              x={LABEL_W + c * CELL_W + CELL_W / 2}
              y={HEADER_H - 8}
              textAnchor="middle"
            >
              {colLabel}
            </text>
          ))}
          {rowLabels.map((rowLabel, r) => (
            <text
              className={r === selectedRow ? styles.selectedLabel : styles.label}
              key={rowLabel}
              x="0"
              y={HEADER_H + r * CELL_H + CELL_H / 2 + 5}
            >
              {rowLabel}
            </text>
          ))}
          {values.map((row, r) =>
            row.map((value, c) => {
              const x = LABEL_W + c * CELL_W;
              const y = HEADER_H + r * CELL_H;
              const off = isMasked(r, c);
              const step = sequentialStep(low, high, value);
              return (
                <g key={c} data-masked={off || undefined}>
                  <title>{`${rowLabels[r]}, ${colLabels[c]}: ${format(r, c)}${off ? " (masked)" : ""}`}</title>
                  <rect
                    className={styles.cell}
                    x={x + 1}
                    y={y + 1}
                    width={CELL_W - 2}
                    height={CELL_H - 2}
                    rx="2"
                    style={{ fill: off ? `url(#${hatch})` : `var(--seq-${step})` }}
                  />
                  <text
                    className={styles.value}
                    x={x + CELL_W / 2}
                    y={y + CELL_H / 2 + 5}
                    textAnchor="middle"
                    style={{ fill: off ? "var(--muted)" : `var(--seq-ink-${step})` }}
                  >
                    {format(r, c)}
                  </text>
                </g>
              );
            }),
          )}
          {selectedRow !== undefined ? (
            <rect
              className={styles.selected}
              data-testid="heatmap-selected-row"
              x={LABEL_W}
              y={HEADER_H + selectedRow * CELL_H}
              width={colLabels.length * CELL_W}
              height={CELL_H}
              rx="3"
            />
          ) : null}
        </svg>
      </div>
      <p className={styles.legend}>
        <span>Low {low.toFixed(precision)}</span>
        <span className={styles.swatches} aria-hidden="true">
          {Array.from({ length: SEQUENTIAL_STEPS }, (_, step) => (
            <span key={step} style={{ background: `var(--seq-${step})` }} />
          ))}
        </span>
        <span>High {high.toFixed(precision)}</span>
      </p>
      {onSelectRow ? (
        <fieldset className={styles.rows}>
          <legend>Selected row</legend>
          {rowLabels.map((rowLabel, r) => (
            <label className={styles.rowOption} key={rowLabel}>
              <input
                type="radio"
                name={`${id}-row`}
                checked={r === selectedRow}
                onChange={() => onSelectRow(r)}
              />
              {rowLabel}
            </label>
          ))}
        </fieldset>
      ) : null}
    </VizFigure>
  );
}

import { linearPercent, shortNumber } from "@/lib/viz/scale";
import styles from "./LineChart.module.css";
import { VizFigure, type VizLabel } from "./VizFigure";

export type LinePoint = { x: number; y: number };

export type LineChartProps = VizLabel & {
  points: LinePoint[];
  xLabel: string;
  yLabel: string;
  // Indices of the points to mark and label. No other value is printed on the plot except the two
  // ends of each axis: there are no interpolated ticks.
  labelled?: number[];
  // The y range. It defaults to the data range and always contains the data.
  yDomain?: [number, number];
};

const HEIGHT = 220;

// One linear scale per axis. The line, the markers and the labels all use the same two scales.
export function LineChart({
  points,
  xLabel,
  yLabel,
  labelled = [],
  yDomain,
  ...label
}: LineChartProps) {
  if (points.length === 0) throw new Error(`Chart "${label.title}" has no points`);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xScale: [number, number] = [Math.min(...xs), Math.max(...xs)];
  const yScale: [number, number] = [
    Math.min(yDomain?.[0] ?? Infinity, ...ys),
    Math.max(yDomain?.[1] ?? -Infinity, ...ys),
  ];
  const left = (x: number) => linearPercent(xScale, x);
  const top = (y: number) => 100 - linearPercent(yScale, y);

  return (
    <VizFigure
      {...label}
      table={{
        headers: [xLabel, yLabel],
        rows: points.map((point) => [point.x, shortNumber(point.y)]),
      }}
    >
      <p className={styles.yTitle}>{yLabel}</p>
      <div className={styles.plot}>
        <svg
          className={styles.chart}
          width="100%"
          height={HEIGHT}
          role="img"
          aria-label={`${label.title}: ${yLabel} against ${xLabel}, ${points.length} points. The values are in the data table.`}
        >
          <line className={styles.axis} x1="0" y1="0" x2="0" y2="100%" />
          <line className={styles.axis} x1="0" y1="100%" x2="100%" y2="100%" />
          <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
            <polyline
              className={styles.line}
              data-testid="line-path"
              points={points.map((point) => `${left(point.x)},${top(point.y)}`).join(" ")}
            />
          </svg>
          <text className={styles.tick} x="-8" y="10" textAnchor="end">
            {shortNumber(yScale[1])}
          </text>
          <text className={styles.tick} x="-8" y="100%" textAnchor="end">
            {shortNumber(yScale[0])}
          </text>
          {labelled.map((index) => {
            const point = points[index];
            if (!point) throw new Error(`Chart "${label.title}" has no point ${index}`);
            const x = left(point.x);
            const y = top(point.y);
            return (
              <g key={index} data-testid="line-label">
                <circle className={styles.marker} cx={`${x}%`} cy={`${y}%`} r="4" />
                <text
                  className={styles.pointLabel}
                  x={`${x}%`}
                  y={`${y}%`}
                  dx={x > 50 ? -8 : 8}
                  dy={y < 15 ? 20 : -10}
                  textAnchor={x > 50 ? "end" : "start"}
                >
                  {`${point.x}: ${shortNumber(point.y)}`}
                </text>
              </g>
            );
          })}
        </svg>
        <div className={styles.xAxis}>
          <span>{shortNumber(xScale[0])}</span>
          <span className={styles.xTitle}>{xLabel}</span>
          <span>{shortNumber(xScale[1])}</span>
        </div>
      </div>
    </VizFigure>
  );
}

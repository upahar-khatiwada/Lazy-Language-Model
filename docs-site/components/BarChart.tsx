import { linearPercent } from "@/lib/viz/scale";
import styles from "./BarChart.module.css";
import { VizFigure, type VizLabel } from "./VizFigure";

export type BarChartProps = VizLabel & {
  labels: string[];
  values: number[];
  valueHeader?: string;
  // The value range. It always includes 0, so every bar starts from the zero line.
  domain?: [number, number];
  precision?: number;
  // Bars drawn as removed, which is also said in words.
  removed?: boolean[];
};

// Labels are HTML and only the bars are SVG, so text keeps its size at any width.
export function BarChart({
  labels,
  values,
  valueHeader = "Value",
  domain,
  precision = 4,
  removed,
  ...label
}: BarChartProps) {
  const low = Math.min(0, domain?.[0] ?? 0, ...values);
  const high = Math.max(0, domain?.[1] ?? 0, ...values);
  const scale: [number, number] = [low, high];
  const zero = linearPercent(scale, 0);
  const format = (i: number) => values[i]!.toFixed(precision) + (removed?.[i] ? " removed" : "");

  return (
    <VizFigure
      {...label}
      table={{
        headers: ["Label", valueHeader],
        rows: labels.map((name, i) => [name, format(i)]),
      }}
    >
      <div
        className={styles.bars}
        role="img"
        aria-label={`${label.title}: ${labels.map((name, i) => `${name} ${format(i)}`).join(", ")}`}
      >
        {labels.map((name, i) => {
          const end = linearPercent(scale, values[i]!);
          return (
            <div className={styles.row} key={name} data-testid="bar-row">
              <code className={styles.name}>{name}</code>
              <svg className={styles.track} width="100%" height="20" aria-hidden="true">
                <rect
                  className={removed?.[i] ? styles.barRemoved : styles.bar}
                  x={`${Math.min(zero, end)}%`}
                  y="2"
                  width={`${Math.abs(end - zero)}%`}
                  height="16"
                />
                <line className={styles.zero} x1={`${zero}%`} x2={`${zero}%`} y1="0" y2="20" />
              </svg>
              <code className={styles.value}>{format(i)}</code>
            </div>
          );
        })}
      </div>
    </VizFigure>
  );
}

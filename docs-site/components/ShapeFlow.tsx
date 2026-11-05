import { formatShape } from "@/lib/viz/shape";
import styles from "./ShapeFlow.module.css";
import { VizFigure, type VizLabel } from "./VizFigure";

export type ShapeStage = { name: string; shape: (string | number)[] };

const BOX_H = 40;
const ARROW_H = 20;

// Boxes top to bottom, one per tensor. Pass real shapes: shapeOf(value) from lib/viz/shape.
export function ShapeFlow({ stages, ...label }: VizLabel & { stages: ShapeStage[] }) {
  const height = stages.length * BOX_H + Math.max(0, stages.length - 1) * ARROW_H;
  const description = stages
    .map((stage) => `${stage.name} ${formatShape(stage.shape)}`)
    .join(", then ");

  return (
    <VizFigure {...label}>
      <svg
        className={styles.chart}
        width="100%"
        height={height}
        role="img"
        aria-label={`${label.title}: ${description}`}
      >
        {stages.map((stage, i) => {
          const y = i * (BOX_H + ARROW_H);
          return (
            <g key={stage.name} data-testid="shape-stage">
              <rect
                className={styles.box}
                x="1"
                y={y + 1}
                width="100%"
                height={BOX_H - 2}
                rx="4"
                style={{ width: "calc(100% - 2px)" }}
              />
              <text className={styles.name} x="12" y={y + BOX_H / 2 + 5}>
                {stage.name}
              </text>
              <text
                className={styles.shape}
                x="100%"
                dx="-12"
                y={y + BOX_H / 2 + 5}
                textAnchor="end"
              >
                {formatShape(stage.shape)}
              </text>
              {i < stages.length - 1 ? (
                <svg x="50%" y={y + BOX_H} overflow="visible">
                  <path
                    className={styles.arrow}
                    d={`M0 2V${ARROW_H - 3}M-4 ${ARROW_H - 7}L0 ${ARROW_H - 3}L4 ${ARROW_H - 7}`}
                  />
                </svg>
              ) : null}
            </g>
          );
        })}
      </svg>
    </VizFigure>
  );
}

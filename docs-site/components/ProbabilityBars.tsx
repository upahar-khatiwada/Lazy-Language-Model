import { BarChart } from "./BarChart";
import type { VizLabel } from "./VizFigure";

// A distribution: the axis always runs from 0 to 1, so two charts can be compared by eye.
// Tokens with probability 0 are marked as removed when `showRemoved` is set.
export function ProbabilityBars({
  labels,
  probabilities,
  showRemoved = false,
  ...label
}: VizLabel & { labels: string[]; probabilities: number[]; showRemoved?: boolean }) {
  return (
    <BarChart
      {...label}
      labels={labels}
      values={probabilities}
      valueHeader="Probability"
      domain={[0, 1]}
      removed={showRemoved ? probabilities.map((value) => value === 0) : undefined}
    />
  );
}

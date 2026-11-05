import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import type { Matrix, Vector } from "@/lib/math/tensor";
import { MatrixPlayground } from "./MatrixPlayground";

// The starting values are the inputs of a golden case, so the first result on screen has a value
// from the real Python to compare with.
export function MatrixExample({ concept }: { concept: string }) {
  const { inputs } = getGoldenCase("ops", "matmul.three_columns");
  return (
    <div id={exampleAnchor(concept)}>
      <MatrixPlayground X={inputs.X as Matrix} W={inputs.W as Matrix} b={inputs.b as Vector} />
    </div>
  );
}

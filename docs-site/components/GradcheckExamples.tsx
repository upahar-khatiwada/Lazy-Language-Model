import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import { numericalGradVector, relativeError } from "@/lib/math/gradcheck";
import type { Vector } from "@/lib/math/tensor";
import { CentralDifferenceLab } from "./CentralDifferenceLab";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { WorkedTable } from "./OpsExamples";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases. Central differences and relative errors come from lib/math,
// which tests/math.golden.test.ts compares with the real functions. check_params samples with NumPy's
// generator, which has no mirror, so its tables show the stored result of running the real function.

export function CentralDifferenceExample({ concept }: { concept: string }) {
  const { x } = getGoldenCase("gradcheck", "numerical_grad.test_quadratic").inputs as { x: Vector };
  const g = numericalGradVector((v) => v.reduce((total, value) => total + value * value, 0), x);
  const { x: labX, h: labH } = getGoldenCase("gradcheck", "numerical_grad.cube_default_h")
    .inputs as { x: number; h: number };

  return (
    <div className={styles.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="The test's quadratic"
        kind="illustrative"
        caption="The hand-picked input of test_numerical_grad_on_quadratic. The function is the sum of squares, so its gradient is 2x."
      >
        <WorkedTable
          label="Central difference on the test's quadratic"
          testId="cd-quadratic"
          rows={[
            { label: "Input", code: "x", value: x },
            { label: "Analytic gradient", code: "2 * x", value: x.map((value) => 2 * value) },
            { label: "Central difference", code: "numerical_grad(f, x)", value: g, testId: "cd-g" },
          ]}
        />
      </VizFigure>
      <CentralDifferenceLab x={labX} h={labH} />
    </div>
  );
}

const RELATIVE_ERROR_ROWS = [
  { id: "relative_error.close", note: "An ordinary pair." },
  { id: "relative_error.equal", note: "Equal numbers give 0." },
  { id: "relative_error.opposite_signs", note: "2 is the largest value this formula can give." },
  { id: "relative_error.both_zero", note: "The guard keeps this at 0 rather than 0 / 0." },
  { id: "relative_error.guard_used", note: "Both are below eps, so eps is the denominator." },
  {
    id: "relative_error.tiny_against_small",
    note: "Both are near 0, but b is above eps, so the result is close to 1.",
  },
];

export function RelativeErrorExample({ concept }: { concept: string }) {
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="relative_error on six pairs"
        kind="illustrative"
        caption="Hand-picked pairs. eps is the function's default."
      >
        <DataTable label="Relative error examples" data-testid="relative-error">
          <thead>
            <tr>
              <th scope="col">a</th>
              <th scope="col">b</th>
              <th scope="col">Result</th>
              <th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            {RELATIVE_ERROR_ROWS.map(({ id, note }) => {
              const { a, b } = getGoldenCase("gradcheck", id).inputs as { a: number; b: number };
              const error = relativeError(a, b);
              return (
                <tr key={id} data-case={id}>
                  <td>
                    <code>{a}</code>
                  </td>
                  <td>
                    <code>{b}</code>
                  </td>
                  <td data-testid={`rel-${id}`} data-exact={JSON.stringify(error)}>
                    <code className={styles.nowrap}>
                      {error === 0 ? "0" : error.toPrecision(4)}
                    </code>
                  </td>
                  <td>{note}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

// The limit in test_every_tensor_passes. tests/gradient-checking.test.ts compares it with the source.
export const TEST_LIMIT = 1e-4;

function WorstCell({ name, value }: { name: string; value: number }) {
  const over = value >= TEST_LIMIT && !name.endsWith(".bk");
  return (
    <td data-testid={`worst-${name}`} data-exact={JSON.stringify(value)} data-over={over}>
      <code className={over ? `${styles.nowrap} ${styles.chosen}` : styles.nowrap}>
        {value.toExponential(1)}
      </code>
      {over ? " above" : null}
    </td>
  );
}

export function CheckParamsExample({ concept }: { concept: string }) {
  const worst = (id: string) =>
    getGoldenCase("gradcheck", id).outputs.worst as Record<string, number>;
  const { names, loss_evaluations: evaluations } = getGoldenCase(
    "gradcheck",
    "check_params.correct",
  ).outputs as { names: string[]; loss_evaluations: number };
  const correct = worst("check_params.correct");
  const oneTensor = worst("check_params.one_tensor_wrong");
  const blockOutput = worst("check_params.block_output_wrong");
  const { factor } = getGoldenCase("gradcheck", "check_params.one_tensor_wrong").inputs as {
    factor: number;
  };

  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Worst relative error per name"
        kind="illustrative"
        caption={
          <>
            Stored results of the real <code>check_params</code> on the hand-picked tiny model of{" "}
            <code>tiny_setup</code>, with {names.length} names and {evaluations} loss evaluations
            per column. This is not the trained model. The second and third columns scale a gradient
            by {factor} on purpose, in the script that makes this table only. Values far below the
            limit are round-off and can differ between machines. The test skips the rows marked{" "}
            <code>.bk</code>.
          </>
        }
      >
        <DataTable label="Worst relative error per parameter name" data-testid="check-params">
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Correct backward</th>
              <th scope="col">
                One tensor wrong: <code>grads[&quot;b1.wq&quot;]</code>
              </th>
              <th scope="col">Block output wrong: what block 1 returns</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name}>
                <th scope="row">
                  <code>{name}</code>
                  {name.endsWith(".bk") ? " (skipped)" : null}
                </th>
                <WorstCell name={name} value={correct[name]!} />
                <WorstCell name={name} value={oneTensor[name]!} />
                <WorstCell name={name} value={blockOutput[name]!} />
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

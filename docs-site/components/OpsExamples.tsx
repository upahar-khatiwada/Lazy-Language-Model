import type { ReactNode } from "react";
import { exampleAnchor } from "@/lib/concepts";
import { formatNumbers } from "@/lib/examples";
import { getGoldenCase } from "@/lib/golden";
import { numericalGrad, numericalGradVector, outputDot } from "@/lib/math/gradcheck";
import {
  crossEntropy,
  crossEntropyBackward,
  linearBackward,
  linearForward,
  logSoftmax,
  perplexity,
  relu,
  reluBackward,
  softmax,
  softmaxBackwardRows,
  xavierLimit,
} from "@/lib/math/ops";
import type { Matrix, Vector } from "@/lib/math/tensor";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { MatrixPlayground } from "./MatrixPlayground";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { SoftmaxLab } from "./SoftmaxLab";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases. Results come from lib/math, which tests/math.golden.test.ts
// compares with the real functions at 1e-9. Where lib/math has no mirror (NumPy's generator), the
// value is the stored result of running the real function, and the caption says so.

type Nested = number | Nested[];

const plainNumbers = (value: Nested): string =>
  typeof value === "number" ? showNumber(value) : `[${value.map(plainNumbers).join(", ")}]`;

// `plain` shows 2 rather than 2.0000, for ids and other values where the extra digits would mislead.
export function Value({
  value,
  testId,
  plain = false,
}: {
  value: Nested;
  testId?: string;
  plain?: boolean;
}) {
  return (
    <code data-testid={testId} data-exact={JSON.stringify(value)}>
      {plain ? plainNumbers(value) : formatNumbers(value, 4)}
    </code>
  );
}

export type WorkedRow = { label: ReactNode; code?: string; value: Nested; testId?: string };

// One row per step, in the order the code runs them.
export function WorkedTable({
  label,
  testId,
  rows,
  plain = false,
}: {
  label: string;
  testId: string;
  rows: WorkedRow[];
  plain?: boolean;
}) {
  return (
    <DataTable label={label} data-testid={testId}>
      <thead>
        <tr>
          <th scope="col">Step</th>
          <th scope="col">Code</th>
          <th scope="col">Value</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={index}>
            <td>{row.label}</td>
            <td>{row.code ? <code>{row.code}</code> : null}</td>
            <td>
              <Value value={row.value} testId={row.testId} plain={plain} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

// A matrix where each row starts a new line. On a phone the rows do not wrap, so the table scrolls.
function MatrixLines({ value }: { value: Matrix }) {
  return (
    <code className={styles.matrixLines} data-exact={JSON.stringify(value)}>
      {value.map((row, r) => (
        <span key={r}>{formatNumbers(row, 4)}</span>
      ))}
    </code>
  );
}

export type CheckRow = { name: string; analytic: Matrix; numeric: Matrix };

const flat = (m: Matrix) => m.flat();

// The largest gap between the two, for one tensor.
export function largestGap({ analytic, numeric }: CheckRow): number {
  const left = flat(analytic);
  const right = flat(numeric);
  return Math.max(...left.map((value, i) => Math.abs(value - right[i]!)));
}

// The gradient-check idea. gradcheck.py is test-only (decision D2), so this says so where it appears.
export function GradientCheck({
  title,
  testId,
  scalar,
  rows,
}: {
  title: string;
  testId: string;
  scalar: ReactNode;
  rows: CheckRow[];
}) {
  return (
    <VizFigure
      title={title}
      kind="illustrative"
      caption={
        <>
          A test-time idea, not part of the pipeline: <code>gradcheck.numerical_grad</code> is only
          called by tests. It nudges each number up and down by <code>h = 1e-5</code> and measures
          how {scalar} changes. Tests compare the central differences here with that real function.
        </>
      }
    >
      <DataTable label={title} data-testid={testId} wrapClassName={styles.checkScroll}>
        <thead>
          <tr>
            <th scope="col">Gradient</th>
            <th scope="col">From the backward formula</th>
            <th scope="col">From central differences</th>
            <th scope="col">Largest gap</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name} data-gradient={row.name}>
              <td>
                <code>{row.name}</code>
              </td>
              <td>
                <MatrixLines value={row.analytic} />
              </td>
              <td>
                <MatrixLines value={row.numeric} />
              </td>
              <td data-testid="largest-gap" data-exact={largestGap(row)}>
                <code className={styles.nowrap}>{largestGap(row).toExponential(1)}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

// Linear ------------------------------------------------------------------------------------------

const linearInputs = () =>
  getGoldenCase("ops", "linear.test_file").inputs as { X: Matrix; W: Matrix; b: Vector };
const linearGradient = () =>
  getGoldenCase("ops", "linear.test_file.backward").inputs as { dY: Matrix };

export function LinearPlaygroundExample({ concept }: { concept: string }) {
  const { X, W, b } = linearInputs();
  return (
    <div id={exampleAnchor(concept)}>
      <MatrixPlayground X={X} W={W} b={b} dY={linearGradient().dY} startColumns={3} />
    </div>
  );
}

export function LinearBackwardExample({ concept }: { concept: string }) {
  const { X, W, b } = linearInputs();
  const { dY } = linearGradient();
  const { dX, dW, db } = linearBackward(X, W, dY);
  // The first entry of each product, written out from the same numbers.
  const firstDW = X.map((row, t) => `${row[0]} × ${dY[t]![0]}`).join(" + ");
  const firstDX = dY[0]!.map((value, j) => `${value} × ${W[0]![j]}`).join(" + ");
  const firstDb = dY.map((row) => row[0]).join(" + ");

  return (
    <div className={styles.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="A worked backward"
        kind="illustrative"
        caption="Hand-picked numbers from test_linear_worked_example. They are not learned weights, and dY here is chosen by hand, not computed from a loss."
      >
        <WorkedTable
          label="Linear backward, step by step"
          testId="linear-worked"
          rows={[
            { label: "Input", code: "X", value: X },
            { label: "Weights", code: "W", value: W },
            { label: "Gradient on the output", code: "dY", value: dY },
            {
              label: "Gradient on the weights",
              code: "dW = X.T @ dY",
              value: dW,
              testId: "worked-dw",
            },
            {
              label: "Gradient on the bias",
              code: "db = dY.sum(axis=0)",
              value: db,
              testId: "worked-db",
            },
            {
              label: "Gradient on the input",
              code: "dX = dY @ W.T",
              value: dX,
              testId: "worked-dx",
            },
          ]}
        />
        <p data-testid="linear-entries">
          One entry of each:{" "}
          <code>
            dW[0][0] = {firstDW} = {showNumber(dW[0]![0]!)}
          </code>
          ,{" "}
          <code>
            db[0] = {firstDb} = {showNumber(db[0]!)}
          </code>
          ,{" "}
          <code>
            dX[0][0] = {firstDX} = {showNumber(dX[0]![0]!)}
          </code>
          .
        </p>
      </VizFigure>
      <GradientCheck
        title="Checking this backward numerically"
        testId="linear-check"
        scalar={
          <>
            <code>(Y * dY).sum()</code>, whose gradient on <code>Y</code> is exactly <code>dY</code>
            ,
          </>
        }
        rows={[
          {
            name: "dW",
            analytic: dW,
            numeric: numericalGrad((v) => outputDot(linearForward(X, v, b), dY), W),
          },
          {
            name: "db",
            analytic: [db],
            numeric: [numericalGradVector((v) => outputDot(linearForward(X, W, v), dY), b)],
          },
          {
            name: "dX",
            analytic: dX,
            numeric: numericalGrad((v) => outputDot(linearForward(v, W, b), dY), X),
          },
        ]}
      />
    </div>
  );
}

// ReLU --------------------------------------------------------------------------------------------

export function ReluExample({ concept }: { concept: string }) {
  const { Z, dA } = getGoldenCase("ops", "relu_backward.test_file").inputs as {
    Z: Matrix;
    dA: Matrix;
  };
  const backward = concept.endsWith("backward");
  const mask = Z.map((row) => row.map((value) => (value > 0 ? 1 : 0)));
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title={backward ? "ReLU backward" : "ReLU"}
        kind="illustrative"
        caption="Hand-picked numbers from test_relu. The third value is exactly 0, which is the case the strict > decides."
      >
        <WorkedTable
          label={backward ? "ReLU backward, step by step" : "ReLU, step by step"}
          testId={backward ? "relu-backward" : "relu-forward"}
          rows={
            backward
              ? [
                  { label: "Input from the forward pass", code: "Z", value: Z },
                  { label: "Gradient on the output", code: "dA", value: dA },
                  { label: "Mask", code: "Z > 0", value: mask },
                  {
                    label: "Gradient on the input",
                    code: "dA * (Z > 0)",
                    value: reluBackward(Z, dA),
                    testId: "relu-dz",
                  },
                ]
              : [
                  { label: "Input", code: "X", value: Z },
                  {
                    label: "Output",
                    code: "np.maximum(0.0, X)",
                    value: relu(Z),
                    testId: "relu-output",
                  },
                ]
          }
        />
      </VizFigure>
    </div>
  );
}

// Initialisation ----------------------------------------------------------------------------------

const XAVIER_ROWS = [
  { id: "xavier.tiny", label: "Tiny, hand-picked" },
  { id: "xavier.attention", label: "w{q,k,v,o}" },
  { id: "xavier.feed_forward_in", label: "ff1_w" },
  { id: "xavier.feed_forward_out", label: "ff2_w" },
  { id: "xavier.position_table", label: "pos_emb" },
];

export function XavierExample({ concept }: { concept: string }) {
  const sample = getGoldenCase("ops", "xavier.tiny").outputs.sample as Matrix;
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="The limit for each matrix"
        kind="illustrative"
        caption="The sizes after the first row come from config.py. The limit is computed by the math engine. The lowest and highest values are stored results of running the real xavier_uniform with a generator seeded from config.py, and they are not the values of a trained or real model."
      >
        <DataTable label="Xavier limits" data-testid="xavier-example">
          <thead>
            <tr>
              <th scope="col">Matrix</th>
              <th scope="col">fan_in</th>
              <th scope="col">fan_out</th>
              <th scope="col">Limit a</th>
              <th scope="col">Lowest drawn</th>
              <th scope="col">Highest drawn</th>
            </tr>
          </thead>
          <tbody>
            {XAVIER_ROWS.map(({ id, label }) => {
              const { inputs, outputs } = getGoldenCase("ops", id);
              const { fan_in, fan_out } = inputs as { fan_in: number; fan_out: number };
              const limit = xavierLimit(fan_in, fan_out);
              return (
                <tr key={id} data-case={id}>
                  <td>
                    <code>{label}</code>
                  </td>
                  <td>{fan_in}</td>
                  <td>{fan_out}</td>
                  <td data-testid="xavier-limit" data-exact={limit}>
                    {limit.toFixed(4)}
                  </td>
                  <td>{(outputs.low as number).toFixed(4)}</td>
                  <td>{(outputs.high as number).toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
        <ResultMatrix label="The tiny matrix itself" testId="xavier-sample" values={sample} />
      </VizFigure>
    </div>
  );
}

export function InitExample({ concept }: { concept: string }) {
  const { inputs, outputs } = getGoldenCase("ops", "init_params.tiny");
  const rows = outputs.rows as { name: string; shape: number[]; low: number; high: number }[];
  const { V, Tmax, d, H, F, L } = inputs as Record<string, number>;
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="What init_params returned"
        kind="illustrative"
        caption={`Stored result of running the real init_params with hand-picked sizes: V = ${V}, context_len = ${Tmax}, emb_dim = ${d}, num_heads = ${H}, ff_dim = ${F}, num_layers = ${L}. These are not the model's values.`}
      >
        <DataTable label="Lowest and highest value of each array" data-testid="init-example">
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Shape</th>
              <th scope="col">Lowest</th>
              <th scope="col">Highest</th>
              <th scope="col">Xavier limit</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ name, shape, low, high }) => (
              <tr key={name} data-name={name}>
                <td>
                  <code>{name}</code>
                </td>
                <td>
                  <code>
                    ({shape.join(", ")}
                    {shape.length === 1 ? "," : ""})
                  </code>
                </td>
                <td>{low.toFixed(4)}</td>
                <td>{high.toFixed(4)}</td>
                <td>
                  {shape.length === 2
                    ? xavierLimit(shape[0]!, shape[1]!).toFixed(4)
                    : "does not apply"}
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

// Softmax -----------------------------------------------------------------------------------------

const SHIFT_ROWS = [
  "softmax_forward.basic",
  "softmax_forward.basic_moved",
  "softmax_forward.small_same_gaps",
  "softmax_forward.large_positive",
];

export function SoftmaxShiftTable() {
  return (
    <VizFigure
      title="Moving all scores together changes nothing, and large scores stay finite"
      kind="illustrative"
      caption="Hand-picked scores. The last column computes the exponential of the largest score with no shift: at 1002 it does not fit in a 64-bit float, so normalising would give NaN."
    >
      <DataTable label="Softmax of moved scores" data-testid="softmax-shift">
        <thead>
          <tr>
            <th scope="col">Scores x</th>
            <th scope="col">softmax_forward(x)</th>
            <th scope="col">exp(max x), with no shift</th>
          </tr>
        </thead>
        <tbody>
          {SHIFT_ROWS.map((id) => {
            const x = getGoldenCase("ops", id).inputs.x as Vector;
            const unshifted = Math.exp(Math.max(...x));
            return (
              <tr key={id} data-case={id}>
                <td>
                  <code>{formatNumbers(x, 0)}</code>
                </td>
                <td>
                  <Value value={softmax(x)} testId="softmax-shift-value" />
                </td>
                <td data-testid="softmax-unshifted">
                  <code>{Number.isFinite(unshifted) ? unshifted.toFixed(4) : "Infinity"}</code>
                </td>
              </tr>
            );
          })}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

export function SoftmaxBackwardExample({ concept }: { concept: string }) {
  const { y, dY } = getGoldenCase("ops", "softmax_backward.test_file").inputs as {
    y: Matrix;
    dY: Matrix;
  };
  const dx = softmaxBackwardRows(y, dY);
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Softmax backward"
        kind="illustrative"
        caption="Hand-picked numbers from test_softmax_backward_rows_sum_to_zero, with two rows. Each row is computed on its own, because the sum is along the last axis. dY is chosen by hand."
      >
        <WorkedTable
          label="Softmax backward, step by step"
          testId="softmax-backward"
          rows={[
            { label: "Softmax output", code: "y", value: y },
            { label: "Gradient on the output", code: "dY", value: dY },
            {
              label: "Sum over each row",
              code: "np.sum(y * dY, axis=-1, keepdims=True)",
              value: y.map((row, t) => [row.reduce((sum, p, i) => sum + p * dY[t]![i]!, 0)]),
            },
            {
              label: "Gradient on the scores",
              code: "y * (dY - ...)",
              value: dx,
              testId: "softmax-dx",
            },
          ]}
        />
      </VizFigure>
    </div>
  );
}

// Cross-entropy -----------------------------------------------------------------------------------

export function CrossEntropyExample({ concept }: { concept: string }) {
  const { logits, targets } = getGoldenCase("ops", "cross_entropy.basic").inputs as {
    logits: Matrix;
    targets: number[];
  };
  const steps = logits.map(logSoftmax);
  const { loss, probs } = crossEntropy(logits, targets);
  const extreme = getGoldenCase("ops", "cross_entropy.extreme").inputs as {
    logits: Matrix;
    targets: number[];
  };
  const big = crossEntropy(extreme.logits, extreme.targets);
  return (
    <div className={styles.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="Two positions, three tokens"
        kind="illustrative"
        caption="Hand-picked logits and targets. They are not model output."
      >
        <WorkedTable
          label="Cross-entropy, step by step"
          testId="cross-entropy-worked"
          rows={[
            { label: "Logits", code: "logits", value: logits },
            { label: "Correct token at each position", code: "targets", value: targets },
            {
              label: "Shifted",
              code: "logits - logits.max(axis=-1, keepdims=True)",
              value: steps.map(({ z }) => z),
            },
            {
              label: "Log-probabilities",
              code: "z - np.log(np.exp(z).sum(axis=-1, keepdims=True))",
              value: steps.map(({ logp }) => logp),
            },
            {
              label: "Picked at the targets",
              code: "logp[np.arange(T), targets]",
              value: targets.map((target, t) => steps[t]!.logp[target]!),
            },
            { label: "Loss", code: "-(...).mean()", value: loss, testId: "ce-loss" },
            {
              label: "Probabilities returned",
              code: "np.exp(logp)",
              value: probs,
              testId: "ce-probs",
            },
          ]}
        />
      </VizFigure>
      <VizFigure
        title="Why the code keeps log-probabilities"
        kind="illustrative"
        caption="Hand-picked logits from test_cross_entropy_extreme_logits_no_nan. The target's probability underflows to exactly 0, and the log of 0 is minus infinity. The code never takes that log, so its loss is finite."
      >
        <WorkedTable
          label="An extreme position"
          testId="cross-entropy-extreme"
          rows={[
            { label: "Logits", code: "logits", value: extreme.logits },
            { label: "Correct token", code: "targets", value: extreme.targets },
            { label: "Probabilities", code: "np.exp(logp)", value: big.probs },
            { label: "Loss", code: "loss", value: big.loss, testId: "ce-extreme-loss" },
          ]}
        />
      </VizFigure>
    </div>
  );
}

export function CrossEntropyBackwardExample({ concept }: { concept: string }) {
  const { logits, targets } = getGoldenCase("ops", "cross_entropy.basic").inputs as {
    logits: Matrix;
    targets: number[];
  };
  const { probs } = crossEntropy(logits, targets);
  const dZ = crossEntropyBackward(probs, targets);
  const T = targets.length;
  const subtracted = probs.map((row, t) => row.map((p, v) => (v === targets[t] ? p - 1 : p)));
  const firstProb = probs[0]![targets[0]!]!;
  return (
    <div className={styles.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="A worked backward"
        kind="illustrative"
        caption="The same hand-picked logits and targets as the cross-entropy example above."
      >
        <WorkedTable
          label="Cross-entropy backward, step by step"
          testId="cross-entropy-backward"
          rows={[
            { label: "Probabilities", code: "dZ = probs.copy()", value: probs },
            {
              label: "Subtract 1 at each correct token",
              code: "dZ[np.arange(T), targets] -= 1.0",
              value: subtracted,
            },
            { label: "Divide by T", code: "return dZ / T", value: dZ, testId: "ce-dz" },
          ]}
        />
        <p data-testid="ce-entry">
          By hand, at position 0 and its correct token:{" "}
          <code>
            ({showNumber(firstProb)} - 1) / {T} = {showNumber(dZ[0]![targets[0]!]!)}
          </code>
          . In this row the entry for the correct token is negative and the others are positive:
          raising the correct logit lowers the loss.
        </p>
      </VizFigure>
      <GradientCheck
        title="Checking this backward numerically"
        testId="cross-entropy-check"
        scalar={<>the loss</>}
        rows={[
          {
            name: "dZ",
            analytic: dZ,
            numeric: numericalGrad((v) => crossEntropy(v, targets).loss, logits),
          },
        ]}
      />
    </div>
  );
}

export function SoftmaxLabExample() {
  const { logits } = getGoldenCase("ops", "cross_entropy.test_file").inputs as { logits: Matrix };
  return <SoftmaxLab logits={logits[0]!} />;
}

// Perplexity --------------------------------------------------------------------------------------

const PERPLEXITY_ROWS = [
  { id: "perplexity.basic_loss", note: "The loss of the cross-entropy example above" },
  { id: "perplexity.uniform_five", note: "ln 5, the loss of uniform probabilities over 5 tokens" },
  {
    id: "perplexity.test_file",
    note: "ln 300, the value test_perplexity_of_uniform_is_vocab_size uses",
  },
];

export function PerplexityExample({ concept }: { concept: string }) {
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Loss and perplexity"
        kind="illustrative"
        caption="Hand-picked losses, not losses from a training run."
      >
        <DataTable label="Perplexity of three losses" data-testid="perplexity-example">
          <thead>
            <tr>
              <th scope="col">Loss</th>
              <th scope="col">Where it comes from</th>
              <th scope="col">Perplexity</th>
            </tr>
          </thead>
          <tbody>
            {PERPLEXITY_ROWS.map(({ id, note }) => {
              const loss = getGoldenCase("ops", id).inputs.loss as number;
              const value = perplexity(loss);
              return (
                <tr key={id} data-case={id}>
                  <td>{loss.toFixed(4)}</td>
                  <td>{note}</td>
                  <td data-testid="perplexity-value" data-exact={value}>
                    {value.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

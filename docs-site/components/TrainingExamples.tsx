import { CODE_REFS } from "@/data/code-refs";
import { LEARNING_STAGES, LOG_LINES, LOG_ROWS, type LearningLink } from "@/data/learning-step";
import { formatLineRange, getSymbol, requireTarget, targetLines } from "@/lib/code-targets";
import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import { highlightPython } from "@/lib/highlight";
import { adamStep, createAdamState, type TensorMap } from "@/lib/math/optim";
import type { Matrix } from "@/lib/math/tensor";
import { perplexity } from "@/lib/math/ops";
import type { FlatParams } from "@/lib/math/block";
import type { ModelSizes } from "@/lib/math/model";
import { batchLossAndGrads, loopStep, validationLoss } from "@/lib/math/train";
import { pageTitle, writtenHref } from "@/lib/pages";
import { expandBlockName, loadInitParams, realShape, symbolicShape } from "@/lib/parameters";
import { resolveDimension } from "@/lib/dimensions";
import { loadConfig } from "@/lib/repo";
import { AdamCalculator } from "./AdamCalculator";
import { ClipLab } from "./ClipLab";
import { DataTable } from "./DataTable";
import examples from "./Examples.module.css";
import { InlineCode } from "./InlineCode";
import { ScheduleLab } from "./ScheduleLab";
import { StepCalculator } from "./StepCalculator";
import { fine } from "./WorkedRows";
import type { FlowCode } from "./GradientFlow";
import {
  LearningPipeline,
  type ResolvedLearningLink,
  type ResolvedLearningStage,
} from "./LearningPipeline";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases. Results come from lib/math, which tests/math.golden.test.ts
// compares with the real functions at 1e-9. Printed text is the stored result of evaluating train.main's
// own format strings, because Python's formatting has no mirror.

function Exact({ value, testId }: { value: number; testId?: string }) {
  return (
    <code data-testid={testId} data-exact={JSON.stringify(value)}>
      {fine(value)}
    </code>
  );
}

// One real step ------------------------------------------------------------------------------------

export type LearningOutputs = {
  tokens: string[];
  ids: number[];
  vocab_size: number;
  train_windows: number;
  val_windows: number;
  xb: number[][];
  yb: number[][];
  logits: Matrix;
  probs: Matrix;
  loss: number;
  grads: TensorMap;
  norm: number;
  clipped: TensorMap;
  total_steps: number;
  lr: number;
  params: TensorMap;
  val_loss: number;
  seed_ids: number[];
  printed: string[];
};

const learningCase = () => getGoldenCase("train", "learning.one_step");
const learning = () => learningCase().outputs as LearningOutputs;

const clipped = (text: string, length = 44) =>
  text.length > length ? `${text.slice(0, length)}...` : text;

const list = (values: readonly (number | string)[], count = 8) =>
  `[${values.slice(0, count).join(", ")}${values.length > count ? ", ..." : ""}]`;

const headEntry = (tensors: TensorMap) => (tensors.head_b as number[])[0]!;

// The value each detail panel shows, all from one real step.
function stageValue(id: string): ResolvedLearningStage["value"] {
  const o = learning();
  const inputs = learningCase().inputs as {
    text: string;
    params: TensorMap;
    batch_size: number;
  };
  const showRow = (row: number[]) => list(row.map((value) => Number(value.toFixed(4))));
  const value = (label: string, text: string, exact?: number | string) => ({
    label,
    text,
    ...(exact === undefined ? {} : { exact: JSON.stringify(exact) }),
  });
  switch (id) {
    case "text":
      return value("The text", JSON.stringify(clipped(inputs.text)));
    case "tokens":
      return value("First ids", list(o.ids));
    case "windows":
      return value(
        "First batch, inputs and targets of one window",
        `${list(o.xb[0]!)} -> ${list(o.yb[0]!)}`,
      );
    case "forward":
      return value("Logits of the first window, position 0", showRow(o.logits[0]!));
    case "probabilities":
      return value("probs of the first window, position 0", showRow(o.probs[0]!));
    case "loss":
      return value("Loss of the batch", fine(o.loss), o.loss);
    case "backward":
      return value(
        "Each window fills one gradient per parameter. The batch has",
        `${o.xb.length} windows`,
      );
    case "gradients":
      return value("Mean gradient, head_b[0]", fine(headEntry(o.grads)), headEntry(o.grads));
    case "clipping":
      return value(
        "The norm is above the limit, so head_b[0] after clipping is",
        `${fine(headEntry(o.clipped))} (|g| = ${fine(o.norm)})`,
        headEntry(o.clipped),
      );
    case "schedule":
      return value("lr at step 1", fine(o.lr), o.lr);
    case "adamw":
      return value(
        "head_b[0] before the update",
        fine(headEntry(inputs.params)),
        headEntry(inputs.params),
      );
    case "updated":
      return value("head_b[0] after the update", fine(headEntry(o.params)), headEntry(o.params));
    case "repeat":
      return value(
        "This text gives",
        `${o.train_windows} training windows, ${o.total_steps} steps at batch_size ${inputs.batch_size}`,
      );
    default:
      throw new Error(`No value for stage ${id}`);
  }
}

function resolveLink(link: LearningLink): ResolvedLearningLink {
  const target = requireTarget(link.target);
  return {
    statement: targetLines(target).trim(),
    location: `${target.symbol.file}:${formatLineRange(target.startLine, target.endLine)}`,
    href: target.permalink,
  };
}

async function frame(id: string): Promise<FlowCode> {
  const symbol = getSymbol(id);
  return {
    lines: await highlightPython(symbol.source),
    source: symbol.source,
    filePath: symbol.file,
    functionName: symbol.name,
    startLine: symbol.startLine,
    permalink: symbol.permalink,
  };
}

// Server wrapper: resolves lines and highlights on the server, so Shiki stays out of the browser.
export async function LearningPipelineDiagram() {
  const stages: ResolvedLearningStage[] = LEARNING_STAGES.map((stage) => {
    const own = requireTarget(stage.links.at(-1)!.target);
    const lines =
      own.symbol.id === stage.showSymbol
        ? Array.from({ length: own.endLine - own.startLine + 1 }, (_, i) => own.startLine + i)
        : [];
    return {
      id: stage.id,
      label: stage.label,
      summary: stage.summary,
      page: { title: pageTitle(stage.page), href: writtenHref(stage.page) },
      links: stage.links.map(resolveLink),
      also: stage.alsoCalls
        ? { ...resolveLink({ target: stage.alsoCalls.target }), note: stage.alsoCalls.note }
        : null,
      symbol: stage.showSymbol,
      lines,
      value: stageValue(stage.id),
    };
  });
  const ids = [...new Set(stages.map((stage) => stage.symbol))];
  const code = Object.fromEntries(await Promise.all(ids.map(async (id) => [id, await frame(id)])));
  return <LearningPipeline stages={stages} code={code} />;
}

// One weight --------------------------------------------------------------------------------------

type OneWeightInputs = {
  params: TensorMap;
  grads: TensorMap[];
  settings: { beta1: number; beta2: number; eps: number; weight_decay: number };
  max_norm: number;
  warmup: number;
  total: number;
  lr_max: number;
  lr_min: number;
};

export function runOneWeight() {
  const inputs = getGoldenCase("optim", "adam.one_weight").inputs as OneWeightInputs;
  const { settings } = inputs;
  let state = createAdamState(inputs.params, {
    beta1: settings.beta1,
    beta2: settings.beta2,
    eps: settings.eps,
    weightDecay: settings.weight_decay,
  });
  let params = inputs.params;
  let step = 0;
  const rows = inputs.grads.map((grads) => {
    const result = loopStep(state, params, grads, step, {
      maxNorm: inputs.max_norm,
      warmup: inputs.warmup,
      total: inputs.total,
      lrMax: inputs.lr_max,
      lrMin: inputs.lr_min,
    });
    const before = (params.w as Matrix)[0]![0]!;
    state = result.state;
    params = result.params;
    step = result.step;
    const entry = (tensor: TensorMap[string]) => (tensor as Matrix)[0]![0]!;
    const trace = result.trace.w!;
    return {
      before,
      g: entry(grads.w!),
      norm: result.norm,
      clipped: entry(result.clipped.w!),
      step: result.step,
      lr: result.lr,
      t: state.t,
      m: entry(state.m.w!),
      v: entry(state.v.w!),
      mhat: entry(trace.mhat),
      vhat: entry(trace.vhat),
      raw: entry(trace.raw),
      update: entry(trace.update),
      after: entry(result.params.w!),
    };
  });
  return { inputs, rows };
}

export function WeightWalkthroughExample() {
  const { inputs, rows } = runOneWeight();
  const stepRows: { label: string; code: string; key: keyof (typeof rows)[number] }[] = [
    { label: "Weight before", code: "params[k]", key: "before" },
    { label: "Mean gradient", code: "grads[k]", key: "g" },
    { label: "Length before clipping", code: "norm", key: "norm" },
    { label: "Gradient after clipping", code: "grads[k]", key: "clipped" },
    { label: "Step", code: "step += 1", key: "step" },
    { label: "Learning rate", code: "lr", key: "lr" },
    { label: "Moving average", code: "self.m[k]", key: "m" },
    { label: "Moving average of the square", code: "self.v[k]", key: "v" },
    { label: "Bias-corrected", code: "mhat", key: "mhat" },
    { label: "Bias-corrected", code: "vhat", key: "vhat" },
    { label: "Update before decay", code: "mhat / (np.sqrt(vhat) + self.eps)", key: "raw" },
    { label: "Update with decay", code: "update + self.wd * params[k]", key: "update" },
    { label: "Weight after", code: "params[k] -= lr * update", key: "after" },
  ];
  return (
    <VizFigure
      title="One weight over two steps"
      kind="illustrative"
      caption={
        <>
          The weight and its gradients are hand-picked, and <code>total_steps</code> is{" "}
          {inputs.total}, because the real value depends on the corpus. The settings are the values
          in <code>config.py</code>. Both steps fall inside warmup, so the weight moves very little.
          The first gradient is above <code>grad_clip</code>, and the second is not.
        </>
      }
    >
      <DataTable label="One weight over two steps" data-testid="weight-steps">
        <thead>
          <tr>
            <th scope="col">Value</th>
            <th scope="col">Code</th>
            <th scope="col">Step 1</th>
            <th scope="col">Step 2</th>
          </tr>
        </thead>
        <tbody>
          {stepRows.map((row) => (
            <tr key={row.label + row.code}>
              <td>{row.label}</td>
              <td>
                <code>{row.code}</code>
              </td>
              {rows.map((values, index) => (
                <td key={index}>
                  <Exact value={values[row.key]} testId={`weight-${row.key}-${index}`} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

// What each number means ---------------------------------------------------------------------------

export function LogMeanings() {
  const { printed } = learning();
  const lines = [printed[1]!, printed[2]!];
  return (
    <div className={examples.stack}>
      <VizFigure
        title="The two lines these numbers appear on"
        kind="illustrative"
        caption="Printed by evaluating the real format strings of train.main with the values of one real step on a hand-picked text and a tiny model. train.main itself did not run, and these are not numbers from a training run."
      >
        <pre className={examples.log} data-testid="log-sample" tabIndex={0}>
          {lines.join("\n")}
        </pre>
      </VizFigure>
      <DataTable label="What each printed number means" data-testid="log-meanings">
        <thead>
          <tr>
            <th scope="col">Printed as</th>
            <th scope="col">In train.main</th>
            <th scope="col">Meaning</th>
          </tr>
        </thead>
        <tbody>
          {Object.values(LOG_ROWS).map((row) => (
            <tr key={row.printed}>
              <td>
                <code>{row.printed}</code>
              </td>
              <td>
                <code>{row.code}</code>
              </td>
              <td>
                <InlineCode text={row.meaning} />
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </div>
  );
}

// AdamW ---------------------------------------------------------------------------------------------

type AdamInputs = {
  params: TensorMap;
  grads: TensorMap[];
  lrs: number[];
  settings: { beta1: number; beta2: number; eps: number; weight_decay: number };
};

const engineSettings = ({ settings }: Pick<AdamInputs, "settings">) => ({
  beta1: settings.beta1,
  beta2: settings.beta2,
  eps: settings.eps,
  weightDecay: settings.weight_decay,
});

export function runAdamTwoStep() {
  const inputs = getGoldenCase("optim", "adam.two_step").inputs as AdamInputs;
  let state = createAdamState(inputs.params, engineSettings(inputs));
  let params = inputs.params;
  const steps = inputs.grads.map((grads, index) => {
    const lr = inputs.lrs[index]!;
    const next = adamStep(state, params, grads, lr);
    const before = params;
    state = next.state;
    params = next.params;
    return { before, grads, lr, ...next };
  });
  return { inputs, steps };
}

export function AdamTwoStepExample({ concept }: { concept: string }) {
  const { inputs, steps } = runAdamTwoStep();
  const { beta1, beta2, eps, weight_decay } = inputs.settings;
  const entry = (tensor: TensorMap[string]) => tensor.flat()[0] as number;
  type Step = (typeof steps)[number];
  const ROWS: { label: string; code: string; get: (step: Step, key: string) => number }[] = [
    { label: "Before", code: "params[k]", get: (step, key) => entry(step.before[key]!) },
    { label: "Gradient", code: "g = grads[k]", get: (step, key) => entry(step.grads[key]!) },
    { label: "Learning rate", code: "lr", get: (step) => step.lr },
    { label: "Counter", code: "self.t", get: (step) => step.state.t },
    { label: "Average", code: "self.m[k]", get: (step, key) => entry(step.state.m[key]!) },
    {
      label: "Average of squares",
      code: "self.v[k]",
      get: (step, key) => entry(step.state.v[key]!),
    },
    { label: "Corrected", code: "mhat", get: (step, key) => entry(step.trace[key]!.mhat) },
    { label: "Corrected", code: "vhat", get: (step, key) => entry(step.trace[key]!.vhat) },
    {
      label: "Before decay",
      code: "mhat / (np.sqrt(vhat) + self.eps)",
      get: (step, key) => entry(step.trace[key]!.raw),
    },
    { label: "Update", code: "update", get: (step, key) => entry(step.trace[key]!.update) },
    {
      label: "After",
      code: "params[k] -= lr * update",
      get: (step, key) => entry(step.params[key]!),
    },
  ];
  const names = ["w", "b"] as const;
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="Two steps, with and without decay"
        kind="illustrative"
        caption={
          <>
            The numbers of <code>test_adam_two_step_hand_trace</code>: start at 1, gradients{" "}
            {entry(inputs.grads[0]!.w!)} then {entry(inputs.grads[1]!.w!)}, and <code>lr</code>{" "}
            {inputs.lrs[0]}, which is far above <code>lr_max</code> so that the moves are easy to
            see. Other settings are <code>beta1</code> = {beta1}, <code>beta2</code> = {beta2},{" "}
            <code>adam_eps</code> = {eps} and <code>weight_decay</code> = {weight_decay}, the values
            in <code>config.py</code>. <code>w</code> has two dimensions and <code>b</code> has one.
          </>
        }
      >
        <DataTable label="Two Adam steps" data-testid="adam-two-step">
          <thead>
            <tr>
              <th scope="col">Value</th>
              <th scope="col">Code</th>
              {steps.flatMap((_, index) =>
                names.map((name) => (
                  <th scope="col" key={`${index}${name}`}>
                    Step {index + 1}, <code>{name}</code>
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.label + row.code}>
                <td>{row.label}</td>
                <td>
                  <code>{row.code}</code>
                </td>
                {steps.flatMap((step, index) =>
                  names.map((name) => (
                    <td key={`${index}${name}`}>
                      <Exact
                        value={row.get(step, name)}
                        testId={`two-step-${row.code === "update" ? "update" : row.code === "params[k] -= lr * update" ? "after" : row.code}-${index}-${name}`}
                      />
                    </td>
                  )),
                )}
              </tr>
            ))}
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

export function AdamCalculatorExample() {
  const { inputs } = runAdamTwoStep();
  return (
    <AdamCalculator
      start={{
        w: inputs.params.w!.flat()[0] as number,
        g: inputs.grads[0]!.w!.flat()[0] as number,
        m: 0,
        v: 0,
        t: 0,
        lr: inputs.lrs[0]!,
      }}
      settings={engineSettings(inputs)}
    />
  );
}

// Decay ---------------------------------------------------------------------------------------------

export type DecayRow = {
  name: string;
  ndim: number;
  symbolic: string;
  real: string;
  shrunk: boolean;
};

// Names and dimensions come from calling the real init_params, and whether a tensor shrinks comes from
// a real Adam.step on all ones with zero gradients.
export function decayRows(): DecayRow[] {
  const { parameters } = loadInitParams();
  const { rows } = getGoldenCase("optim", "adam.decay_rule").outputs as {
    rows: { name: string; ndim: number; changed: boolean }[];
  };
  const values = {
    Tmax: resolveDimension("Tmax").value!,
    d: resolveDimension("d").value!,
    F: resolveDimension("F").value!,
  };
  return parameters.map(({ name, dims }) => {
    const concrete = name.replace("{l}", "0");
    const row = rows.find((candidate) => candidate.name === concrete);
    if (!row) throw new Error(`The decay rule case has no ${concrete}`);
    return {
      name,
      ndim: row.ndim,
      symbolic: symbolicShape(dims),
      real: realShape(dims, values),
      shrunk: row.changed,
    };
  });
}

export function decayCounts() {
  const all = expandBlockName(loadInitParams().parameters, resolveDimension("L").value!);
  const decayed = all.filter(({ dims }) => dims.length >= 2);
  return { decayed: decayed.length, kept: all.length - decayed.length, total: all.length };
}

export function DecayTable() {
  const rows = decayRows();
  const { decayed, kept } = decayCounts();
  return (
    <div className={examples.stack}>
      <DataTable label="Which parameters get weight decay" data-testid="decay-table">
        <thead>
          <tr>
            <th scope="col">Key</th>
            <th scope="col">Shape</th>
            <th scope="col">With config.py values</th>
            <th scope="col">Dimensions</th>
            <th scope="col">Weight decay</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name} data-decay={row.shrunk} data-key={row.name}>
              <td>
                <code>{row.name}</code>
              </td>
              <td>
                <code>{row.symbolic}</code>
              </td>
              <td>
                <code>{row.real}</code>
              </td>
              <td>{row.ndim}</td>
              <td>{row.shrunk ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      <p data-testid="decay-counts">
        With <code>num_layers</code> = {resolveDimension("L").value}, the <code>b{"{l}"}.</code>{" "}
        keys appear once per block, so {decayed} tensors get decay and {kept} do not.
      </p>
    </div>
  );
}

// Schedule ------------------------------------------------------------------------------------------

const configNumber = (name: string) => {
  const value = loadConfig().constants.find((constant) => constant.name === name)?.value;
  if (typeof value !== "number") throw new Error(`config.json has no numeric ${name}`);
  return value;
};

export function ScheduleLabExample({ concept }: { concept: string }) {
  const { inputs } = getGoldenCase("optim", "lr_schedule.key_points") as unknown as {
    inputs: { total: number };
  };
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      <ScheduleLab
        start={{
          total: inputs.total,
          warmup: configNumber("warmup_steps"),
          lrMax: configNumber("lr_max"),
          lrMin: configNumber("lr_min"),
        }}
      />
    </div>
  );
}

// Clipping ------------------------------------------------------------------------------------------

export function ClipLabExample({ concept }: { concept: string }) {
  const grads = getGoldenCase("optim", "clip_grad_norm.above").inputs.grads as {
    a: number[];
    b: number[][];
  };
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      <ClipLab a={grads.a} b={grads.b.flat()} maxNorm={configNumber("grad_clip")} />
    </div>
  );
}

// Mini-batches --------------------------------------------------------------------------------------

type BatchInputs = { params: FlatParams; xb: number[][]; yb: number[][]; cfg: ModelSizes };

export function runBatchAverage() {
  const inputs = getGoldenCase("train", "batch.three_windows").inputs as BatchInputs;
  const { loss, grads, losses } = batchLossAndGrads(
    inputs.params,
    inputs.xb,
    inputs.yb,
    inputs.cfg,
  );
  const B = inputs.xb.length;
  let running = 0;
  const rows = losses.map((value, index) => {
    running += value / B;
    return {
      index,
      x: inputs.xb[index]!,
      y: inputs.yb[index]!,
      loss: value,
      part: value / B,
      running,
    };
  });
  return { B, loss, grads, rows };
}

export function BatchAverageExample({ concept }: { concept: string }) {
  const { B, loss, rows } = runBatchAverage();
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title={`A batch of ${B} windows`}
        kind="illustrative"
        caption="A tiny model: the parameters are random draws from the real init_params, not learned values, and the ids are hand-picked. Each row is one pass of the loop."
      >
        <DataTable label="The loss through the loop" data-testid="batch-rows">
          <thead>
            <tr>
              <th scope="col">Window</th>
              <th scope="col">
                <code>x</code>
              </th>
              <th scope="col">
                <code>y</code>
              </th>
              <th scope="col">
                <code>loss</code>
              </th>
              <th scope="col">
                <code>loss / B</code>
              </th>
              <th scope="col">
                <code>total_loss</code>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.index}>
                <td>{row.index}</td>
                <td>
                  <code>{list(row.x)}</code>
                </td>
                <td>
                  <code>{list(row.y)}</code>
                </td>
                <td>
                  <Exact value={row.loss} testId={`batch-loss-${row.index}`} />
                </td>
                <td>
                  <Exact value={row.part} testId={`batch-part-${row.index}`} />
                </td>
                <td>
                  <Exact value={row.running} testId={`batch-running-${row.index}`} />
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
        <p>
          The function returns <Exact value={loss} testId="batch-total" />.
        </p>
      </VizFigure>
    </div>
  );
}

// Step counts ---------------------------------------------------------------------------------------

export function StepCountExample({ concept }: { concept: string }) {
  const { rows } = getGoldenCase("train", "steps.counts").outputs as {
    rows: { windows: number }[];
  };
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      <StepCalculator
        windows={rows[0]!.windows}
        batchSize={configNumber("batch_size")}
        epochs={configNumber("epochs")}
      />
    </div>
  );
}

// Validation ----------------------------------------------------------------------------------------

type ValidationInputs = {
  params: FlatParams;
  inputs: number[][];
  targets: number[][];
  cfg: ModelSizes;
};

export function runValidation() {
  const inputs = getGoldenCase("train", "validation.tiny").inputs as ValidationInputs;
  const { loss, losses } = validationLoss(inputs.params, inputs.cfg, inputs.inputs, inputs.targets);
  return { loss, losses, ppl: perplexity(loss) };
}

export function ValidationExample({ concept }: { concept: string }) {
  const { loss, losses, ppl } = runValidation();
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title={`${losses.length} validation windows`}
        kind="illustrative"
        caption="The same tiny model and windows as the batch example, so the three losses are the same. Validation takes a mean of all windows, and those parameters are not learned values."
      >
        <DataTable label="Validation" data-testid="validation-rows">
          <thead>
            <tr>
              <th scope="col">Value</th>
              <th scope="col">Code</th>
              <th scope="col">Result</th>
            </tr>
          </thead>
          <tbody>
            {losses.map((value, index) => (
              <tr key={index}>
                <td>Window {index}</td>
                <td>
                  <code>loss</code>
                </td>
                <td>
                  <Exact value={value} testId={`validation-loss-${index}`} />
                </td>
              </tr>
            ))}
            <tr>
              <td>Mean</td>
              <td>
                <code>float(np.mean(losses))</code>
              </td>
              <td>
                <Exact value={loss} testId="validation-mean" />
              </td>
            </tr>
            <tr>
              <td>Perplexity</td>
              <td>
                <code>perplexity(v)</code>
              </td>
              <td>
                <Exact value={ppl} testId="validation-ppl" />
              </td>
            </tr>
          </tbody>
        </DataTable>
      </VizFigure>
    </div>
  );
}

// Logging lines -------------------------------------------------------------------------------------

export function LogLines() {
  const { printed } = learning();
  return (
    <VizFigure title="Every line train.py prints" kind="illustrative">
      <ol className={examples.logList} data-testid="log-lines">
        {LOG_LINES.map((line, index) => {
          const target = requireTarget(CODE_REFS[line.ref]!);
          return (
            <li key={line.ref} data-testid={`log-line-${index}`}>
              <p>
                <strong>{line.when}</strong>{" "}
                <a className="code-link" href={target.permalink}>
                  {target.symbol.file}:{formatLineRange(target.startLine, target.endLine)}
                </a>
              </p>
              <pre className={examples.log} data-testid={`log-output-${index}`} tabIndex={0}>
                {printed[index]!}
              </pre>
              <p>
                <InlineCode text={line.numbers} />
              </p>
            </li>
          );
        })}
      </ol>
    </VizFigure>
  );
}

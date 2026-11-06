// @vitest-environment happy-dom

// Every interactive component, mounted as its page mounts it and driven by real events in a simulated
// DOM. What each one shows is compared with data/golden, the results of running the repository's real
// Python functions, not with lib/math. So a wrong formula in the engine or in a component fails here.
import { act, createElement, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, test } from "vitest";
import { AttentionLabExample } from "@/components/AttentionExamples";
import { BpeExample } from "@/components/BpeExample";
import { ResidualStreamExample } from "@/components/BlockExamples";
import { ClipLab } from "@/components/ClipLab";
import { WindowExample } from "@/components/DataExamples";
import { EmbeddingLab } from "@/components/EmbeddingLab";
import { GenerationStepperExample, labPresets } from "@/components/GenerationExamples";
import { CentralDifferenceExample } from "@/components/GradcheckExamples";
import { LayerNormLabExample } from "@/components/LayerNormExamples";
import { MatrixExample } from "@/components/MatrixExample";
import { MatrixInput } from "@/components/MatrixInput";
import { LinearPlaygroundExample, SoftmaxLabExample } from "@/components/OpsExamples";
import { ProbabilityLab } from "@/components/ProbabilityLab";
import { ScheduleLab } from "@/components/ScheduleLab";
import { StepPlayer } from "@/components/StepPlayer";
import { showToken } from "@/components/Tokens";
import {
  AdamCalculatorExample,
  ClipLabExample,
  ScheduleLabExample,
  StepCountExample,
} from "@/components/TrainingExamples";
import { getGoldenCase } from "@/lib/golden";
import { STREAM_STEPS } from "@/lib/residual-stream";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The attention lab's code frame is highlighted by Shiki, which loads WebAssembly with this method.
// happy-dom's Response does not have it.
const responseProto = Response.prototype as Response & { bytes?: () => Promise<Uint8Array> };
responseProto.bytes ??= async function (this: Response) {
  return new Uint8Array(await this.arrayBuffer());
};

let root: Root | null = null;
let page: HTMLElement;

async function mount(element: ReactElement) {
  page = document.createElement("div");
  document.body.append(page);
  root = createRoot(page);
  await act(async () => root!.render(element));
}

afterEach(async () => {
  if (root) await act(async () => root!.unmount());
  root = null;
  document.body.innerHTML = "";
});

const byTestId = (id: string) => {
  const element = page.querySelector<HTMLElement>(`[data-testid="${id}"]`);
  if (!element) throw new Error(`Nothing has data-testid="${id}"`);
  return element;
};

// JSON has no infinity, so a masked score is null in data-exact.
const exact = (id: string) => JSON.parse(byTestId(id).dataset.exact!);

const click = (element: HTMLElement) => act(async () => element.click());

// React tracks an input's value, so the change has to go through the native setter.
async function type(element: HTMLElement, value: string | number) {
  const prototype =
    element instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")!.set!;
  await act(async () => {
    setter.call(element, String(value));
    element.dispatchEvent(
      new Event(element instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }),
    );
  });
}

const stepButton = (title: string) => {
  const button = [...page.querySelectorAll<HTMLElement>("button")].find((candidate) =>
    candidate.textContent?.includes(title),
  );
  if (!button) throw new Error(`No button mentions "${title}"`);
  return button;
};

const cell = (label: string) => {
  const input = page.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`);
  if (!input) throw new Error(`No cell is labelled "${label}"`);
  return input;
};

// Compares numbers to 1e-9, and anything else exactly.
function expectClose(actual: unknown, expected: unknown, path = "value") {
  if (typeof expected === "number" && Number.isFinite(expected)) {
    expect(typeof actual, path).toBe("number");
    expect(Math.abs((actual as number) - expected), path).toBeLessThanOrEqual(1e-9);
    return;
  }
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual), path).toBe(true);
    expect((actual as unknown[]).length, path).toBe(expected.length);
    expected.forEach((item, i) => expectClose((actual as unknown[])[i], item, `${path}[${i}]`));
    return;
  }
  expect(actual, path).toEqual(expected);
}

const matrixToJson = (matrix: number[][]) =>
  matrix.map((row) => row.map((value) => (value === -Infinity ? null : value)));

describe("AttentionLab, against attention_forward.cat_sat", () => {
  const { outputs } = getGoldenCase("attention", "attention_forward.cat_sat");
  const A = outputs.A as number[][][];
  const scaled = outputs.scaled as number[][][];
  const masked = outputs.masked as number[][][];

  async function open() {
    await mount(await AttentionLabExample());
  }

  test("each row of A sums to 1 and masked cells are exactly 0, in both heads", async () => {
    await open();
    await click(stepButton("Softmax"));
    for (const head of [0, 1]) {
      await click(byTestId(`lab-head-${head}`));
      const shown: number[][] = exact("lab-matrix");
      expectClose(shown, A[head]);
      shown.forEach((row, t) => {
        expect(Math.abs(row.reduce((total, value) => total + value, 0) - 1)).toBeLessThan(1e-12);
        row.forEach((value, c) => {
          if (c > t) expect(value).toBe(0);
          else expect(value).toBeGreaterThan(0);
        });
      });
    }
  });

  test("the scale and mask steps show the real matrices", async () => {
    await open();
    await click(stepButton("Scale"));
    expectClose(exact("lab-matrix"), scaled[0]);
    await click(stepButton("Mask"));
    expectClose(exact("lab-matrix"), matrixToJson(masked[0]!));
    expect(exact("lab-matrix")[0][1]).toBeNull();
  });

  test("the selected token and head choose the weights shown", async () => {
    await open();
    await click(stepButton("Softmax"));
    expectClose(exact("lab-weights"), A[0]![2]);
    await click(byTestId("lab-token-4"));
    expectClose(exact("lab-weights"), A[0]![4]);
    await click(byTestId("lab-head-1"));
    expectClose(exact("lab-weights"), A[1]![4]);
  });

  test("the output step shows the real O and out", async () => {
    await open();
    await click(stepButton("Output"));
    expectClose(exact("lab-o"), outputs.O);
    expectClose(exact("lab-out"), outputs.out);
  });
});

describe("SoftmaxLab, against cross_entropy.test_file", () => {
  const { outputs } = getGoldenCase("ops", "cross_entropy.test_file");
  const probs = (outputs.probs as number[][])[0]!;

  test("reproduces the real distribution and loss", async () => {
    await mount(createElement(SoftmaxLabExample));
    expectClose(exact("lab-probs"), [probs]);
    expectClose(exact("lab-loss"), outputs.loss);
  });

  test("the gradient is P minus the one-hot row of the chosen token", async () => {
    await mount(createElement(SoftmaxLabExample));
    const { outputs: backward } = getGoldenCase("ops", "cross_entropy.test_file.backward");
    expectClose(exact("lab-gradient"), backward.dZ);

    await type(byTestId("lab-target"), 2);
    expectClose(exact("lab-loss"), -Math.log(probs[2]!));
    expectClose(exact("lab-gradient"), [[probs[0], probs[1], probs[2]! - 1]]);
  });

  test("editing a logit changes the distribution as softmax says", async () => {
    await mount(createElement(SoftmaxLabExample));
    await type(cell("logits, row t0, column token 0"), 0);
    // All three logits now give [0, 1, 0], worked by hand.
    const e = [1, Math.E, 1];
    const total = e[0]! + e[1]! + e[2]!;
    expectClose(exact("lab-probs"), [e.map((value) => value / total)]);
  });
});

describe("MatrixPlayground", () => {
  test("reproduces matmul.three_columns, with and without the bias", async () => {
    await mount(createElement(MatrixExample, { concept: "notation.matmul" }));
    await type(byTestId("matmul-columns"), 3);
    const noBias = getGoldenCase("ops", "matmul.three_columns.no_bias").outputs.Y;
    const withBias = getGoldenCase("ops", "matmul.three_columns").outputs.Y;
    expectClose(exact("matmul-product"), noBias);
    expectClose(exact("matmul-output"), withBias);
  });

  test("the linear page reproduces forward and backward", async () => {
    await mount(createElement(LinearPlaygroundExample, { concept: "ops.linear_forward" }));
    const backward = getGoldenCase("ops", "linear.test_file.backward").outputs;
    expectClose(exact("matmul-output"), getGoldenCase("ops", "linear.test_file").outputs.Y);
    expectClose(exact("linear-dw"), backward.dW);
    expectClose(exact("linear-db"), [backward.db]);
    expectClose(exact("linear-dx"), backward.dX);
  });

  test("editing X gives the real result for the edited input", async () => {
    await mount(createElement(LinearPlaygroundExample, { concept: "ops.linear_forward" }));
    const edited = getGoldenCase("ops", "linear.edited");
    await type(cell("X, row t0, column k0"), (edited.inputs.X as number[][])[0]![0]!);
    expectClose(exact("matmul-output"), edited.outputs.Y);
    expectClose(exact("matmul-product"), getGoldenCase("ops", "linear.edited.no_bias").outputs.Y);
    const backward = getGoldenCase("ops", "linear.edited.backward");
    for (const [t, row] of (backward.inputs.dY as number[][]).entries()) {
      for (const [j, value] of row.entries())
        await type(cell(`dY, row t${t}, column j${j}`), value);
    }
    expectClose(exact("linear-dw"), backward.outputs.dW);
    expectClose(exact("linear-db"), [backward.outputs.db]);
    expectClose(exact("linear-dx"), backward.outputs.dX);
  });

  test("choosing two columns gives the real two-column result", async () => {
    await mount(createElement(LinearPlaygroundExample, { concept: "ops.linear_forward" }));
    await type(byTestId("matmul-columns"), 2);
    const two = getGoldenCase("ops", "linear.two_columns").outputs.Y;
    expectClose(exact("matmul-output"), two);
    expect(byTestId("matmul-rule").textContent).toContain("(2, 2) @ (2, 2) → (2, 2)");
  });
});

describe("LayerNormLab, against layer_norm", () => {
  test("reproduces the statistics and output of test_file", async () => {
    await mount(createElement(LayerNormLabExample, { concept: "ops.layer_norm_forward" }));
    const { outputs } = getGoldenCase("ops", "layer_norm.test_file");
    expectClose(exact("ln-mu"), (outputs.mu as number[][])[0]![0]);
    expectClose(exact("ln-var"), (outputs.var as number[][])[0]![0]);
    expectClose(exact("ln-s"), (outputs.s as number[][])[0]![0]);
    expectClose(exact("ln-xhat"), outputs.Xhat);
    expectClose(exact("ln-output"), outputs.Y);
  });

  test("a constant row gives beta, and says why", async () => {
    await mount(createElement(LayerNormLabExample, { concept: "ops.layer_norm_forward" }));
    const { inputs, outputs } = getGoldenCase("ops", "layer_norm.constant_row");
    const row = (inputs.X as number[][])[0]!;
    for (const [i, value] of row.entries()) await type(cell(`X, row t0, column i${i}`), value);
    expectClose(exact("ln-output"), outputs.Y);
    expect(byTestId("ln-status").textContent).toContain("its variance is 0");
  });
});

describe("CentralDifferenceLab, against numerical_grad", () => {
  test("x cubed matches at the default h and at a large h", async () => {
    await mount(createElement(CentralDifferenceExample, { concept: "gradcheck.numerical_grad" }));
    const atDefault = getGoldenCase("gradcheck", "numerical_grad.cube_default_h").outputs;
    expectClose(exact("cd-numeric-3"), (atDefault.g as number[])[0]);
    expectClose(exact("cd-error-3"), atDefault.error);

    const large = getGoldenCase("gradcheck", "numerical_grad.cube_large_h");
    const h = [...page.querySelectorAll<HTMLInputElement>('input[type="number"]')].at(-1)!;
    await type(h, large.inputs.h as number);
    expectClose(exact("cd-numeric-3"), (large.outputs.g as number[])[0]);
    expectClose(exact("cd-error-3"), large.outputs.error);
  });

  test("h of 0 is refused, because the code divides by it", async () => {
    await mount(createElement(CentralDifferenceExample, { concept: "gradcheck.numerical_grad" }));
    const h = [...page.querySelectorAll<HTMLInputElement>('input[type="number"]')].at(-1)!;
    await type(h, 0);
    expect(byTestId("cd-status").textContent).toContain("h must be above 0");
    expect(exact("cd-numeric-3")).toBeNull();
  });
});

describe("EmbeddingLab, against embed_forward and embed_backward", () => {
  type Forward = { tok_emb: number[][]; pos_emb: number[][]; ids: number[] };
  const forward = (id: string) => getGoldenCase("embeddings", id);
  const backward = (id: string) => getGoldenCase("embeddings", id);

  async function mountFor(forwardId: string, backwardId: string) {
    const { tok_emb, pos_emb, ids } = forward(forwardId).inputs as Forward;
    const { dX0 } = backward(backwardId).inputs as { dX0: number[][] };
    await mount(createElement(EmbeddingLab, { tokEmb: tok_emb, posEmb: pos_emb, ids, dX0 }));
  }

  test("repeated ids: X0, and gradients that add up", async () => {
    await mountFor("embed_forward.test_file", "embed_backward.test_file");
    const out = forward("embed_forward.test_file").outputs;
    expectClose(exact("lab-tokens"), out.tokens);
    expectClose(exact("lab-positions"), out.positions);
    expectClose(exact("lab-x0"), out.X0);
    const grads = backward("embed_backward.test_file").outputs;
    expectClose(exact("lab-dtok"), grads.tok);
    expectClose(exact("lab-dpos"), grads.pos);
    expectClose(exact("lab-overwritten"), grads.overwritten);
  });

  test("choosing other ids gives the real result for those ids", async () => {
    await mountFor("embed_forward.test_file", "embed_backward.test_file");
    const { ids } = forward("embed_forward.no_repeat").inputs as Forward;
    for (const [t, id] of ids.entries()) await type(byTestId(`lab-id-${t}`), id);
    expectClose(exact("lab-x0"), forward("embed_forward.no_repeat").outputs.X0);
    expectClose(exact("lab-dtok"), backward("embed_backward.no_repeat").outputs.tok);
    expect(byTestId("lab-repeat").textContent).toContain("No id repeats");
  });
});

describe("ScheduleLab, against lr_schedule.key_points", () => {
  const { inputs, outputs } = getGoldenCase("optim", "lr_schedule.key_points");
  const steps = inputs.steps as number[];
  const lrs = outputs.lrs as number[];

  test("every key step shows the real rate", async () => {
    await mount(
      createElement(ScheduleLab, {
        start: {
          total: inputs.total as number,
          warmup: inputs.warmup as number,
          lrMax: inputs.lr_max as number,
          lrMin: inputs.lr_min as number,
        },
      }),
    );
    const shown = [...page.querySelectorAll<HTMLElement>('[data-testid^="schedule-step-"]')];
    expect(shown.length).toBeGreaterThanOrEqual(3);
    for (const stepCell of shown) {
      const step = Number(stepCell.textContent);
      const index = steps.indexOf(step);
      expect(index, `step ${step} has a golden value`).toBeGreaterThanOrEqual(0);
      const id = stepCell.dataset.testid!.replace("schedule-step-", "");
      expectClose(exact(`schedule-lr-${id}`), lrs[index]);
    }
  });

  test("the page mounts it from config.py, and the fields recompute it", async () => {
    await mount(createElement(ScheduleLabExample, { concept: "optim.lr_schedule" }));
    await type(byTestId("schedule-warmup"), inputs.warmup as number);
    await type(byTestId("schedule-total"), inputs.total as number);
    await type(byTestId("schedule-max"), inputs.lr_max as number);
    await type(byTestId("schedule-min"), inputs.lr_min as number);
    const stepCell = byTestId("schedule-step-first");
    const index = steps.indexOf(Number(stepCell.textContent));
    expectClose(exact("schedule-lr-first"), lrs[index]);
  });
});

describe("ClipLab, against clip_grad_norm.above", () => {
  const { inputs, outputs } = getGoldenCase("optim", "clip_grad_norm.above");
  const grads = inputs.grads as { a: number[]; b: number[][] };
  const clipped = outputs.grads as { a: number[]; b: number[][] };

  const mountClip = () =>
    mount(
      createElement(ClipLab, {
        a: grads.a,
        b: grads.b.flat(),
        maxNorm: inputs.max_norm as number,
      }),
    );

  test("norm, factor and every tensor match", async () => {
    await mountClip();
    expectClose(exact("clip-norm"), outputs.norm);
    expectClose(exact("clip-factor"), (inputs.max_norm as number) / (outputs.norm as number));
    expectClose(exact("clip-after-a"), clipped.a);
    expectClose(exact("clip-after-b"), clipped.b.flat());
  });

  test("the chart's length after clipping is max_norm", async () => {
    await mountClip();
    const row = [...page.querySelectorAll("tr")].find((candidate) =>
      candidate.textContent?.startsWith("norm after"),
    );
    expect(Number(row!.lastElementChild!.textContent)).toBeCloseTo(inputs.max_norm as number, 3);
  });

  test("a limit above the norm changes nothing", async () => {
    await mountClip();
    await type(byTestId("clip-limit"), 10);
    expect(byTestId("clip-status").textContent).toContain("no tensor changes");
    expectClose(exact("clip-after-a"), grads.a);
    expectClose(exact("clip-norm"), outputs.norm);
  });

  test("the page mounts it from the same golden input", async () => {
    await mount(createElement(ClipLabExample, { concept: "optim.clip_grad_norm" }));
    expectClose(exact("clip-norm"), outputs.norm);
  });
});

describe("AdamCalculator, against adam.two_step", () => {
  type Tensor = { w: number[][] };
  const { inputs, outputs } = getGoldenCase("optim", "adam.two_step");
  const stepsOut = outputs.steps as {
    params: Tensor;
    m: Tensor;
    v: Tensor;
    t: number;
    mhat: Tensor;
    vhat: Tensor;
    raw: Tensor;
    update: Tensor;
  }[];

  function expectStep(index: number) {
    const step = stepsOut[index]!;
    const entry = (tensor: Tensor) => tensor.w[0]![0];
    expectClose(exact("calc-t"), step.t);
    expectClose(exact("calc-m"), entry(step.m));
    expectClose(exact("calc-v"), entry(step.v));
    expectClose(exact("calc-mhat"), entry(step.mhat));
    expectClose(exact("calc-vhat"), entry(step.vhat));
    expectClose(exact("calc-raw"), entry(step.raw));
    expectClose(exact("calc-update"), entry(step.update));
    expectClose(exact("calc-w"), entry(step.params));
  }

  test("the first step", async () => {
    await mount(createElement(AdamCalculatorExample));
    expectStep(0);
  });

  test("copying the result back, as the caption says, gives the second step", async () => {
    await mount(createElement(AdamCalculatorExample));
    const first = stepsOut[0]!;
    const grads = inputs.grads as { w: number[][] }[];
    await type(byTestId("calc-input-w"), first.params.w[0]![0]!);
    await type(byTestId("calc-input-g"), grads[1]!.w[0]![0]!);
    await type(byTestId("calc-input-m"), first.m.w[0]![0]!);
    await type(byTestId("calc-input-v"), first.v.w[0]![0]!);
    await type(byTestId("calc-input-t"), first.t);
    expectStep(1);
  });

  test("a one-dimensional tensor skips weight decay", async () => {
    await mount(createElement(AdamCalculatorExample));
    await type(byTestId("calc-ndim"), 1);
    const b = stepsOut[0]! as unknown as { params: { b: number[] } };
    expectClose(exact("calc-w"), b.params.b[0]);
    expect(page.querySelector('[data-testid="calc-update"]')).toBeNull();
  });
});

describe("StepCalculator, against steps.counts", () => {
  type Row = {
    windows: number;
    batch_size: number;
    epochs: number;
    steps_per_epoch: number;
    total_steps: number;
    last_size: number;
  };
  const rows = getGoldenCase("train", "steps.counts").outputs.rows as Row[];

  test("every row of the real run", async () => {
    await mount(createElement(StepCountExample, { concept: "train.step_counts" }));
    for (const row of rows) {
      await type(byTestId("steps-windows"), row.windows);
      await type(byTestId("steps-batchSize"), row.batch_size);
      await type(byTestId("steps-epochs"), row.epochs);
      expect(byTestId("steps-per-epoch").textContent).toContain(`= ${row.steps_per_epoch}`);
      expect(byTestId("steps-total").textContent).toContain(`= ${row.total_steps}`);
      expect(byTestId("steps-last").textContent).toContain(`has ${row.last_size} `);
    }
  });
});

describe("ProbabilityLab, against sample_next", () => {
  test("every preset shows the real final distribution", async () => {
    const presets = labPresets();
    await mount(createElement(ProbabilityLab, { presets }));
    for (const preset of presets) {
      await click(byTestId(`lab-preset-${preset.id}`));
      const { outputs } = getGoldenCase("sampling", preset.id);
      if (outputs.kind !== "distribution") continue;
      const p = outputs.p as number[];
      p.forEach((value, token) => expectClose(exact(`lab-final-${token}`), value));
    }
  });

  test("a preset with ties keeps the real count", async () => {
    const presets = labPresets();
    expect(presets.some((preset) => preset.id === "lab.top_k_then_top_p")).toBe(true);
    await mount(createElement(ProbabilityLab, { presets }));
    await click(byTestId("lab-preset-lab.top_k_then_top_p"));
    const { outputs } = getGoldenCase("sampling", "lab.top_k_then_top_p");
    const kept = (outputs.p as number[]).filter((value) => value > 0).length;
    expect(kept).toBe(outputs.keep_count);
    expect(byTestId("lab-table").querySelectorAll("tbody tr").length).toBe(
      (outputs.p as number[]).length,
    );
  });
});

describe("GenerationStepper, against stepper", () => {
  type Inputs = { tokens: string[]; seed_ids: number[] };

  async function walk(id: string) {
    const { inputs, outputs } = getGoldenCase("sampling", id);
    const { tokens, seed_ids } = inputs as unknown as Inputs;
    const ids = outputs.ids as number[];
    const windows = outputs.windows as number[][];
    const next = page.querySelector<HTMLElement>('[data-testid="step-player"]');
    expect(next).not.toBeNull();
    for (let step = 0; step < windows.length; step++) {
      const appended = ids[seed_ids.length + step]!;
      await click(
        [...page.querySelectorAll<HTMLElement>("button")].find(
          (button) => button.textContent === "Next",
        )!,
      );
      expect(byTestId("stepper-window").textContent).toBe(
        windows[step]!.map((token) => tokens[token]).join(" "),
      );
      expect(byTestId("stepper-next").textContent).toBe(tokens[appended]);
    }
  }

  test("config.py settings follow the real generate, step by step", async () => {
    await mount(createElement(GenerationStepperExample));
    await walk("stepper.config_defaults");
  });

  test("greedy mode follows the real generate, step by step", async () => {
    await mount(createElement(GenerationStepperExample));
    await type(byTestId("stepper-mode"), "greedy");
    await walk("stepper.greedy");
  });
});

describe("BpeStepper, against train_bpe_on_text", () => {
  type Merge = { pair: [string, string]; merged: string; frequency: number };
  const next = () =>
    [...page.querySelectorAll<HTMLElement>("button")].find(
      (button) => button.textContent === "Next",
    )!;

  async function walk(id: string) {
    const merges = getGoldenCase("bpe", id).outputs.merges as Merge[];
    for (const merge of merges) {
      await click(next());
      const chosen = byTestId("bpe-chosen").textContent!;
      expect(chosen).toContain(`${showToken(merge.pair[0])} + ${showToken(merge.pair[1])}`);
      expect(chosen).toContain(`frequency ${merge.frequency}`);
      expect(chosen).toContain(showToken(merge.merged));
    }
    await click(next());
    expect(byTestId("bpe-stop").textContent).toContain(`with ${merges.length} merges`);
  }

  test("the first text follows the real merges", async () => {
    await mount(createElement(BpeExample, { concept: "tokenizer.train_bpe" }));
    await walk("train_bpe_on_text.stepper");
  });

  test("the second text follows the real merges", async () => {
    await mount(createElement(BpeExample, { concept: "tokenizer.train_bpe" }));
    const { text } = getGoldenCase("bpe", "train_bpe_on_text.test_file").inputs;
    await type(byTestId("bpe-text"), text as string);
    await walk("train_bpe_on_text.test_file");
  });
});

describe("WindowExplorer, against make_windows.test_file", () => {
  const { outputs } = getGoldenCase("data", "make_windows.test_file");
  const tableRows = () =>
    [...byTestId("window-table").querySelectorAll("tbody tr")].map((row) =>
      [...row.children].map((c) => c.textContent),
    );

  test("starts, inputs, targets and count match", async () => {
    await mount(createElement(WindowExample, { concept: "data.make_windows" }));
    const expected = (outputs.starts as number[]).map((start, w) => [
      String(start),
      (outputs.inputs as number[][])[w]!.join(", "),
      (outputs.targets as number[][])[w]!.join(", "),
    ]);
    expect(tableRows()).toEqual(expected);
    expect(byTestId("window-count").textContent).toContain(`= ${outputs.count}`);
    expect(byTestId("window-shift").textContent).toContain("moved one place left");
  });

  test("too few ids gives Python's message", async () => {
    await mount(createElement(WindowExample, { concept: "data.make_windows" }));
    await type(byTestId("window-ids"), "1, 2");
    await type(byTestId("window-length"), 3);
    expect(byTestId("window-status").textContent).toContain(
      "make_windows raises ValueError: need more than seq_len=3 tokens, got 2",
    );
  });
});

describe("ResidualStream, against block_forward.tiny", () => {
  const { inputs, outputs } = getGoldenCase("block", "block_forward.tiny");
  const fields = { ...outputs, X: inputs.X } as Record<string, number[][]>;

  test("every step shows the real rows, for the selected row", async () => {
    await mount(createElement(ResidualStreamExample, { concept: "transformer.block_forward" }));
    for (const row of [1, 0, 2]) {
      await click(byTestId(`stream-row-${row}`));
      await click(stepButton(STREAM_STEPS[0]!.title));
      for (const [index, step] of STREAM_STEPS.entries()) {
        await click(stepButton(`${index + 1}.`));
        expectClose(
          exact(`stream-panel-${step.id}`),
          step.rows.map((field) => fields[field]![row]),
        );
      }
    }
  });
});

describe("StepPlayer", () => {
  const steps = ["One", "Two", "Three"].map((title) => ({
    id: title,
    title,
    content: createElement("p", { "data-testid": "body" }, `Body of ${title}`),
  }));
  const named = (name: string) =>
    [...page.querySelectorAll<HTMLElement>("button")].find(
      (button) => button.textContent === name,
    )!;

  test("Next and Previous move and stop at the ends, without losing focus", async () => {
    await mount(createElement(StepPlayer, { label: "Test", steps }));
    expect(named("Previous").getAttribute("aria-disabled")).toBe("true");
    expect(named("Previous").hasAttribute("disabled")).toBe(false);
    await click(named("Previous"));
    expect(byTestId("body").textContent).toBe("Body of One");

    await click(named("Next"));
    await click(named("Next"));
    expect(byTestId("body").textContent).toBe("Body of Three");
    expect(byTestId("step-status").textContent).toBe("Step 3 of 3: Three");
    expect(byTestId("step-status").getAttribute("aria-live")).toBe("polite");
    expect(named("Next").getAttribute("aria-disabled")).toBe("true");
    await click(named("Next"));
    expect(byTestId("body").textContent).toBe("Body of Three");

    await click(named("Previous"));
    expect(byTestId("body").textContent).toBe("Body of Two");
  });

  test("the current step is marked by more than colour", async () => {
    await mount(createElement(StepPlayer, { label: "Test", steps }));
    await click(stepButton("2."));
    const current = page.querySelectorAll('[aria-current="step"]');
    expect(current.length).toBe(1);
    expect(current[0]!.textContent).toContain("Two");
  });
});

describe("MatrixInput", () => {
  test("a bad cell keeps the last value and says so, and every cell has a label", async () => {
    const changes: number[][][] = [];
    await mount(
      createElement(MatrixInput, {
        label: "X",
        rowLabels: ["t0"],
        colLabels: ["k0", "k1"],
        value: [[1, 2]],
        onChange: (next: number[][]) => changes.push(next),
      }),
    );
    const input = cell("X, row t0, column k0");
    await type(input, "");
    expect(changes).toEqual([]);
    expect(input.getAttribute("aria-invalid")).toBe("true");
    expect(byTestId("matrix-status").textContent).toContain("last valid value");
    await type(input, 5);
    expect(changes).toEqual([[[5, 2]]]);
    expect(input.getAttribute("aria-invalid")).toBeNull();
  });
});

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { CODE_REFS } from "@/data/code-refs";
import { LEARNING_STAGES, LOG_LINES } from "@/data/learning-step";
import { getSymbol, loadSymbolIndex, requireTarget, targetLines } from "@/lib/code-targets";
import { readPageBody } from "@/lib/content";
import { getGoldenCase, GOLDEN_FILES } from "@/lib/golden";
import { expandBlockName, loadInitParams } from "@/lib/parameters";
import { lrSchedule } from "@/lib/math/optim";
import { LOOP_STEPS, stepCounts } from "@/lib/math/train";
import { chartSteps, scheduleKeyPoints, schedulePoints } from "@/lib/schedule-lab";
import {
  decayCounts,
  decayRows,
  runBatchAverage,
  runValidation,
} from "@/components/TrainingExamples";
import { getConcept } from "@/lib/concepts";
import { linksForEquation } from "@/lib/links";
import { navSlugs } from "@/lib/pages";
import { loadCallGraph, loadConfig } from "@/lib/repo";
import { verifySiteData } from "@/lib/site-verify";

const base = () => ({
  symbols: loadSymbolIndex(),
  callGraph: loadCallGraph(),
  config: loadConfig(),
  navSlugs: navSlugs(),
  parameterNames: [],
  repoFiles: [],
});

const stageProblems = (learningStages: typeof LEARNING_STAGES) =>
  verifySiteData({ ...base(), learningStages }).filter((text) => text.includes("learning stage"));

describe("Part A: the pipeline follows train.main", () => {
  const body = readPageBody("training/how-the-model-learns");
  const firstLine = (id: string) => {
    const stage = LEARNING_STAGES.find((candidate) => candidate.id === id)!;
    return requireTarget(stage.links[0]!.target).startLine;
  };

  test("it has the required stages, in the required order", () => {
    expect(LEARNING_STAGES.map((stage) => stage.label)).toEqual([
      "Input text",
      "Tokenization",
      "Input and target windows",
      "Forward pass",
      "Predicted probabilities",
      "Loss",
      "Backward pass",
      "Gradients",
      "Gradient clipping",
      "Learning-rate schedule",
      "AdamW",
      "Updated parameters",
      "Repeat",
    ]);
  });

  test("gradient, then clip, then learning rate, then optimizer step, by line number", () => {
    const lines = ["gradients", "clipping", "schedule", "adamw"].map(firstLine);
    expect(lines).toEqual([...lines].sort((a, b) => a - b));
    expect(new Set(lines).size).toBe(lines.length);
  });

  test("the statements of the real loop are in that order", () => {
    const { statements } = getGoldenCase("train", "loop.order").outputs as { statements: string[] };
    expect(statements[0]).toContain("batch_loss_and_grads");
    const positions = LOOP_STEPS.map((name) =>
      statements.findIndex((statement) => statement.includes(name)),
    );
    expect(positions).toEqual([1, 2, 3, 4]);
  });

  test("the page lists that order in words", () => {
    const numbered = [...body.matchAll(/^\d\. `?(\w+)/gm)].map((match) => match[1]);
    expect(numbered).toEqual(["batch_loss_and_grads", "clip_grad_norm", "step", "opt"]);
  });

  test("every stage links to a page in the navigation and to real lines, starting in train.main", () => {
    for (const stage of LEARNING_STAGES) {
      expect(navSlugs(), stage.id).toContain(stage.page);
      expect(stage.links.length, stage.id).toBeGreaterThan(0);
      expect(requireTarget(stage.links[0]!.target).symbol.id, stage.id).toBe("train.main");
      expect(getSymbol(stage.showSymbol).id).toBe(stage.showSymbol);
    }
    expect(stageProblems(LEARNING_STAGES)).toEqual([]);
  });

  test("a stage's final hop quotes the expression or calls the function it names", () => {
    for (const stage of LEARNING_STAGES) {
      const last = stage.links.at(-1)!;
      const text = targetLines(requireTarget(last.target));
      if (last.inline) expect(text, stage.id).toContain(last.inline);
      else expect(text, stage.id).toContain(last.call!.split(".").at(-1)!);
    }
  });

  test.each([
    [
      "a swap of clipping and schedule",
      () => {
        const stages = [...LEARNING_STAGES];
        const clip = stages.findIndex((stage) => stage.id === "clipping");
        [stages[clip], stages[clip + 1]] = [stages[clip + 1]!, stages[clip]!];
        return stages;
      },
      "comes before the previous stage",
    ],
    [
      "a call that is not made at that line",
      () =>
        LEARNING_STAGES.map((stage) =>
          stage.id === "clipping"
            ? { ...stage, links: [{ ...stage.links[0]!, call: "optim.lr_schedule" }] }
            : stage,
        ),
      "does not call optim.lr_schedule",
    ],
    [
      "a first hop outside train.main",
      () =>
        LEARNING_STAGES.map((stage) =>
          stage.id === "gradients" ? { ...stage, links: stage.links.slice(1) } : stage,
        ),
      "the first hop must be a line of train.main",
    ],
    [
      "an unknown page",
      () =>
        LEARNING_STAGES.map((stage) =>
          stage.id === "adamw" ? { ...stage, page: "nope/nope" } : stage,
        ),
      "not in the navigation",
    ],
  ])("site verification rejects %s", (_name, make, message) => {
    expect(stageProblems(make()).join("\n")).toContain(message);
  });

  test("the page's quoted logging interval is the one in the code", () => {
    const main = getSymbol("train.main").source;
    expect(main).toContain("if step == 1 or step % 50 == 0:");
    expect(body).toContain("one line every 50 steps, and also at step 1");
  });

  test("the printed lines are the real formats", () => {
    const { printed } = getGoldenCase("train", "learning.one_step").outputs as {
      printed: string[];
    };
    expect(printed).toHaveLength(5);
    expect(printed[0]).toMatch(/^V=\d+ {2}ln V=\d\.\d{3}$/);
    expect(printed[1]).toMatch(/^step {5}1 {2}loss \d\.\d{3} {2}\|g\| \d\.\d{2} {2}lr \d\.\d{5}$/);
    expect(printed[2]).toMatch(/^epoch 1: val loss \d\.\d{3} {2}ppl \d+\.\d$/);
    expect(printed[3]).toMatch(/^ {2}sample: /);
    expect(printed[4]).toBe("saved checkpoint to checkpoints/model.*");
  });

  test("Part A has no claim about what the model focuses on", () => {
    expect(body).not.toMatch(/focus|attends to|understands/i);
  });
});

const STEPS = [
  "What is it?",
  "Why does the model need it?",
  "Mathematics",
  "What does each symbol mean?",
  "Tiny numerical example",
  "Tensor shapes",
  "Actual implementation",
  "Code and math connection",
  "Where is it used?",
];

function expectNineSteps(section: string, where: string) {
  const found = [...section.matchAll(/<ConceptStep>([^<]+)<\/ConceptStep>/g)].map((m) => m[1]);
  expect(found, where).toEqual(STEPS);
}

const sectionsOf = (body: string) =>
  Object.fromEntries(
    body
      .split(/^## /m)
      .slice(1)
      .map((text) => [text.split("\n")[0]!, text]),
  );

describe("Part B: AdamW, schedule and clipping match the code", () => {
  const body = readPageBody("training/adamw-learning-rate-clipping");
  const sections = sectionsOf(body);
  const config = (name: string) =>
    loadConfig().constants.find((constant) => constant.name === name)!.value as number;

  test("each concept follows the nine steps", () => {
    expect(Object.keys(sections)).toEqual(["AdamW", "Learning-rate schedule", "Gradient clipping"]);
    for (const [name, section] of Object.entries(sections)) expectNineSteps(section, name);
  });

  test("every equation of the three concepts appears on the page, once", () => {
    for (const id of ["optim.Adam.step", "optim.lr_schedule", "optim.clip_grad_norm"]) {
      for (const { id: equation } of getConcept(id).equations) {
        expect(body.split(`<MathBlock id="${equation}" />`).length - 1, equation).toBe(1);
      }
    }
  });

  test("the weight-decay list matches the code's ndim rule for every name", () => {
    const rows = decayRows();
    expect(rows.map((row) => row.name)).toEqual(loadInitParams().parameters.map((p) => p.name));
    for (const row of rows) expect(row.shrunk, row.name).toBe(row.ndim >= 2);
    expect(rows.filter((row) => row.shrunk).map((row) => row.name)).toEqual([
      "tok_emb",
      "pos_emb",
      "b{l}.wq",
      "b{l}.wk",
      "b{l}.wv",
      "b{l}.wo",
      "b{l}.ff1_w",
      "b{l}.ff2_w",
      "head_w",
    ]);
    expect(getSymbol("optim.Adam.step").source).toContain("if params[k].ndim >= 2:");
  });

  test("the real step and the engine agree on the keys of the whole dictionary", () => {
    const { rows } = getGoldenCase("optim", "adam.decay_rule").outputs as {
      rows: { name: string; ndim: number; changed: boolean }[];
    };
    const expanded = expandBlockName(loadInitParams().parameters, 2);
    expect(rows.map((row) => row.name)).toEqual(expanded.map((row) => row.key));
    for (const row of rows) expect(row.changed, row.name).toBe(row.ndim >= 2);
  });

  test("the counts are worked out from the shapes, not typed", () => {
    const { decayed, kept, total } = decayCounts();
    const L = config("num_layers");
    const once = 3;
    expect(decayed).toBe(once + 6 * L);
    expect(decayed + kept).toBe(total);
    expect(body).not.toMatch(/\b(15|23) tensors\b/);
  });

  test("schedule values at step 0, warmup end and the final step match golden output", () => {
    const item = getGoldenCase("optim", "lr_schedule.key_points");
    const { steps, warmup, total, lr_max, lr_min } = item.inputs as {
      steps: number[];
      warmup: number;
      total: number;
      lr_max: number;
      lr_min: number;
    };
    const { lrs } = item.outputs as { lrs: number[] };
    expect(warmup).toBe(config("warmup_steps"));
    expect(lr_max).toBe(config("lr_max"));
    expect(lr_min).toBe(config("lr_min"));
    const keys = scheduleKeyPoints({ total, warmup, lrMax: lr_max, lrMin: lr_min });
    const at = (id: string) => keys.find((key) => key.id === id)!;
    const golden = (step: number) => lrs[steps.indexOf(step)]!;
    expect(at("start").step).toBe(0);
    expect(at("start").lr).toBeCloseTo(golden(0), 12);
    expect(at("warmup").lr).toBeCloseTo(golden(warmup), 12);
    expect(at("final").lr).toBeCloseTo(golden(total), 12);
    expect(golden(0)).toBe(0);
    expect(golden(warmup)).toBe(lr_max);
    expect(golden(total)).toBeCloseTo(lr_min, 15);
  });

  test("the chart points are the function's values, at the corner too", () => {
    const settings = { total: 100000, warmup: 50, lrMax: 1e-3, lrMin: 1e-4 };
    const points = schedulePoints(settings);
    expect(points.length).toBeLessThan(300);
    for (const step of [0, 1, 50, 51, 100000]) expect(chartSteps(100000, 50)).toContain(step);
    for (const { x, y } of points) expect(y).toBe(lrSchedule(x, 50, 100000, 1e-3, 1e-4));
    const xs = points.map((point) => point.x);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
    expect(new Set(xs).size).toBe(xs.length);
  });

  test("a warmup longer than training never reaches lr_max", () => {
    const settings = { total: 10, warmup: 50, lrMax: 1e-3, lrMin: 1e-4 };
    expect(chartSteps(10, 50)).not.toContain(50);
    expect(scheduleKeyPoints(settings).some((key) => key.id === "warmup")).toBe(false);
  });

  test("the order clip, schedule, step on the page matches train.py", () => {
    const { startLine, endLine } = requireTarget({
      file: "train.py",
      symbol: "train.main",
      match: "norm = clip_grad_norm[\\s\\S]*?opt\\.step\\(params, grads, lr\\)",
      regex: true,
    });
    expect(endLine - startLine).toBe(4);
    const main = getSymbol("train.main").source;
    const order = ["clip_grad_norm(grads", "step += 1", "lr_schedule(step", "opt.step(params"].map(
      (text) => main.indexOf(text),
    );
    expect(order.every((index) => index >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(body).toContain(
      "So clipping comes first, then the learning rate, then the optimizer step.",
    );
  });

  test("the two-step example reproduces the numbers the repository's test asserts", () => {
    const { steps } = getGoldenCase("optim", "adam.two_step").outputs as {
      steps: {
        params: { w: number[][]; b: number[] };
        update: { w: number[][]; b: number[] };
        raw: { w: number[][] };
      }[];
    };
    expect(steps[0]!.params.b[0]).toBeCloseTo(0.9, 4);
    expect(steps[1]!.params.b[0]).toBeCloseTo(0.8068, 4);
    expect(steps[0]!.update.w[0]![0]).not.toBe(steps[0]!.raw.w[0]![0]);
    const source = getSymbol("tests.test_optim.test_adam_two_step_hand_trace").source;
    expect(source).toContain("0.9000");
    expect(source).toContain("0.8068");
    expect(body).toContain("`0.9000` and `0.8068`");
  });

  test("the page links to code on every equation, and to the tests", () => {
    for (const id of ["adam.decay", "adam.param", "lr.cosine", "clip.norm", "clip.factor"]) {
      const roles = linksForEquation(id).map((link) => link.role);
      expect(roles, id).toContain("implements");
      expect(roles, id).toContain("test");
    }
  });

  test("nothing the code lacks is documented", () => {
    const text = body.toLowerCase();
    for (const word of [
      "sgd",
      "nesterov",
      "per-tensor limit",
      "clip by value",
      "step decay",
      "adagrad",
    ]) {
      expect(text, word).not.toContain(word);
    }
  });
});

describe("Part B: typed numbers in the clipping text are the demo's numbers", () => {
  test("3, 4 and 5", () => {
    const body = readPageBody("training/adamw-learning-rate-clipping");
    expect(body).toContain("one length of 3 and one of 4, which gives a norm of 5");
    const { inputs, outputs } = getGoldenCase("optim", "clip_grad_norm.above");
    const grads = inputs.grads as { a: number[]; b: number[][] };
    expect(Math.hypot(...grads.a)).toBe(3);
    expect(Math.hypot(...grads.b.flat())).toBe(4);
    expect(outputs.norm).toBe(5);
  });
});

describe("Part C: formulas match train.py", () => {
  const body = readPageBody("training/training-loop");
  const sections = sectionsOf(body);
  const batch = getSymbol("train.batch_loss_and_grads").source;

  test("sections, and the nine steps on each concept", () => {
    expect(Object.keys(sections)).toEqual([
      "Mini-batch gradient",
      "Step counts",
      "Validation",
      "The sample printed each epoch",
      "Logging lines",
      "Checkpoint",
    ]);
    for (const name of ["Mini-batch gradient", "Step counts", "Validation"]) {
      expectNineSteps(sections[name]!, name);
    }
  });

  test("the division by B happens inside the loop, on each window", () => {
    const lines = batch.split(/\r?\n/).map((line) => line.trim());
    const loop = lines.indexOf("for x, y in zip(xb, yb):");
    const lossLine = lines.indexOf("total_loss += loss / B");
    const gradLine = lines.indexOf("grads[k] += g[k] / B");
    const end = lines.indexOf("return total_loss, grads");
    expect(lines.indexOf("B = len(xb)")).toBeLessThan(loop);
    expect(lossLine).toBeGreaterThan(loop);
    expect(gradLine).toBeGreaterThan(loop);
    expect(lossLine).toBeLessThan(end);
    expect(gradLine).toBeLessThan(end);
    const add = (id: string) =>
      getConcept("train.batch_loss_and_grads").equations.find((e) => e.id === id)!.latex;
    expect(add("batch.loss-add")).toContain("\\frac{L_i}{B}");
    expect(add("batch.grad-add")).toContain("\\frac{g_{k,i}}{B}");
    expect(sections["Mini-batch gradient"]).toContain(
      "The division by `B` happens inside the loop",
    );
    expect(
      linksForEquation("batch.loss-add").some((l) => l.match === "total_loss += loss / B"),
    ).toBe(true);
    expect(linksForEquation("batch.grad-add").some((l) => l.match === "grads[k] += g[k] / B")).toBe(
      true,
    );
  });

  test("the example's running total equals the real function's result", () => {
    const { loss, rows } = runBatchAverage();
    const golden = getGoldenCase("train", "batch.three_windows").outputs;
    expect(Math.abs(loss - (golden.loss as number))).toBeLessThan(1e-12);
    expect(Math.abs(rows.at(-1)!.running - (golden.loss as number))).toBeLessThan(1e-12);
  });

  test("a batch of one is loss_and_grads itself", () => {
    const { outputs } = getGoldenCase("train", "batch.one_window");
    expect(outputs.loss).toBe((outputs.losses as number[])[0]);
    expect((outputs.grads as { head_b: number[] }).head_b).toEqual(
      (outputs.head_b_each as number[][])[0],
    );
  });

  test("the validation example equals the real function, and ppl is exp of the mean", () => {
    const { loss, ppl, losses } = runValidation();
    const golden = getGoldenCase("train", "validation.tiny").outputs;
    expect(Math.abs(loss - (golden.loss as number))).toBeLessThan(1e-12);
    expect(Math.abs(ppl - (golden.perplexity as number))).toBeLessThan(1e-9);
    expect(ppl).toBeCloseTo(Math.exp(loss), 12);
    // Same windows as the batch case. loss_and_grads and validation_loss differ only in round-off.
    const batchLosses = getGoldenCase("train", "batch.three_windows").outputs.losses as number[];
    losses.forEach((value, i) => expect(Math.abs(value - batchLosses[i]!)).toBeLessThan(1e-12));
  });

  test("the calculator equals the code's ceil arithmetic, on every golden row", () => {
    const { rows } = getGoldenCase("train", "steps.counts").outputs as {
      rows: {
        windows: number;
        batch_size: number;
        epochs: number;
        steps_per_epoch: number;
        total_steps: number;
        yielded: number;
        last_size: number;
      }[];
    };
    expect(rows.length).toBeGreaterThan(3);
    for (const row of rows) {
      const counts = stepCounts(row.windows, row.batch_size, row.epochs);
      expect(counts.stepsPerEpoch).toBe(row.steps_per_epoch);
      expect(counts.totalSteps).toBe(row.total_steps);
      expect(counts.lastBatch).toBe(row.last_size);
      expect(row.yielded).toBe(row.steps_per_epoch);
    }
    expect(getSymbol("train.main").source).toContain(
      "steps_per_epoch = math.ceil(len(tr_x) / config.batch_size)",
    );
  });

  test("the sample section's facts are what the code does", () => {
    const main = getSymbol("train.main").source;
    expect(main).toContain("seed_ids = [int(i) for i in va_x.iloc[0][:3]]");
    expect(main).toContain("out = generate(params, cfg, seed_ids, 30, config.seq_len,");
    expect(main).toContain("config.temperature, config.top_k, config.top_p, rng)");
    expect(main).toContain("iterate_batches(tr_x, tr_y, config.batch_size, rng)");
    expect(main).toContain('print("  sample:", repr(tok.decode(out)))');
    expect(getSymbol("generate.generate").source).toContain("ids = list(seed_ids)");
    const text = sections["The sample printed each epoch"]!;
    expect(text).toContain("first 3 tokens of the first validation window");
    expect(text).toContain("It generates 30 new tokens.");
    const { outputs, inputs } = getGoldenCase("train", "learning.one_step");
    expect(outputs.seed_ids).toHaveLength(3);
    expect(inputs.seq_len).toBeGreaterThan(3);
  });

  test("the logging table has every print in train.main, in file order", () => {
    const main = getSymbol("train.main").source;
    const prints = main.split(/\r?\n/).filter((line) => line.trim().startsWith("print("));
    expect(LOG_LINES).toHaveLength(prints.length);
    const lines = LOG_LINES.map(({ ref }) => requireTarget(CODE_REFS[ref]!).startLine);
    expect(lines).toEqual([...lines].sort((a, b) => a - b));
    LOG_LINES.forEach(({ ref }, index) => {
      expect(targetLines(requireTarget(CODE_REFS[ref]!)).trim()).toBe(prints[index]!.trim());
    });
    expect(getSymbol("train.main").source).toContain("if step == 1 or step % 50 == 0:");
    expect(LOG_LINES[1]!.when).toContain("divisible by 50");
  });

  test("the checkpoint is saved once, after the epoch loop, and the page links its page", () => {
    const lines = getSymbol("train.main").source.split(/\r?\n/);
    const loop = lines.findIndex((line) => line.includes("for epoch in range(config.epochs):"));
    const save = lines.findIndex((line) =>
      line.includes("save_checkpoint(prefix, params, cfg, tok)"),
    );
    expect(save).toBeGreaterThan(loop);
    expect(lines[save]!.match(/^\s*/)![0].length).toBeLessThan(
      lines[loop + 1]!.match(/^\s*/)![0].length,
    );
    expect(sections.Checkpoint).toContain('<PageLink page="code/checkpoints-tests">');
    const saved = getSymbol("checkpoint.save_checkpoint").source;
    expect(saved).toContain(".npz");
    expect(saved).toContain(".json");
    expect(saved).toContain(".tok.json");
  });

  test("the general explanation is labelled, and the page claims no result", () => {
    expect(sections["Mini-batch gradient"]).toMatch(/In general, one window's gradient/);
    expect(body).not.toMatch(/focus|attends to|understands/i);
  });
});

describe("lr_max is reached at the last warmup step", () => {
  const body = readPageBody("training/adamw-learning-rate-clipping");
  const item = getGoldenCase("optim", "lr_schedule.key_points");
  const { steps, warmup, lr_max } = item.inputs as {
    steps: number[];
    warmup: number;
    lr_max: number;
  };
  const { lrs } = item.outputs as { lrs: number[] };

  test("the real function gives lr_max at warmup and less on the next step", () => {
    expect(lrs[steps.indexOf(warmup)]).toBe(lr_max);
    expect(lrs[steps.indexOf(warmup + 1)]).toBeLessThan(lr_max);
  });

  test("the page says so, with the code lines, and names the comment", () => {
    expect(body).toContain("**`lr_max` is reached at the last warmup step.**");
    for (const id of ["schedule.warmup-if", "schedule.warmup-return", "schedule.config-comment"]) {
      expect(body).toContain(`<CodeLine id="${id}" />`);
    }
    expect(body).toContain('says "reached right after warmup"');
    expect(
      getSymbol("config.lr_max").source +
        requireTarget(CODE_REFS["schedule.config-comment"]!).symbol.file,
    ).toContain("config.py");
    const comment = loadConfig().constants.find(({ name }) => name === "lr_max")!.comment;
    expect(comment).toContain("reached right after warmup");
  });

  test("the key-point table lists the first cosine step when there is one", () => {
    const settings = { warmup: 50, lrMax: 1e-3, lrMin: 1e-4 };
    expect(scheduleKeyPoints({ ...settings, total: 200 }).map((key) => key.id)).toContain("after");
    expect(scheduleKeyPoints({ ...settings, total: 50 }).map((key) => key.id)).not.toContain(
      "after",
    );
  });
});

describe("verify:golden stays stable across versions", () => {
  const fixture = JSON.parse(
    readFileSync(join(process.cwd(), "data", "golden", "learning-step.fixture.json"), "utf-8"),
  ) as {
    recorded_with: Record<string, string>;
    xb: number[][];
    yb: number[][];
    sample_ids: number[];
    sample_text: string;
  };
  const { outputs } = getGoldenCase("train", "learning.one_step");

  test("the one-step case uses the stored batch and sample exactly", () => {
    expect(outputs.xb).toEqual(fixture.xb);
    expect(outputs.yb).toEqual(fixture.yb);
    expect(outputs.seed_ids).toEqual(fixture.sample_ids.slice(0, 3));
    const shown = fixture.sample_text.replaceAll("\n", "\\n");
    expect((outputs.printed as string[])[3]).toBe(`  sample: '${shown}'`);
  });

  test("the fixture and every golden file record their versions", () => {
    for (const name of ["python", "numpy", "pandas"]) {
      expect(fixture.recorded_with[name]).toMatch(/^\d+\.\d+/);
    }
    for (const file of GOLDEN_FILES) {
      const { environment } = JSON.parse(
        readFileSync(join(process.cwd(), "data", "golden", `${file}.json`), "utf-8"),
      ) as { environment: Record<string, string> };
      expect(Object.keys(environment).sort(), file).toEqual(["numpy", "pandas", "python"]);
    }
  });

  test("the fresh run does not shuffle or sample for this case", () => {
    const script = readFileSync(join(process.cwd(), "scripts", "make_golden.py"), "utf-8");
    const body = script.slice(
      script.indexOf("def learning_step_case"),
      script.indexOf("def train_file_cases"),
    );
    expect(body).not.toContain("iterate_batches");
    expect(body).not.toContain("generate.generate");
  });
});

describe("the AdamW calculator", () => {
  test("every statement it shows is a real line of Adam.step, in that order", () => {
    const source = readFileSync(join(process.cwd(), "components", "AdamCalculator.tsx"), "utf-8");
    const shown = [...source.matchAll(/code: "([^"]+)"/g)].map((match) => match[1]!);
    expect(shown.length).toBeGreaterThan(6);
    const lines = getSymbol("optim.Adam.step")
      .source.split(/\r?\n/)
      .map((line) => line.trim());
    const positions = shown.map((statement) =>
      lines.findIndex((line) => line.startsWith(statement)),
    );
    positions.forEach((position, i) => expect(position, shown[i]).toBeGreaterThanOrEqual(0));
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });
});

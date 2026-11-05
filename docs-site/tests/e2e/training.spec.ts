import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

async function hydrated(locator: Locator) {
  await expect
    .poll(() =>
      locator.evaluate((element) =>
        Object.keys(element).some((key) => key.startsWith("__reactProps")),
      ),
    )
    .toBe(true);
}

type GoldenFile = { cases: { id: string; inputs: any; outputs: any }[] }; // eslint-disable-line @typescript-eslint/no-explicit-any
const golden = (file: string) =>
  JSON.parse(readFileSync(`data/golden/${file}.json`, "utf-8")) as GoldenFile;
const goldenCase = (file: string, id: string) => golden(file).cases.find((item) => item.id === id)!;

const fileLines = (file: string) => readFileSync(`../${file}`, "utf-8").split(/\r?\n/);

const horizontalOverflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

async function expectExact(locator: Locator, expected: number) {
  const exact = Number(await locator.getAttribute("data-exact"));
  expect(Math.abs(exact - expected), `${exact} against ${expected}`).toBeLessThan(1e-9);
}

const LEARNS = "/training/how-the-model-learns/";

test.describe("How the Model Learns", () => {
  test("every stage opens a panel with a page link, real lines and an example label", async ({
    page,
  }) => {
    await page.goto(LEARNS);
    const diagram = page.getByTestId("learning-pipeline");
    const stages = diagram.locator("[data-testid^=learning-stage-]");
    await expect(stages).toHaveCount(13);
    await hydrated(stages.first());

    const files: Record<string, string[]> = {};
    for (let i = 0; i < 13; i++) {
      await stages.nth(i).click();
      await expect(stages.nth(i)).toHaveAttribute("aria-pressed", "true");
      const panel = page.getByTestId("learning-panel");
      await expect(panel.getByTestId("learning-page")).toContainText("Page:");
      await expect(
        panel.getByTestId("learning-value").getByText("Illustrative Example"),
      ).toBeVisible();

      const links = panel.getByTestId("learning-link");
      const count = await links.count();
      expect(count).toBeGreaterThan(0);
      expect(await links.first().locator("a").innerText()).toMatch(/^train\.py:\d+/);
      for (let l = 0; l < count; l++) {
        const statement = (await links.nth(l).locator("code").innerText()).split("\n")[0]!.trim();
        const location = await links.nth(l).locator("a").innerText();
        const [file, range] = location.split(":");
        const line = Number(range!.split("-")[0]);
        files[file!] ??= fileLines(file!);
        expect(files[file!]![line - 1]!.trim().startsWith(statement), location).toBe(true);
      }
      await expect(panel.getByTestId("learning-code").locator("pre")).toBeVisible();
    }
  });

  test("stages in train.main appear in code order: gradient, clip, schedule, step", async ({
    page,
  }) => {
    await page.goto(LEARNS);
    const stage = (id: string) => page.getByTestId(`learning-stage-${id}`);
    await hydrated(stage("gradients"));
    const mainLine = async (id: string) => {
      await stage(id).click();
      const text = await page.getByTestId("learning-link").first().locator("a").innerText();
      return Number(text.split(":")[1]!.split("-")[0]);
    };
    const lines = [
      await mainLine("gradients"),
      await mainLine("clipping"),
      await mainLine("schedule"),
      await mainLine("adamw"),
    ];
    const real = fileLines("train.py");
    expect(real[lines[0]! - 1]).toContain("batch_loss_and_grads");
    expect(real[lines[1]! - 1]).toContain("clip_grad_norm");
    expect(real[lines[2]! - 1]).toContain("lr_schedule");
    expect(real[lines[3]! - 1]).toContain("opt.step");
    expect(lines).toEqual([...lines].sort((a, b) => a - b));
  });

  test("the highlighted line in the panel is the stage's own line", async ({ page }) => {
    await page.goto(LEARNS);
    const stage = page.getByTestId("learning-stage-updated");
    await hydrated(stage);
    await stage.click();
    const highlighted = page
      .getByTestId("learning-code")
      .locator("[data-highlighted=true]")
      .first();
    await expect(highlighted).toContainText("params[k] -= lr * update");
  });

  test("the walk-through equals the real functions", async ({ page }) => {
    await page.goto(LEARNS);
    const { rows } = goldenCase("optim", "adam.one_weight").outputs as {
      rows: Record<string, number | number[][]>[];
    };
    const value = (item: number | number[][]) => (typeof item === "number" ? item : item[0]![0]!);
    const pairs: [string, string][] = [
      ["norm", "norm"],
      ["clipped", "clipped"],
      ["lr", "lr"],
      ["m", "m"],
      ["v", "v"],
      ["mhat", "mhat"],
      ["vhat", "vhat"],
      ["raw", "raw"],
      ["update", "update"],
      ["after", "w"],
    ];
    for (let i = 0; i < 2; i++) {
      for (const [testId, key] of pairs) {
        await expectExact(page.getByTestId(`weight-${testId}-${i}`), value(rows[i]![key]!));
      }
    }
    await expect(page.getByText("Illustrative Example").first()).toBeVisible();
  });

  test("the printed sample lines are the real output of the format strings", async ({ page }) => {
    await page.goto(LEARNS);
    const { printed } = goldenCase("train", "learning.one_step").outputs as { printed: string[] };
    await expect(page.getByTestId("log-sample")).toHaveText(`${printed[1]}\n${printed[2]}`);
  });

  for (const { width, height, theme } of [
    { width: 1280, height: 800, theme: "light" },
    { width: 1280, height: 800, theme: "dark" },
    { width: 1000, height: 800, theme: "light" },
    { width: 390, height: 800, theme: "light" },
    { width: 390, height: 800, theme: "dark" },
  ]) {
    test(`no axe findings and no sideways scroll at ${width} ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.emulateMedia({ colorScheme: theme as "light" | "dark" });
      await page.goto(LEARNS);
      const results = await new AxeBuilder({ page }).analyze();
      expect(results.violations).toEqual([]);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
    });
  }
});

const ADAMW = "/training/adamw-learning-rate-clipping/";

test.describe("AdamW / Learning Rate / Clipping", () => {
  const config = JSON.parse(readFileSync("data/generated/config.json", "utf-8")) as {
    constants: { name: string; value: number }[];
  };
  const configValue = (name: string) => config.constants.find((c) => c.name === name)!.value;

  test("sections are in the order of the code", async ({ page }) => {
    await page.goto(ADAMW);
    await expect(page.locator("main").getByRole("heading", { level: 2 })).toHaveText([
      "AdamW",
      "Learning-rate schedule",
      "Gradient clipping",
    ]);
  });

  test("the quoted loop lines are the file's lines, clip before schedule before step", async ({
    page,
  }) => {
    await page.goto(ADAMW);
    const text = (await page.locator("main figure").first().innerText()).replaceAll("\r", "");
    const clip = text.indexOf("clip_grad_norm");
    const schedule = text.indexOf("lr_schedule");
    const step = text.indexOf("opt.step");
    expect(clip).toBeGreaterThan(0);
    expect(schedule).toBeGreaterThan(clip);
    expect(step).toBeGreaterThan(schedule);
  });

  test("the decay table has exactly the keys the ndim rule says", async ({ page }) => {
    await page.goto(ADAMW);
    const { parameters } = JSON.parse(readFileSync("data/derived/init-params.json", "utf-8")) as {
      parameters: { name: string; dims: string[] }[];
    };
    const rows = page.locator("[data-testid=decay-table] tbody tr");
    await expect(rows).toHaveCount(parameters.length);
    for (const { name, dims } of parameters) {
      await expect(rows.and(page.locator(`[data-key="${name}"]`))).toHaveAttribute(
        "data-decay",
        String(dims.length >= 2),
      );
    }
    await expect(page.getByTestId("decay-counts")).toContainText(
      "15 tensors get decay and 23 do not",
    );
  });

  test("the two-step example equals the real function", async ({ page }) => {
    await page.goto(ADAMW);
    const { steps } = goldenCase("optim", "adam.two_step").outputs as {
      steps: {
        params: Record<string, number[] | number[][]>;
        update: Record<string, number[] | number[][]>;
      }[];
    };
    for (const index of [0, 1]) {
      for (const name of ["w", "b"]) {
        await expectExact(
          page.getByTestId(`two-step-after-${index}-${name}`),
          steps[index]!.params[name]!.flat()[0] as number,
        );
        await expectExact(
          page.getByTestId(`two-step-update-${index}-${name}`),
          steps[index]!.update[name]!.flat()[0] as number,
        );
      }
    }
  });

  test("the calculator reproduces both steps of the real function", async ({ page }) => {
    await page.goto(ADAMW);
    const { steps } = goldenCase("optim", "adam.two_step").outputs as {
      steps: {
        params: { w: number[][]; b: number[] };
        m: { w: number[][] };
        v: { w: number[][] };
      }[];
    };
    const { grads } = goldenCase("optim", "adam.two_step").inputs as {
      grads: { w: number[][] }[];
    };
    await expectExact(page.getByTestId("calc-w"), steps[0]!.params.w[0]![0]!);
    const select = page.getByTestId("calc-ndim");
    await hydrated(select);
    await select.selectOption("1");
    await expectExact(page.getByTestId("calc-w"), steps[0]!.params.b[0]!);
    await select.selectOption("2");

    const fill = (key: string, value: number) =>
      page.getByTestId(`calc-input-${key}`).fill(String(value));
    await fill("w", steps[0]!.params.w[0]![0]!);
    await fill("g", grads[1]!.w[0]![0]!);
    await fill("m", steps[0]!.m.w[0]![0]!);
    await fill("v", steps[0]!.v.w[0]![0]!);
    await fill("t", 1);
    await expectExact(page.getByTestId("calc-w"), steps[1]!.params.w[0]![0]!);
    await expect(page.getByTestId("calc-t")).toHaveText("2");
  });

  test("the schedule shows golden values at step 0, warmup end and the last step", async ({
    page,
  }) => {
    await page.goto(ADAMW);
    const item = goldenCase("optim", "lr_schedule.key_points");
    const { steps, warmup, total } = item.inputs as {
      steps: number[];
      warmup: number;
      total: number;
    };
    const { lrs } = item.outputs as { lrs: number[] };
    const at = (step: number) => lrs[steps.indexOf(step)]!;
    await expectExact(page.getByTestId("schedule-lr-start"), at(0));
    await expectExact(page.getByTestId("schedule-lr-first"), at(1));
    await expectExact(page.getByTestId("schedule-lr-warmup"), at(warmup));
    await expectExact(page.getByTestId("schedule-lr-final"), at(total));
    await expect(page.getByTestId("schedule-warmup")).toHaveValue(
      String(configValue("warmup_steps")),
    );
    await expect(page.getByTestId("schedule-max")).toHaveValue(String(configValue("lr_max")));
    await expect(page.getByTestId("schedule-min")).toHaveValue(String(configValue("lr_min")));
  });

  test("the schedule line is to scale, and follows edits", async ({ page }) => {
    await page.goto(ADAMW);
    const lab = page.getByTestId("schedule-lab");
    const points = async () =>
      (await lab.getByTestId("line-path").getAttribute("points"))!
        .split(" ")
        .map((pair) => pair.split(",").map(Number) as [number, number]);
    const check = async (total: number, warmup: number) => {
      const line = await points();
      // Straight up during warmup: x and height above the axis grow together.
      const corner = line.findIndex(([, y]) => Math.abs(y) < 1e-9);
      expect(corner).toBeGreaterThan(0);
      expect(line[corner]![0]).toBeCloseTo((warmup / total) * 100, 6);
      expect(line[0]).toEqual([0, 100]);
      expect(line.at(-1)![0]).toBeCloseTo(100, 6);
      // lr_min sits at lr_min / lr_max of the height.
      expect(100 - line.at(-1)![1]).toBeCloseTo(10, 6);
    };
    const total = page.getByTestId("schedule-total");
    await hydrated(total);
    await check(200, 50);
    await total.fill("1000");
    await page.getByTestId("schedule-warmup").fill("100");
    await expect(page.getByTestId("schedule-step-final")).toHaveText("1000");
    await check(1000, 100);
    await total.fill("0");
    await expect(page.getByTestId("schedule-status")).toContainText("whole number");
    await total.fill("10");
    await expect(page.getByTestId("schedule-status")).toContainText("never reaches lr_max");
  });

  test("the clipping demo equals golden output, scaling all tensors by one factor", async ({
    page,
  }) => {
    await page.goto(ADAMW);
    const above = goldenCase("optim", "clip_grad_norm.above").outputs as {
      norm: number;
      grads: { a: number[]; b: number[][] };
    };
    await expectExact(page.getByTestId("clip-norm"), above.norm);
    await expectExact(page.getByTestId("clip-factor"), 1 / above.norm);
    expect(
      JSON.parse((await page.getByTestId("clip-after-a").getAttribute("data-exact"))!),
    ).toEqual(above.grads.a);
    expect(
      JSON.parse((await page.getByTestId("clip-after-b").getAttribute("data-exact"))!),
    ).toEqual(above.grads.b.flat());
    const limit = page.getByTestId("clip-limit");
    await hydrated(limit);
    await limit.fill("10");
    const below = goldenCase("optim", "clip_grad_norm.below").outputs as {
      grads: { a: number[]; b: number[][] };
    };
    await expect(page.getByTestId("clip-status")).toContainText("no tensor changes");
    await expect(page.getByTestId("clip-factor")).toHaveCount(0);
    expect(
      JSON.parse((await page.getByTestId("clip-after-a").getAttribute("data-exact"))!),
    ).toEqual(below.grads.a);
  });

  for (const { width, height, theme } of [
    { width: 1280, height: 800, theme: "light" },
    { width: 1280, height: 800, theme: "dark" },
    { width: 1000, height: 800, theme: "light" },
    { width: 390, height: 800, theme: "light" },
    { width: 390, height: 800, theme: "dark" },
  ]) {
    test(`no axe findings and no sideways scroll at ${width} ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.emulateMedia({ colorScheme: theme as "light" | "dark" });
      await page.goto(ADAMW);
      const results = await new AxeBuilder({ page }).analyze();
      expect(results.violations).toEqual([]);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
    });
  }
});

const LOOP = "/training/training-loop/";

test.describe("Training Loop", () => {
  test("sections are in the required order", async ({ page }) => {
    await page.goto(LOOP);
    await expect(page.locator("main").getByRole("heading", { level: 2 })).toHaveText([
      "Mini-batch gradient",
      "Step counts",
      "Validation",
      "The sample printed each epoch",
      "Logging lines",
      "Checkpoint",
    ]);
  });

  test("the batch and validation examples equal the real functions", async ({ page }) => {
    await page.goto(LOOP);
    const batch = goldenCase("train", "batch.three_windows").outputs as {
      loss: number;
      losses: number[];
    };
    for (const [index, loss] of batch.losses.entries()) {
      await expectExact(page.getByTestId(`batch-loss-${index}`), loss);
      await expectExact(page.getByTestId(`batch-part-${index}`), loss / batch.losses.length);
    }
    await expectExact(page.getByTestId("batch-running-2"), batch.loss);
    await expectExact(page.getByTestId("batch-total"), batch.loss);

    const validation = goldenCase("train", "validation.tiny").outputs as {
      loss: number;
      losses: number[];
      perplexity: number;
    };
    for (const [index, loss] of validation.losses.entries()) {
      await expectExact(page.getByTestId(`validation-loss-${index}`), loss);
    }
    await expectExact(page.getByTestId("validation-mean"), validation.loss);
    await expectExact(page.getByTestId("validation-ppl"), validation.perplexity);
  });

  test("the calculator equals the real ceil arithmetic on every golden row", async ({ page }) => {
    await page.goto(LOOP);
    const { rows } = goldenCase("train", "steps.counts").outputs as {
      rows: {
        windows: number;
        batch_size: number;
        epochs: number;
        steps_per_epoch: number;
        total_steps: number;
        last_size: number;
      }[];
    };
    const windows = page.getByTestId("steps-windows");
    await hydrated(windows);
    for (const row of rows) {
      await windows.fill(String(row.windows));
      await page.getByTestId("steps-batchSize").fill(String(row.batch_size));
      await page.getByTestId("steps-epochs").fill(String(row.epochs));
      await expect(page.getByTestId("steps-per-epoch")).toHaveText(
        `steps_per_epoch = ceil(${row.windows} / ${row.batch_size}) = ${row.steps_per_epoch}`,
      );
      await expect(page.getByTestId("steps-total")).toHaveText(
        `total_steps = ${row.epochs} * ${row.steps_per_epoch} = ${row.total_steps}`,
      );
      await expect(page.getByTestId("steps-last")).toContainText(String(row.last_size));
    }
    await windows.fill("0");
    await expect(page.getByTestId("steps-status")).toContainText("whole number");
    await expect(page.getByTestId("steps-total")).toHaveCount(0);
  });

  test("each logged output is the real format, and each line links to its print", async ({
    page,
  }) => {
    await page.goto(LOOP);
    const { printed } = goldenCase("train", "learning.one_step").outputs as { printed: string[] };
    const real = fileLines("train.py");
    for (const [index, text] of printed.entries()) {
      const row = page.getByTestId(`log-line-${index}`);
      await expect(page.getByTestId(`log-output-${index}`)).toHaveText(text);
      const location = await row.locator("a.code-link").innerText();
      expect(real[Number(location.split(":")[1]) - 1]!.trim().startsWith("print(")).toBe(true);
    }
  });

  test("the sample code and the checkpoint code are the file's lines", async ({ page }) => {
    await page.goto(LOOP);
    const text = (await page.locator("main").innerText()).replaceAll("\r", "");
    expect(text).toContain("seed_ids = [int(i) for i in va_x.iloc[0][:3]]");
    expect(text).toContain("save_checkpoint(prefix, params, cfg, tok)");
    await expect(page.locator("main").getByText("Checkpoints & Tests").first()).toBeVisible();
  });

  test("the page before links here now", async ({ page }) => {
    await page.goto(LEARNS);
    await expect(
      page.locator("main").getByRole("link", { name: "Training Loop" }).first(),
    ).toBeVisible();
    await expect(
      page.locator("main").getByRole("link", { name: "AdamW / Learning Rate / Clipping" }).first(),
    ).toBeVisible();
  });

  for (const { width, height, theme } of [
    { width: 1280, height: 800, theme: "light" },
    { width: 1280, height: 800, theme: "dark" },
    { width: 1000, height: 800, theme: "light" },
    { width: 390, height: 800, theme: "light" },
    { width: 390, height: 800, theme: "dark" },
  ]) {
    test(`no axe findings and no sideways scroll at ${width} ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await page.emulateMedia({ colorScheme: theme as "light" | "dark" });
      await page.goto(LOOP);
      const results = await new AxeBuilder({ page }).analyze();
      expect(results.violations).toEqual([]);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
    });
  }
});

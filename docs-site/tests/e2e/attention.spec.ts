import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const PAGE = "/mathematics/attention/";

type GoldenFile = { cases: { id: string; inputs: any; outputs: any }[] }; // eslint-disable-line @typescript-eslint/no-explicit-any
const golden = JSON.parse(readFileSync("data/golden/attention.json", "utf-8")) as GoldenFile;
const forward = golden.cases.find((item) => item.id === "attention_forward.cat_sat")!.outputs;
const backward = golden.cases.find((item) => item.id === "attention_backward.cat_sat")!.outputs;
const shapes = golden.cases.find((item) => item.id === "attention_forward.config_shapes")!.outputs;

const realLines = readFileSync("../transformer.py", "utf-8").split(/\r?\n/);

type Nested = number | string | null | Nested[];

// JSON has no infinity: the golden files store "-Infinity", and data-exact on the page holds null.
function expectClose(actual: Nested, expected: Nested, path = "value") {
  if (expected === "-Infinity") {
    expect(actual, path).toBeNull();
    return;
  }
  if (typeof expected === "number") {
    expect(typeof actual, path).toBe("number");
    expect(Math.abs((actual as number) - expected), `${path}: ${actual}, ${expected}`).toBeLessThan(
      1e-9,
    );
    return;
  }
  expect((actual as Nested[]).length, path).toBe((expected as Nested[]).length);
  (expected as Nested[]).forEach((item, i) =>
    expectClose((actual as Nested[])[i]!, item, `${path}[${i}]`),
  );
}

async function expectExact(locator: Locator, expected: Nested) {
  await expect
    .poll(async () => {
      const exact = await locator.getAttribute("data-exact");
      try {
        expectClose(JSON.parse(exact!) as Nested, expected);
        return true;
      } catch {
        return exact;
      }
    })
    .toBe(true);
}

async function hydrated(locator: Locator) {
  await expect
    .poll(() =>
      locator.evaluate((element) =>
        Object.keys(element).some((key) => key.startsWith("__reactProps")),
      ),
    )
    .toBe(true);
}

const horizontalOverflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

async function openLab(page: Page) {
  await page.goto(PAGE);
  const lab = page.getByTestId("attention-lab");
  await hydrated(lab.getByTestId("lab-token-0"));
  return lab;
}

const stepButton = (lab: Locator, name: string) => lab.getByRole("button", { name, exact: true });

const highlighted = (lab: Locator) =>
  lab
    .getByTestId("lab-code")
    .locator("[data-highlighted=true]")
    .evaluateAll((lines) => lines.map((line) => Number(line.getAttribute("data-line"))));

test.describe("the page", () => {
  test("sections are in the required order", async ({ page }) => {
    await page.goto(PAGE);
    await expect(page.locator("main").getByRole("heading", { level: 2 })).toHaveText([
      "Why attention?",
      "Q, K, V projections",
      "Splitting into heads",
      "Scores",
      "Scaling by sqrt(d_h)",
      "Causal mask",
      "Softmax per row",
      "Weighted values",
      "Merging heads",
      "Output projection",
      "The transformer.py implementation",
      "Tensor shapes with the real config",
      "Backward pass",
      "Role in the model",
    ]);
  });

  test("the real config values are on the page", async ({ page }) => {
    await page.goto(PAGE);
    const main = page.locator("main");
    await expect(main.locator('[data-config-value="num_heads"]').first()).toHaveText("2");
    await expect(main.locator('[data-config-value="head_dim"]').first()).toHaveText("16");
    await expect(main).toContainText("sub-word tokens");
    await expect(page.getByTestId("ex-real-dh")).toHaveText("d_h = 16");
  });

  test("every line of each function equals the file, with its real number", async ({ page }) => {
    await page.goto(PAGE);
    for (const [fn, first, last] of [
      ["attention_forward", 70, 86],
      ["attention_backward", 92, 109],
      ["split_heads", 60, 62],
      ["merge_heads", 65, 67],
    ] as const) {
      // The lab holds a copy of attention_forward and comes first, so take the last full block.
      const block = page
        .locator("main figure")
        .filter({ has: page.locator(`[data-line="${first}"]`) })
        .filter({ has: page.locator(`[data-line="${last}"]`) })
        .last();
      const rows = await block.locator("[data-line]").evaluateAll((items) =>
        items.map((item) => ({
          line: Number(item.getAttribute("data-line")),
          number: item.querySelector("span")!.textContent,
          text: [...item.children]
            .slice(1)
            .map((token) => token.textContent)
            .join(""),
        })),
      );
      expect(rows.map((row) => row.line)).toEqual(
        Array.from({ length: last - first + 1 }, (_, i) => first + i),
      );
      for (const row of rows) {
        expect(row.number).toBe(String(row.line));
        expect(row.text.trimEnd(), `${fn}:${row.line}`).toBe(realLines[row.line - 1]!.trimEnd());
      }
    }
  });

  test("step links point at real sections", async ({ page }) => {
    const lab = await openLab(page);
    for (const step of ["Scores", "Scale", "Mask", "Softmax", "Output"]) {
      await stepButton(lab, step).click();
      const href = await lab.getByRole("link", { name: "Read this step" }).getAttribute("href");
      await expect(page.locator(`h2[id="${href!.slice(1)}"]`), step).toHaveCount(1);
    }
  });

  test("the shape visualizer shows the stored shapes at the real config", async ({ page }) => {
    await page.goto(PAGE);
    const flow = page.getByTestId("attention-shapes");
    const stages = flow.getByTestId("shape-stage");
    const expected = [...Object.values(shapes.forward), ...Object.values(shapes.backward)];
    await expect(stages).toHaveCount(expected.length);
    await expect(stages.nth(0)).toContainText("Xn");
    await expect(stages.nth(0)).toContainText("(16, 32)");
    await expect(flow).toContainText("(2, 16, 16)");
    await expect(page.getByTestId("grad-shapes")).toContainText("(32, 32)");
  });
});

test.describe("the lab", () => {
  test("each step's matrix equals the real function, in both heads", async ({ page }) => {
    const lab = await openLab(page);
    const matrix = lab.getByTestId("lab-matrix");
    for (const head of [0, 1]) {
      await lab.getByTestId(`lab-head-${head}`).click();
      await stepButton(lab, "Scale").click();
      await expectExact(matrix, forward.scaled[head]);
      await stepButton(lab, "Mask").click();
      await expectExact(matrix, forward.masked[head]);
      await stepButton(lab, "Softmax").click();
      await expectExact(matrix, forward.A[head]);
    }
  });

  test("masked cells are hatched and exactly 0, and rows sum to 1", async ({ page }) => {
    const lab = await openLab(page);
    await stepButton(lab, "Softmax").click();
    const exact = JSON.parse(
      (await lab.getByTestId("lab-matrix").getAttribute("data-exact"))!,
    ) as number[][];
    exact.forEach((row, i) => {
      expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-12);
      row.forEach((value, j) => {
        if (j > i) expect(value).toBe(0);
        else expect(value).toBeGreaterThan(0);
      });
    });
    await expect(lab.getByTestId("lab-matrix").locator("svg g[data-masked]")).toHaveCount(15);
    await stepButton(lab, "Mask").click();
    await expect(lab.getByTestId("lab-matrix").locator("svg g[data-masked]")).toHaveCount(15);
    await expect(lab.getByTestId("lab-matrix").locator("svg")).toContainText("-inf");
  });

  test("clicking a step moves the highlight to its lines", async ({ page }) => {
    const lab = await openLab(page);
    const expected: [string, string[]][] = [
      ["Scores", ["S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)"]],
      ["Scale", ["S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)"]],
      ["Mask", ["S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, S)"]],
      ["Softmax", ["A = softmax_forward(S)"]],
      ["Output", ["O = merge_heads(A @ Vh)", 'out = O @ params[p + "wo"] + params[p + "bo"]']],
    ];
    for (const [step, code] of expected) {
      await stepButton(lab, step).click();
      await expect(page.getByTestId("step-status")).toContainText(step);
      const lines = await highlighted(lab);
      expect(
        lines.map((line) => realLines[line - 1]!.trim()),
        step,
      ).toEqual(code);
    }
  });

  test("the selected token moves the row, the bars and the formula", async ({ page }) => {
    const lab = await openLab(page);
    await stepButton(lab, "Output").click();
    const token = lab.getByTestId("lab-token-4");
    await token.click();
    await expect(token).toHaveAttribute("aria-pressed", "true");
    await expectExact(lab.getByTestId("lab-weights"), forward.A[0][4]);
    await expect(lab.getByTestId("lab-formula")).toContainText("(A @ Vh)[4] =");
    const selected = lab.getByTestId("heatmap-selected-row");
    expect(Number(await selected.getAttribute("y"))).toBeGreaterThan(0);
    await expectExact(lab.getByTestId("lab-o"), forward.O);
    await expectExact(lab.getByTestId("lab-out"), forward.out);
  });

  test("the data source is Illustrative, and Actual Model Output is disabled", async ({ page }) => {
    const lab = await openLab(page);
    const source = lab.getByTestId("lab-source");
    await expect(source.getByRole("radio", { name: "Illustrative Example" })).toBeChecked();
    const actual = source.getByRole("radio", { name: "Actual Model Output" });
    await expect(actual).toBeDisabled();
    await expect(source.locator("label[title]")).toHaveAttribute("title", /not available yet/);
    await expect(source).toContainText("Actual Model Output is not available yet.");
  });

  test("badges: every lab figure is Illustrative, and the slot is Actual with no numbers", async ({
    page,
  }) => {
    const lab = await openLab(page);
    await stepButton(lab, "Output").click();
    await expect(lab.locator("figcaption").getByText("Illustrative Example")).toHaveCount(2);
    const slot = page.getByTestId("actual-output-slot");
    await expect(slot.getByText("Actual Model Output")).toBeVisible();
    await expect(slot).toContainText("No data yet");
    await expect(slot.locator("table, svg, [data-exact]")).toHaveCount(0);
    await expect(page.locator("main figcaption").getByText("Actual Model Output")).toHaveCount(0);
  });
});

test.describe("the backward example", () => {
  test("values equal the real attention_backward", async ({ page }) => {
    await page.goto(PAGE);
    await expectExact(page.getByTestId("back-doh"), backward.dOh[0]);
    await expectExact(page.getByTestId("back-dvh"), backward.dVh[0]);
    await expectExact(page.getByTestId("back-da"), backward.dA[0]);
    await expectExact(page.getByTestId("back-ds"), backward.dS[0]);
    await expectExact(page.getByTestId("back-dqh"), backward.dQh[0]);
    await expectExact(page.getByTestId("back-dkh"), backward.dKh[0]);
    await expectExact(page.getByTestId("back-dxn"), backward.dXn);
    await expectExact(page.getByTestId("back-wq"), backward.grads.wq);
    await expectExact(page.getByTestId("back-wo"), backward.grads.wo);
  });

  test("a masked cell has a weight of 0, a non-zero dA and a dS of 0", async ({ page }) => {
    await page.goto(PAGE);
    await expect(page.getByTestId("masked-a")).toHaveAttribute("data-exact", "0");
    await expect(page.getByTestId("masked-ds")).toHaveAttribute("data-exact", /^-?0$/);
    expect(Number(await page.getByTestId("masked-da").getAttribute("data-exact"))).not.toBe(0);
  });

  test("the formula agrees with central differences", async ({ page }) => {
    await page.goto(PAGE);
    const gaps = page.getByTestId("attention-check").getByTestId("largest-gap");
    await expect(gaps).toHaveCount(5);
    for (const value of await gaps.evaluateAll((cells) =>
      cells.map((cell) => Number(cell.getAttribute("data-exact"))),
    )) {
      expect(value).toBeLessThan(1e-8);
    }
  });
});

test.describe("at 390px", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("the heatmap scrolls inside its own container, not the page", async ({ page }) => {
    const lab = await openLab(page);
    for (const step of ["Scores", "Scale", "Mask", "Softmax", "Output"]) {
      await stepButton(lab, step).click();
      const scroller = lab.getByTestId("lab-matrix").getByRole("group").first();
      await expect(scroller).toBeVisible();
      expect(await scroller.evaluate((el) => getComputedStyle(el).overflowX)).toBe("auto");
      expect(await scroller.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
    }
  });
});

for (const [width, theme] of [
  [1280, "light"],
  [1280, "dark"],
  [1000, "light"],
  [390, "light"],
  [390, "dark"],
] as const) {
  test.describe(`at ${width}px in ${theme}`, () => {
    test.use({ viewport: { width, height: 844 }, colorScheme: theme });

    for (const step of ["Scores", "Output"]) {
      test(`${step}: no axe findings and no sideways scroll`, async ({ page }) => {
        const lab = await openLab(page);
        await stepButton(lab, step).click();
        const { violations } = await new AxeBuilder({ page }).analyze();
        expect(violations.map(({ id, nodes }) => `${id}: ${nodes[0]?.html}`)).toEqual([]);
        expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
      });
    }
  });
}

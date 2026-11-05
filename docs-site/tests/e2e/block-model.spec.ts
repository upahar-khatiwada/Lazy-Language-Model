import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const BLOCK = "/mathematics/transformer-block/";

type GoldenFile = { cases: { id: string; inputs: any; outputs: any }[] }; // eslint-disable-line @typescript-eslint/no-explicit-any
const blockGolden = JSON.parse(readFileSync("data/golden/block.json", "utf-8")) as GoldenFile;
const blockCase = (id: string) => blockGolden.cases.find((item) => item.id === id)!;

const realLines = readFileSync("../transformer.py", "utf-8").split(/\r?\n/);

type Nested = number | Nested[];

function expectClose(actual: Nested, expected: Nested, path = "value") {
  if (typeof expected === "number") {
    expect(Math.abs((actual as number) - expected), `${path}: ${actual}, ${expected}`).toBeLessThan(
      1e-9,
    );
    return;
  }
  expect((actual as Nested[]).length, path).toBe(expected.length);
  expected.forEach((item, i) => expectClose((actual as Nested[])[i]!, item, `${path}[${i}]`));
}

async function expectExact(locator: Locator, expected: Nested) {
  const exact = await locator.getAttribute("data-exact");
  expectClose(JSON.parse(exact!) as Nested, expected);
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

const highlighted = (container: Locator) =>
  container
    .locator("[data-highlighted=true]")
    .evaluateAll((lines) => lines.map((line) => Number(line.getAttribute("data-line"))));

// A displayed statement must be the text of the file at the displayed line.
async function expectStatementsMatchFile(flow: Locator, expected: string[]) {
  const steps = flow.locator("[data-testid^=flow-step-]");
  await expect(steps).toHaveCount(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const step = steps.nth(i);
    const statement = await step.getByTestId("flow-statement").innerText();
    expect(statement).toBe(expected[i]);
    const location = await step.locator("a.code-link").innerText();
    const line = Number(location.split(":")[1]);
    expect(realLines[line - 1]!.trim().startsWith(statement), location).toBe(true);
  }
}

test.describe("Full Model", () => {
  const FULL = "/mathematics/full-model/";
  const modelGolden = JSON.parse(readFileSync("data/golden/model.json", "utf-8")) as GoldenFile;
  const modelCase = (id: string) => modelGolden.cases.find((item) => item.id === id)!;

  test("sections are in the required order", async ({ page }) => {
    await page.goto(FULL);
    await expect(page.locator("main").getByRole("heading", { level: 2 })).toHaveText([
      "Forward pass",
      "Backward pass",
      "Gradient flow",
      "The gradient dictionary",
      "Parameter keys",
    ]);
  });

  test("the diagram follows loss_and_grads, statement by statement, one arrow per block", async ({
    page,
  }) => {
    await page.goto(FULL);
    const block = 'dX = block_backward(dX, cache["blocks"][l], params, l, cfg, grads)';
    await expectStatementsMatchFile(page.getByTestId("gradient-flow"), [
      "logits, cache = forward(params, ids, cfg)",
      "loss, probs = cross_entropy_forward(logits, targets)",
      "grads = {k: np.zeros_like(v) for k, v in params.items()}",
      "dZ = cross_entropy_backward(probs, targets)",
      'dXf, dW, db = linear_backward(cache["Xf"], params["head_w"], dZ)',
      'dX, dg, db = layer_norm_backward(dXf, cache["lnf"], params["lnf_g"])',
      block,
      block,
      "embed_backward(dX, ids, grads)",
    ]);
    const flow = page.getByTestId("gradient-flow");
    await expect(flow.getByTestId("flow-node-dX-block-1")).toContainText("Through block 1");
    await expect(flow.getByTestId("flow-node-dX-block-0")).toContainText("Through block 0");
    const order = await flow
      .locator("[data-testid^=flow-node-dX-block]")
      .evaluateAll((nodes) => nodes.map((node) => node.getAttribute("data-testid")));
    expect(order).toEqual(["flow-node-dX-block-1", "flow-node-dX-block-0"]);
  });

  test("selecting a box shows the function and the lines", async ({ page }) => {
    await page.goto(FULL);
    const flow = page.getByTestId("gradient-flow");
    const head = flow.getByTestId("flow-node-dXf");
    await hydrated(head);
    await head.click();
    await expect(head).toHaveAttribute("aria-pressed", "true");
    await expect(flow.getByTestId("flow-selected")).toContainText("dXf");
    await expect(flow.getByTestId("flow-callee")).toContainText("linear_backward");
    const lines = await highlighted(flow.getByTestId("flow-caller"));
    expect(lines.map((line) => realLines[line - 1]!.trim())).toEqual([
      'dXf, dW, db = linear_backward(cache["Xf"], params["head_w"], dZ)',
      'grads["head_w"] += dW',
      'grads["head_b"] += db',
    ]);
    await flow.getByTestId("flow-node-dX-block-0").click();
    await expect(flow.getByTestId("flow-callee")).toContainText("block_backward");
    await flow.getByTestId("flow-node-grads").click();
    await expect(flow.getByTestId("flow-callee")).toHaveCount(0);
  });

  test("the examples equal the real functions", async ({ page }) => {
    await page.goto(FULL);
    const forward = modelCase("forward.tiny").outputs;
    await expectExact(page.getByTestId("model-x0"), forward.stages[0]);
    await expectExact(page.getByTestId("model-x-block-0"), forward.stages[1]);
    await expectExact(page.getByTestId("model-x-block-1"), forward.stages[2]);
    await expectExact(page.getByTestId("model-xf"), forward.Xf);
    await expectExact(page.getByTestId("model-logits"), forward.logits);
    const back = modelCase("loss_and_grads.tiny").outputs;
    await expectExact(page.getByTestId("model-loss"), back.loss);
    await expectExact(page.getByTestId("model-dz"), back.dZ);
    await expectExact(page.getByTestId("model-dxf"), back.dXf);
    for (const [i, value] of back.dX.entries()) {
      await expectExact(page.getByTestId(`model-dx-${i}`), value);
    }
  });

  test("the grads table has every key, and the numeric check agrees", async ({ page }) => {
    await page.goto(FULL);
    const { grads } = modelCase("loss_and_grads.tiny").outputs;
    const rows = page.getByTestId("model-grads").locator("tbody tr");
    await expect(rows).toHaveCount(Object.keys(grads).length);
    for (const key of Object.keys(grads)) {
      await expect(page.getByTestId("model-grads").locator(`tr[data-key='${key}']`)).toHaveCount(1);
    }
    const gaps = page.getByTestId("model-check").getByTestId("largest-gap");
    expect(await gaps.count()).toBe(9);
    for (const gap of await gaps.all()) {
      expect(Number(await gap.getAttribute("data-exact"))).toBeLessThan(1e-6);
    }
  });

  test("every parameter is on the page with its real shape", async ({ page }) => {
    await page.goto(FULL);
    const { params } = modelCase("model.config_shapes").outputs;
    const V = modelCase("model.config_shapes").inputs.V as number;
    const keys = page.getByTestId("parameter-keys");
    await keys.locator("summary").click();
    await expect(keys.locator("tr[data-key]")).toHaveCount(Object.keys(params).length);
    for (const [key, shape] of Object.entries(params) as [string, number[]][]) {
      const shown = shape.map((size) => (size === V ? "V" : size));
      const text = `(${shown.join(", ")}${shown.length === 1 ? "," : ""})`;
      await expect(keys.locator(`tr[data-key='${key}'] td`).nth(1)).toHaveText(text);
    }
    await expect(page.locator("[data-testid=parameter-table] tr[data-parameter]")).toHaveCount(22);
  });

  test("the softmax note links to the real places", async ({ page }) => {
    await page.goto(FULL);
    const main = page.locator("main");
    await expect(main).toContainText("It does not call softmax_forward");
    await expect(main).toContainText("sample_next calls softmax_forward");
  });

  test("the Block page and Attention link here and to each other", async ({ page }) => {
    await page.goto(BLOCK);
    await expect(
      page.locator("main").getByRole("link", { name: "Full Model", exact: true }),
    ).toHaveAttribute("href", FULL);
    await page.goto("/mathematics/attention/");
    await expect(
      page.locator("main").getByRole("link", { name: "Transformer Block", exact: true }),
    ).toHaveAttribute("href", BLOCK);
  });

  for (const [name, width, scheme] of [
    ["1280 light", 1280, "light"],
    ["1280 dark", 1280, "dark"],
    ["1000 light", 1000, "light"],
    ["390 light", 390, "light"],
    ["390 dark", 390, "dark"],
  ] as const) {
    test(`axe finds nothing and the page does not scroll sideways at ${name}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto(FULL);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
      const { violations } = await new AxeBuilder({ page }).analyze();
      expect(violations.map((item) => item.id)).toEqual([]);
    });
  }
});

test.describe("Transformer Block", () => {
  test("sections are in the required order", async ({ page }) => {
    await page.goto(BLOCK);
    await expect(page.locator("main").getByRole("heading", { level: 2 })).toHaveText([
      "Feed-forward network",
      "Feed-forward backward",
      "The pre-LN block",
      "Residual connections in the backward pass",
      "Caches",
      "Parameters of one block",
    ]);
  });

  test("the real config values are on the page", async ({ page }) => {
    await page.goto(BLOCK);
    const main = page.locator("main");
    await expect(main.locator('[data-config-value="ff_dim"]').first()).toHaveText("128");
    await expect(main.locator('[data-config-value="emb_dim"]').first()).toHaveText("32");
    await expect(main.locator('[data-config-value="num_layers"]').first()).toHaveText("2");
    await expect(page.getByTestId("tensor-shape-ffn-forward")).toContainText("(16, 128)");
    await expect(page.getByTestId("tensor-shape-block-forward")).toContainText("(16, 32)");
  });

  test("every line of each function equals the file, with its real number", async ({ page }) => {
    await page.goto(BLOCK);
    const blocks = page.locator("main figure:has(pre)");
    let checked = 0;
    for (const figure of await blocks.all()) {
      for (const line of await figure.locator("[data-line]").all()) {
        const number = Number(await line.getAttribute("data-line"));
        const text = await line.evaluate((element) => {
          const copy = element.cloneNode(true) as HTMLElement;
          copy.querySelector("[aria-hidden]")?.remove();
          return copy.textContent ?? "";
        });
        const file = await figure.locator("figcaption span").first().innerText();
        if (file !== "transformer.py") continue;
        expect(text.trimEnd(), `line ${number}`).toBe(realLines[number - 1]!.trimEnd());
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(40);
  });

  test("the feed-forward examples equal the real functions", async ({ page }) => {
    await page.goto(BLOCK);
    const forward = blockCase("ffn_forward.test_file").outputs;
    await expectExact(page.getByTestId("ffn-z"), forward.Z);
    await expectExact(page.getByTestId("ffn-a"), forward.A);
    await expectExact(page.getByTestId("ffn-out"), forward.out);
    const back = blockCase("ffn_backward.test_file").outputs;
    await expectExact(page.getByTestId("ffn-back-da"), back.dA);
    await expectExact(page.getByTestId("ffn-back-dz"), back.dZ);
    await expectExact(page.getByTestId("ffn-back-dxn"), back.dXn);
  });

  test("the block's backward example equals the real function", async ({ page }) => {
    await page.goto(BLOCK);
    const back = blockCase("block_backward.tiny").outputs;
    await expectExact(page.getByTestId("block-dxn2"), back.dXn2);
    await expectExact(page.getByTestId("block-dx1-branch"), back.dX1_branch);
    await expectExact(page.getByTestId("block-dx1"), back.dX1);
    await expectExact(page.getByTestId("block-dxn"), back.dXn);
    await expectExact(page.getByTestId("block-dx-branch"), back.dX_branch);
    await expectExact(page.getByTestId("block-dx"), back.dX);
    const off = blockCase("block_backward.branches_off");
    await expectExact(page.getByTestId("off-dx"), off.outputs.dX);
    await expectExact(page.getByTestId("off-dx"), off.inputs.dY);
    await expect(page.getByTestId("off-dx1-branch")).not.toContainText("-0");
  });

  test("each numeric check agrees with the backward", async ({ page }) => {
    await page.goto(BLOCK);
    for (const id of ["ffn-check", "block-check"]) {
      const gaps = page.getByTestId(id).getByTestId("largest-gap");
      expect(await gaps.count()).toBeGreaterThan(4);
      for (const gap of await gaps.all()) {
        expect(Number(await gap.getAttribute("data-exact"))).toBeLessThan(1e-6);
      }
    }
  });

  test.describe("the stepper", () => {
    const FIELDS: Record<string, string[]> = {
      Input: ["X"],
      "LayerNorm 1": ["X", "Xn"],
      Attention: ["Xn", "att"],
      "First add": ["X", "att", "X1"],
      "LayerNorm 2": ["X1", "Xn2"],
      "Feed-forward": ["Xn2", "ff"],
      "Second add": ["X1", "ff", "Y"],
    };
    const { inputs, outputs } = blockCase("block_forward.tiny");
    const all: Record<string, number[][]> = { X: inputs.X, ...outputs };

    test("every step and row shows the real vectors, and the line it names", async ({ page }) => {
      await page.goto(BLOCK);
      const stream = page.getByTestId("residual-stream");
      await hydrated(stream.getByTestId("stream-row-0"));
      for (const row of [0, 1, 2]) {
        await stream.getByTestId(`stream-row-${row}`).click();
        for (const [title, fields] of Object.entries(FIELDS)) {
          await stream.getByRole("button", { name: title, exact: false }).first().click();
          const panel = stream.locator("[data-testid^=stream-panel-]");
          await expectExact(
            panel,
            fields.map((field) => all[field]![row]!),
          );
          const vectors = stream.getByTestId("stream-vectors").locator("tr");
          await expect(vectors).toHaveCount(fields.length);
          const line = Number((await stream.getByTestId("stream-line").innerText()).split(":")[1]);
          const code = await stream.getByTestId("stream-code").innerText();
          expect(realLines[line - 1]!.trim().startsWith(code), title).toBe(true);
        }
      }
    });

    test("an add shows both terms and the sum", async ({ page }) => {
      await page.goto(BLOCK);
      const stream = page.getByTestId("residual-stream");
      await hydrated(stream.getByTestId("stream-row-0"));
      await stream.getByRole("button", { name: "First add" }).click();
      const rows = stream.getByTestId("stream-vectors").locator("tr");
      await expect(rows.nth(1)).toContainText("+");
      await expect(rows.nth(1)).toContainText("att[1]");
      await expect(rows.nth(2)).toContainText("=");
      await expect(rows.nth(2)).toContainText("X1[1]");
      await expect(stream.getByTestId("stream-code")).toHaveText("X1 = X + att");
    });

    test("it is labelled Illustrative Example", async ({ page }) => {
      await page.goto(BLOCK);
      const figure = page.locator("figure, section, div").filter({
        has: page.getByTestId("residual-stream"),
      });
      await expect(figure.last().locator("..")).toContainText("Illustrative Example");
    });
  });

  test.describe("the block diagram", () => {
    test("it follows block_backward, statement by statement", async ({ page }) => {
      await page.goto(BLOCK);
      await expectStatementsMatchFile(page.getByTestId("gradient-flow"), [
        'dXn2 = ffn_backward(dY, c["ff"], params, p, grads)',
        'dX1_branch, dg, db = layer_norm_backward(dXn2, c["ln2"], params[p + "ln2_g"])',
        "dX1 = dY + dX1_branch",
        'dXn = attention_backward(dX1, c["att"], params, p, cfg, grads)',
        'dX_branch, dg, db = layer_norm_backward(dXn, c["ln1"], params[p + "ln1_g"])',
        "return dX1 + dX_branch",
      ]);
    });

    test("selecting a step shows its function and marks the lines", async ({ page }) => {
      await page.goto(BLOCK);
      const flow = page.getByTestId("gradient-flow");
      const node = flow.getByTestId("flow-node-dXn");
      await hydrated(node);
      await node.click();
      await expect(node).toHaveAttribute("aria-pressed", "true");
      await expect(flow.getByTestId("flow-callee")).toContainText("attention_backward");
      const caller = flow.getByTestId("flow-caller");
      const lines = await highlighted(caller);
      expect(lines.length).toBe(1);
      expect(realLines[lines[0]! - 1]).toContain("attention_backward(dX1");

      const add = flow.getByTestId("flow-node-dX1");
      await add.click();
      await expect(flow.getByTestId("flow-callee")).toHaveCount(0);
      expect(realLines[(await highlighted(caller))[0]! - 1]).toContain("dX1 = dY + dX1_branch");
    });
  });

  test("the cache table shows the fields and readers", async ({ page }) => {
    await page.goto(BLOCK);
    const table = page.getByTestId("cache-table-block");
    const row = (name: string) => table.locator(`tr[data-cache='${name}']`);
    await expect(row('c["att"]').locator("[data-field]")).toHaveCount(6);
    await expect(row('c["ff"]').locator("[data-field]")).toHaveCount(3);
    await expect(row('c["ff"]')).toContainText("ffn_backward");
    await expect(row('c["att"]')).toContainText("attention_backward");
    await expect(row('c["ln1"]')).toContainText("layer_norm_backward");
  });

  for (const [name, width, scheme] of [
    ["1280 light", 1280, "light"],
    ["1280 dark", 1280, "dark"],
    ["1000 light", 1000, "light"],
    ["390 light", 390, "light"],
    ["390 dark", 390, "dark"],
  ] as const) {
    test(`axe finds nothing and the page does not scroll sideways at ${name}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto(BLOCK);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
      const { violations } = await new AxeBuilder({ page }).analyze();
      expect(violations.map((item) => item.id)).toEqual([]);
    });
  }
});

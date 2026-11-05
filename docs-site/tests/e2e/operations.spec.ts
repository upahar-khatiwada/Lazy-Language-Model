import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const LINEAR = "/mathematics/linear-relu-initialisation/";
const SOFTMAX = "/mathematics/softmax-cross-entropy/";
const LAYERNORM = "/mathematics/layernorm/";

type GoldenFile = {
  cases: { id: string; inputs: Record<string, unknown>; outputs: Record<string, unknown> }[];
};

const golden = JSON.parse(readFileSync("data/golden/ops.json", "utf-8")) as GoldenFile;
const goldenCase = (id: string) => golden.cases.find((item) => item.id === id)!;

type Nested = number | Nested[];
const TOLERANCE = 1e-9;

function expectClose(actual: Nested, expected: Nested, path = "value") {
  if (typeof expected === "number") {
    expect(typeof actual, path).toBe("number");
    expect(Math.abs((actual as number) - expected), `${path}: ${actual}, ${expected}`).toBeLessThan(
      TOLERANCE,
    );
    return;
  }
  expect(Array.isArray(actual), path).toBe(true);
  expect((actual as Nested[]).length, path).toBe(expected.length);
  expected.forEach((item, i) => expectClose((actual as Nested[])[i]!, item, `${path}[${i}]`));
}

// What the page computed, at full precision, against the real Python.
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

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

const cell = (page: Page, label: string, row: string, column: string) =>
  page.getByRole("spinbutton", { name: `${label}, row ${row}, column ${column}` });

test.describe("the matrix playground", () => {
  async function expectLinear(page: Page, id: string) {
    await expectExact(
      page.getByTestId("matmul-product"),
      goldenCase(`${id}.no_bias`).outputs.Y as Nested,
    );
    await expectExact(page.getByTestId("matmul-output"), goldenCase(id).outputs.Y as Nested);
    const backward = goldenCase(`${id}.backward`).outputs as Record<string, Nested>;
    await expectExact(page.getByTestId("linear-dw"), backward.dW!);
    await expectExact(page.getByTestId("linear-db"), [backward.db!]);
    await expectExact(page.getByTestId("linear-dx"), backward.dX!);
  }

  test("it starts at the golden values and is labelled", async ({ page }) => {
    await page.goto(LINEAR);
    await expectLinear(page, "linear.test_file");
    const figure = page.locator("figure").filter({ has: page.getByTestId("linear-dw") });
    await expect(figure.getByText("Illustrative Example")).toBeVisible();
  });

  test("editing X and dY gives the golden result", async ({ page }) => {
    await page.goto(LINEAR);
    const x = cell(page, "X", "t0", "k0");
    await hydrated(x);
    await x.fill("3");
    await cell(page, "dY", "t1", "j2").fill("2");
    await expectLinear(page, "linear.edited");
  });

  test("two columns of W give the golden result, and dY and b follow", async ({ page }) => {
    await page.goto(LINEAR);
    const columns = page.getByTestId("matmul-columns");
    await hydrated(columns);
    await columns.selectOption("2");
    await expect(page.getByTestId("matmul-rule")).toContainText("(2, 2) @ (2, 2) → (2, 2)");
    await expectLinear(page, "linear.two_columns");
  });
});

test.describe("the worked linear backward", () => {
  test("the table equals the real linear_backward", async ({ page }) => {
    await page.goto(LINEAR);
    const { dW, db, dX } = goldenCase("linear.test_file.backward").outputs as Record<
      string,
      Nested
    >;
    await expectExact(page.getByTestId("worked-dw"), dW!);
    await expectExact(page.getByTestId("worked-db"), db!);
    await expectExact(page.getByTestId("worked-dx"), dX!);
  });

  test("central differences agree with it, and the check is labelled test-only", async ({
    page,
  }) => {
    await page.goto(LINEAR);
    const table = page.getByTestId("linear-check");
    const { dW, db, dX } = goldenCase("gradcheck.linear").outputs as Record<string, Nested>;
    await expectExact(table.locator('tr[data-gradient="dW"] td:nth-child(3) code'), dW!);
    await expectExact(table.locator('tr[data-gradient="db"] td:nth-child(3) code'), [db!]);
    await expectExact(table.locator('tr[data-gradient="dX"] td:nth-child(3) code'), dX!);
    for (const gap of await page.getByTestId("linear-check").getByTestId("largest-gap").all()) {
      expect(Number(await gap.getAttribute("data-exact"))).toBeLessThan(1e-6);
    }
    const figure = page.locator("figure").filter({ has: table });
    await expect(figure).toContainText("A test-time idea, not part of the pipeline");
  });
});

test.describe("the note on inline linear layers", () => {
  test("it links to the seven forward lines and the backward lines", async ({ page }) => {
    await page.goto(LINEAR);
    const section = page.locator("main li").filter({ hasText: "Forward: always inline." });
    const links = section.locator("a.code-link");
    await expect(links).toHaveCount(7);
    for (const link of await links.all()) {
      await expect(link).toHaveText(/^transformer\.py:\d+$/);
      await expect(link).toHaveAttribute("href", /\/transformer\.py#L\d+$/);
    }
    await expect(
      page.locator("main li").filter({ hasText: "Backward: both." }).locator("a.code-link"),
    ).toHaveCount(5);
  });
});

test.describe("ReLU, Xavier and starting values", () => {
  test("ReLU equals the real functions", async ({ page }) => {
    await page.goto(LINEAR);
    await expectExact(
      page.getByTestId("relu-output"),
      goldenCase("relu_forward.test_file").outputs.Y as Nested,
    );
    await expectExact(
      page.getByTestId("relu-dz"),
      goldenCase("relu_backward.test_file").outputs.dZ as Nested,
    );
  });

  test("Xavier limits equal the local a of the real function", async ({ page }) => {
    await page.goto(LINEAR);
    const rows = page.getByTestId("xavier-example").locator("tr[data-case]");
    await expect(rows).toHaveCount(5);
    for (const row of await rows.all()) {
      const id = (await row.getAttribute("data-case"))!;
      await expectExact(row.getByTestId("xavier-limit"), goldenCase(id).outputs.a as Nested);
    }
    await expectExact(
      page.getByTestId("xavier-sample"),
      goldenCase("xavier.tiny").outputs.sample as Nested,
    );
  });

  test("every parameter is listed under its starting value", async ({ page }) => {
    await page.goto(LINEAR);
    const table = page.getByTestId("initial-values");
    await expect(table.locator('tr[data-init="ones"]')).toContainText("lnf_g");
    await expect(table.locator('tr[data-init="ones"]')).not.toContainText("lnf_b");
    await expect(table.locator('tr[data-init="zeros"]')).toContainText("head_b");
    await expect(table.locator('tr[data-init="xavier_uniform"]')).toContainText("tok_emb");
  });
});

test.describe("the softmax lab", () => {
  async function expectLab(page: Page, id: string) {
    const { loss, probs } = goldenCase(id).outputs as Record<string, Nested>;
    await expectExact(page.getByTestId("lab-loss"), loss!);
    await expectExact(page.getByTestId("lab-probs"), probs!);
    await expectExact(
      page.getByTestId("lab-gradient"),
      goldenCase(`${id}.backward`).outputs.dZ as Nested,
    );
  }

  test("it starts at the golden values and is labelled", async ({ page }) => {
    await page.goto(SOFTMAX);
    await expectLab(page, "cross_entropy.test_file");
    const figure = page.locator("figure").filter({ has: page.getByTestId("lab-loss") });
    await expect(figure.getByText("Illustrative Example")).toBeVisible();
  });

  test("the target selector gives the golden loss and gradient", async ({ page }) => {
    await page.goto(SOFTMAX);
    const target = page.getByTestId("lab-target");
    await hydrated(target);
    await target.selectOption("1");
    await expectLab(page, "cross_entropy.lab_target_1");
    await target.selectOption("2");
    await expectLab(page, "cross_entropy.lab_target_2");
  });

  test("editing a logit gives the golden result, and the bars follow", async ({ page }) => {
    await page.goto(SOFTMAX);
    const target = page.getByTestId("lab-target");
    await hydrated(target);
    await target.selectOption("1");
    await cell(page, "logits", "t0", "token 0").fill("3");
    await expectLab(page, "cross_entropy.lab_edited");
    const probs = goldenCase("cross_entropy.lab_edited").outputs.probs as number[][];
    await expect(page.getByTestId("bar-row").first()).toContainText(probs[0]![0]!.toFixed(4));
  });

  test("there is no temperature control", async ({ page }) => {
    await page.goto(SOFTMAX);
    const lab = page.locator("figure").filter({ has: page.getByTestId("lab-loss") });
    await expect(lab.getByText(/temperature/i)).toHaveCount(0);
  });
});

test.describe("the other softmax and loss examples", () => {
  test("softmax values equal the real function, and large scores overflow only unshifted", async ({
    page,
  }) => {
    await page.goto(SOFTMAX);
    const rows = page.getByTestId("softmax-shift").locator("tr[data-case]");
    await expect(rows).toHaveCount(4);
    for (const row of await rows.all()) {
      const id = (await row.getAttribute("data-case"))!;
      await expectExact(row.getByTestId("softmax-shift-value"), goldenCase(id).outputs.y as Nested);
    }
    await expect(
      page
        .locator('tr[data-case="softmax_forward.large_positive"]')
        .getByTestId("softmax-unshifted"),
    ).toHaveText("Infinity");
  });

  test("softmax backward equals the real function", async ({ page }) => {
    await page.goto(SOFTMAX);
    await expectExact(
      page.getByTestId("softmax-dx"),
      goldenCase("softmax_backward.test_file").outputs.dx as Nested,
    );
  });

  test("cross-entropy forward and backward equal the real functions", async ({ page }) => {
    await page.goto(SOFTMAX);
    const { loss, probs } = goldenCase("cross_entropy.basic").outputs as Record<string, Nested>;
    await expectExact(page.getByTestId("ce-loss"), loss!);
    await expectExact(page.getByTestId("ce-probs"), probs!);
    const { dZ } = goldenCase("cross_entropy.basic.backward").outputs as { dZ: number[][] };
    await expectExact(page.getByTestId("ce-dz"), dZ);
    await expectExact(
      page.getByTestId("ce-extreme-loss"),
      goldenCase("cross_entropy.extreme").outputs.loss as Nested,
    );
    await expect(page.getByTestId("ce-entry")).toContainText(
      `= ${String(Number(dZ[0]![0]!.toFixed(4)))}`,
    );
  });

  test("central differences agree with the cross-entropy backward", async ({ page }) => {
    await page.goto(SOFTMAX);
    const table = page.getByTestId("cross-entropy-check");
    await expectExact(
      table.locator('tr[data-gradient="dZ"] td:nth-child(3) code'),
      goldenCase("gradcheck.cross_entropy").outputs.dZ as Nested,
    );
    expect(Number(await table.getByTestId("largest-gap").getAttribute("data-exact"))).toBeLessThan(
      1e-6,
    );
  });

  test("perplexity equals the real function, and train.main's print line is quoted", async ({
    page,
  }) => {
    await page.goto(SOFTMAX);
    for (const id of ["perplexity.test_file", "perplexity.uniform_five", "perplexity.basic_loss"]) {
      await expectExact(
        page.locator(`tr[data-case="${id}"]`).getByTestId("perplexity-value"),
        goldenCase(id).outputs.perplexity as Nested,
      );
    }
    const text = (await page.locator("main").innerText()).replaceAll("\r", "");
    expect(text).toContain("ppl {perplexity(v):.1f}");
  });
});

test.describe("the LayerNorm lab", () => {
  async function expectLab(page: Page, id: string) {
    const { Y, Xhat, s, mu, var: variance } = goldenCase(id).outputs as Record<string, number[][]>;
    await expectExact(page.getByTestId("ln-output"), Y!);
    await expectExact(page.getByTestId("ln-xhat"), Xhat!);
    await expectExact(page.getByTestId("ln-s"), s![0]![0]!);
    await expectExact(page.getByTestId("ln-mu"), mu![0]![0]!);
    await expectExact(page.getByTestId("ln-var"), variance![0]![0]!);
  }

  test("it starts at the golden output and is labelled", async ({ page }) => {
    await page.goto(LAYERNORM);
    await expectLab(page, "layer_norm.test_file");
    const figure = page.locator("figure").filter({ has: page.getByTestId("ln-output") });
    await expect(figure.getByText("Illustrative Example")).toBeVisible();
    await expect(page.getByTestId("ln-status")).toHaveText("");
  });

  test("editing X, gamma and beta gives the golden output", async ({ page }) => {
    await page.goto(LAYERNORM);
    const first = cell(page, "X", "t0", "i0");
    await hydrated(first);
    await first.fill("2");
    await cell(page, "gamma", "gamma", "i2").fill("1");
    await cell(page, "beta", "beta", "i1").fill("1");
    await expectLab(page, "layer_norm.edited");
  });

  test("a row with no spread explains epsilon, and the output is beta", async ({ page }) => {
    await page.goto(LAYERNORM);
    const first = cell(page, "X", "t0", "i0");
    await hydrated(first);
    for (const column of ["i0", "i1", "i2", "i3"]) {
      await cell(page, "X", "t0", column).fill("4");
    }
    await expectLab(page, "layer_norm.constant_row");
    await expect(page.getByTestId("ln-status")).toContainText("Only epsilon keeps s finite");
  });
});

test.describe("the LayerNorm backward and its derivation", () => {
  test("the worked table equals the real layer_norm_backward", async ({ page }) => {
    await page.goto(LAYERNORM);
    const { dX, dgamma, dbeta } = goldenCase("layer_norm.test_file.backward").outputs as Record<
      string,
      Nested
    >;
    await expectExact(page.getByTestId("ln-dx"), dX!);
    await expectExact(page.getByTestId("ln-dgamma"), dgamma!);
    await expectExact(page.getByTestId("ln-dbeta"), dbeta!);
  });

  test("the derivation's result on the page equals the real function", async ({ page }) => {
    await page.goto(LAYERNORM);
    const real = goldenCase("layer_norm.test_file.backward").outputs.dX as Nested;
    await expectExact(page.getByTestId("ln-real-dx"), real);
    await expectExact(page.getByTestId("ln-derived-dx"), real);
    await expectExact(
      page.getByTestId("ln-jacobian-numeric"),
      goldenCase("gradcheck.layer_norm_jacobian").outputs.J as Nested,
    );
  });

  test("the concept is labelled Derived from implementation, with its PDF reference", async ({
    page,
  }) => {
    await page.goto(LAYERNORM);
    const card = page.getByRole("region", { name: "Concept: Deriving the backward formula" });
    await expect(card.getByText("Derived from implementation")).toBeVisible();
    const trace = page.getByTestId("trace").filter({ hasText: "layer_norm_forward" }).last();
    await expect(trace).toContainText("Not stated in the PDF");
  });

  test("central differences agree, and the check is labelled test-only", async ({ page }) => {
    await page.goto(LAYERNORM);
    const table = page.getByTestId("ln-check");
    const { dX, dgamma, dbeta } = goldenCase("gradcheck.layer_norm").outputs as Record<
      string,
      Nested
    >;
    await expectExact(table.locator('tr[data-gradient="dX"] td:nth-child(3) code'), dX!);
    await expectExact(table.locator('tr[data-gradient="dgamma"] td:nth-child(3) code'), [dgamma!]);
    await expectExact(table.locator('tr[data-gradient="dbeta"] td:nth-child(3) code'), [dbeta!]);
    for (const gap of await table.getByTestId("largest-gap").all()) {
      expect(Number(await gap.getAttribute("data-exact"))).toBeLessThan(1e-6);
    }
    const figure = page.locator("figure").filter({ has: table });
    await expect(figure).toContainText("A test-time idea, not part of the pipeline");
  });

  test("the equations' code table links to exact lines of ops.py", async ({ page }) => {
    await page.goto(LAYERNORM);
    const link = page
      .getByTestId("equation-code")
      .locator("tr")
      .filter({ hasText: "Xhat = (X - mu) * s" })
      .first()
      .locator("a.code-link");
    await expect(link).toHaveText(/^ops\.py:\d+$/);
    await expect(link).toHaveAttribute("href", /\/ops\.py#L\d+$/);
  });

  test("the epsilon note quotes the dataclass default", async ({ page }) => {
    await page.goto(LAYERNORM);
    const text = (await page.locator("main").innerText()).replaceAll("\r", "");
    expect(text).toContain("ln_eps: float = 1e-5");
    expect(text).toContain("nothing reads it");
  });

  test("only the LayerNorm parameters are listed, and each says how it starts", async ({
    page,
  }) => {
    await page.goto(LAYERNORM);
    const rows = page.getByTestId("parameter-table").locator("tr[data-parameter]");
    await expect(rows).toHaveCount(6);
    await expect(page.locator('tr[data-parameter="lnf_g"]')).toContainText("Starts as ones");
    await expect(page.locator('tr[data-parameter="lnf_b"]')).toContainText("Starts as zeros");
  });
});

for (const { path, name } of [
  { path: LINEAR, name: "Linear / ReLU / Initialisation" },
  { path: SOFTMAX, name: "Softmax & Cross-Entropy" },
  { path: LAYERNORM, name: "LayerNorm" },
]) {
  for (const width of [1280, 1000, 390]) {
    test.describe(`${name} at ${width}px`, () => {
      test.use({ viewport: { width, height: 900 } });

      test("no horizontal scroll", async ({ page }) => {
        await page.goto(path);
        expect(await horizontalOverflow(page)).toBe(0);
      });

      for (const theme of ["light", "dark"] as const) {
        test(`no axe findings in ${theme}`, async ({ page }) => {
          await page.emulateMedia({ colorScheme: theme });
          await page.goto(path);
          const { violations } = await new AxeBuilder({ page }).analyze();
          expect(violations.map((violation) => violation.id)).toEqual([]);
        });
      }
    });
  }
}

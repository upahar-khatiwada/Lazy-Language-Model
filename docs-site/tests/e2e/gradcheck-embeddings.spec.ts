import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const GRADCHECK = "/testing/gradient-checking/";
const EMBEDDINGS = "/mathematics/embeddings/";

type GoldenFile = { cases: { id: string; inputs: any; outputs: any }[] }; // eslint-disable-line @typescript-eslint/no-explicit-any

const load = (name: string) =>
  JSON.parse(readFileSync(`data/golden/${name}.json`, "utf-8")) as GoldenFile;
const embeddings = load("embeddings");
const gradcheck = load("gradcheck");
const embeddingCase = (id: string) => embeddings.cases.find((item) => item.id === id)!;
const gradcheckCase = (id: string) => gradcheck.cases.find((item) => item.id === id)!;

type Nested = number | Nested[];

function expectClose(actual: Nested, expected: Nested, path = "value") {
  if (typeof expected === "number") {
    expect(typeof actual, path).toBe("number");
    expect(Math.abs((actual as number) - expected), `${path}: ${actual}, ${expected}`).toBeLessThan(
      1e-9,
    );
    return;
  }
  expect((actual as Nested[]).length, path).toBe(expected.length);
  expected.forEach((item, i) => expectClose((actual as Nested[])[i]!, item, `${path}[${i}]`));
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

async function setInput(page: Page, name: string, value: string) {
  const input = page.getByRole("spinbutton", { name, exact: true });
  await hydrated(input);
  await input.fill(value);
}

test.describe("the Gradient Checking page", () => {
  test("it says it is test-only, and the sections are in order", async ({ page }) => {
    await page.goto(GRADCHECK);
    const main = page.locator("main");
    await expect(main).toContainText("This whole page is test-only mathematics.");
    await expect(main.getByRole("heading", { level: 2 })).toHaveText([
      "Central differences",
      "Relative error",
      "Checking every parameter",
      "What the tests assert",
      "Reading a failure",
    ]);
  });

  test("the lab starts from the golden case and follows h", async ({ page }) => {
    await page.goto(GRADCHECK);
    const start = gradcheckCase("numerical_grad.cube_default_h").outputs;
    await expectExact(page.getByTestId("cd-error-3"), start.error);
    await expectExact(page.getByTestId("cd-numeric-3"), start.g[0]);

    await expect(page.getByRole("spinbutton", { name: "h", exact: true })).toHaveValue("0.00001");
    await setInput(page, "h", "0.1");
    const large = gradcheckCase("numerical_grad.cube_large_h").outputs;
    await expectExact(page.getByTestId("cd-numeric-3"), large.g[0]);
    await expectExact(page.getByTestId("cd-error-3"), large.error);
    const square = gradcheckCase("numerical_grad.square_large_h").outputs;
    await expectExact(page.getByTestId("cd-error-2"), square.error);

    await setInput(page, "h", "1e-13");
    const tiny = gradcheckCase("numerical_grad.cube_tiny_h").outputs;
    await expectExact(page.getByTestId("cd-numeric-3"), tiny.g[0]);
  });

  test("h of 0 shows a message and no number", async ({ page }) => {
    await page.goto(GRADCHECK);
    await setInput(page, "h", "0");
    await expect(page.getByTestId("cd-status")).toContainText("h must be above 0");
    await expect(page.getByTestId("cd-numeric-3")).toContainText("not defined");
    await setInput(page, "h", "");
    await expect(page.getByTestId("cd-status")).toContainText("Enter a number");
    await setInput(page, "h", "0.1");
    await setInput(page, "x", "");
    await expect(page.getByTestId("cd-status")).toContainText("Enter a number");
    await setInput(page, "x", "1.5");
    const large = gradcheckCase("numerical_grad.cube_large_h").outputs;
    await expectExact(page.getByTestId("cd-numeric-3"), large.g[0]);
  });

  test("the relative errors equal the golden values", async ({ page }) => {
    await page.goto(GRADCHECK);
    for (const item of gradcheck.cases.filter(({ id }) => id.startsWith("relative_error."))) {
      await expectExact(page.getByTestId(`rel-${item.id}`), item.outputs.error);
    }
  });

  test("the report table marks exactly the expected cells", async ({ page }) => {
    await page.goto(GRADCHECK);
    const table = page.getByTestId("check-params");
    const { names } = gradcheckCase("check_params.correct").outputs as { names: string[] };
    await expect(table.locator("tbody tr")).toHaveCount(names.length);
    const over = table.locator('td[data-over="true"]');
    const wrong = gradcheckCase("check_params.block_output_wrong").outputs.worst as Record<
      string,
      number
    >;
    const expected =
      1 +
      Object.entries(wrong).filter(([name, value]) => !name.endsWith(".bk") && value >= 1e-4)
        .length;
    await expect(over).toHaveCount(expected);
    await expect(
      table
        .getByRole("row", { name: /^b1\.wq/ })
        .locator("td")
        .nth(1),
    ).toHaveAttribute("data-over", "true");
    await expect(
      table
        .getByRole("row", { name: /^b1\.wq/ })
        .locator("td")
        .nth(0),
    ).toHaveAttribute("data-over", "false");
  });

  test("the example is labelled illustrative", async ({ page }) => {
    await page.goto(GRADCHECK);
    await expect(page.locator("main").getByText("Actual Model Output")).toHaveCount(0);
    expect(await page.locator("main").getByText("Illustrative Example").count()).toBeGreaterThan(2);
  });
});

test.describe("the Embeddings page", () => {
  const setId = async (page: Page, t: number, id: number) => {
    const select = page.getByTestId(`lab-id-${t}`);
    await hydrated(select);
    await select.selectOption(String(id));
  };

  test("the worked examples equal the real functions", async ({ page }) => {
    await page.goto(EMBEDDINGS);
    const forward = embeddingCase("embed_forward.test_file").outputs;
    await expectExact(page.getByTestId("tokens"), forward.tokens);
    await expectExact(page.getByTestId("positions"), forward.positions);
    const backward = embeddingCase("embed_backward.test_file").outputs;
    await expectExact(page.getByTestId("dtok"), backward.tok);
    await expectExact(page.getByTestId("dpos"), backward.pos);
    await expectExact(page.getByTestId("overwritten"), backward.overwritten);
    await expect(page.getByTestId("dtok-entry")).toContainText("[1, 0] + [2, 2] = [3, 2]");
    // Ids are integers, and the page shows them that way.
    await expect(page.getByTestId("embed-backward")).toContainText("[2, 0, 2]");
    await expect(page.getByTestId("token-lookup")).toContainText("[2, 0, 2]");
  });

  test("the lab starts with the repeated id, and its gradient accumulates", async ({ page }) => {
    await page.goto(EMBEDDINGS);
    const forward = embeddingCase("embed_forward.test_file").outputs;
    const backward = embeddingCase("embed_backward.test_file").outputs;
    await expectExact(page.getByTestId("lab-x0"), forward.X0);
    await expectExact(page.getByTestId("lab-dtok"), backward.tok);
    await expectExact(page.getByTestId("lab-overwritten"), backward.overwritten);
    await expect(page.getByTestId("lab-repeat")).toContainText("Id 2 is at positions 0 and 2");
    await expect(page.getByTestId("lab-row-2")).toContainText("t = 0, t = 2");
    await expect(page.getByTestId("lab-row-1")).toContainText("not used");
  });

  test("picking ids follows the golden cases", async ({ page }) => {
    await page.goto(EMBEDDINGS);
    await setId(page, 0, 1);
    await setId(page, 1, 1);
    await setId(page, 2, 1);
    await expectExact(
      page.getByTestId("lab-dtok"),
      embeddingCase("embed_backward.all_same").outputs.tok,
    );
    await expect(page.getByTestId("lab-repeat")).toContainText(
      "Id 1 is at positions 0 and 1 and 2",
    );

    await setId(page, 0, 0);
    await setId(page, 2, 3);
    const noRepeatForward = embeddingCase("embed_forward.no_repeat").outputs;
    const noRepeat = embeddingCase("embed_backward.no_repeat").outputs;
    await expectExact(page.getByTestId("lab-x0"), noRepeatForward.X0);
    await expectExact(page.getByTestId("lab-dtok"), noRepeat.tok);
    await expectExact(page.getByTestId("lab-overwritten"), noRepeat.overwritten);
    await expect(page.getByTestId("lab-repeat")).toContainText("No id repeats");
  });

  test("a shorter length uses fewer rows, and the last position row gets nothing", async ({
    page,
  }) => {
    await page.goto(EMBEDDINGS);
    const length = page.getByTestId("lab-length");
    await hydrated(length);
    await length.selectOption("2");
    await expect(page.getByTestId("lab-id-2")).toHaveCount(0);
    await setId(page, 0, 3);
    await setId(page, 1, 3);
    await expectExact(
      page.getByTestId("lab-x0"),
      embeddingCase("embed_forward.two_positions").outputs.X0,
    );
    await setId(page, 0, 2);
    await setId(page, 1, 2);
    await expectExact(
      page.getByTestId("lab-dpos"),
      embeddingCase("embed_backward.short").outputs.pos,
    );
    await expectExact(
      page.getByTestId("lab-dtok"),
      embeddingCase("embed_backward.short").outputs.tok,
    );
  });

  test("shapes and row ranges show the values from config.py", async ({ page }) => {
    await page.goto(EMBEDDINGS);
    const config = JSON.parse(readFileSync("data/generated/config.json", "utf-8")) as {
      constants: { name: string; value: unknown }[];
    };
    const value = (name: string) =>
      config.constants.find((constant) => constant.name === name)!.value as number;
    const T = value("seq_len");
    const d = value("emb_dim");
    const Tmax = value("context_len");
    await expect(page.getByTestId("tensor-shape-embedding-sum")).toContainText(`(${T}, ${d})`);
    await expect(page.getByTestId("tensor-shape-position-slice")).toContainText(`(${Tmax}, ${d})`);
    await expect(page.getByTestId("unused-rows")).toHaveText(`${T} to ${Tmax - 1}`);
    await expect(page.locator(`[data-config-value="context_len"]`).first()).toHaveText(
      String(Tmax),
    );
    await expect(page.locator(`[data-config-value="seq_len"]`).first()).toHaveText(String(T));
  });

  test("the update table shows decay on the rows no sequence reaches", async ({ page }) => {
    await page.goto(EMBEDDINGS);
    const { T } = embeddingCase("position_update.no_decay").inputs as { T: number };
    await expect(page.getByTestId("position-row-0").locator("td")).toHaveText([
      "yes",
      "yes",
      "yes",
    ]);
    await expect(page.getByTestId(`position-row-${T}`).locator("td")).toHaveText([
      "no",
      "no",
      "yes",
    ]);
  });

  test("labels: illustrative, and no model output", async ({ page }) => {
    await page.goto(EMBEDDINGS);
    await expect(page.locator("main").getByText("Actual Model Output")).toHaveCount(0);
    expect(await page.locator("main").getByText("Illustrative Example").count()).toBeGreaterThan(3);
  });
});

for (const { width, theme } of [
  { width: 1280, theme: "light" },
  { width: 1280, theme: "dark" },
  { width: 1000, theme: "light" },
  { width: 390, theme: "light" },
  { width: 390, theme: "dark" },
] as const) {
  test.describe(`at ${width}px in ${theme}`, () => {
    test.use({ viewport: { width, height: 844 }, colorScheme: theme });

    for (const path of [GRADCHECK, EMBEDDINGS]) {
      test(`${path}: no axe findings and no sideways scroll`, async ({ page }) => {
        await page.goto(path);
        const { violations } = await new AxeBuilder({ page }).analyze();
        expect(violations.map(({ id, nodes }) => `${id}: ${nodes[0]?.html}`)).toEqual([]);
        expect(await horizontalOverflow(page)).toBeLessThanOrEqual(0);
      });
    }
  });
}

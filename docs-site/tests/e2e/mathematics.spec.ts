import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const NOTATION = "/mathematics/notation-and-shapes/";
const TOKENIZATION = "/mathematics/tokenization/";
const DATA = "/mathematics/data-pipeline/";

type GoldenFile = {
  cases: { id: string; inputs: Record<string, unknown>; outputs: Record<string, unknown> }[];
};

const golden = (file: string) =>
  JSON.parse(readFileSync(`data/golden/${file}.json`, "utf-8")) as GoldenFile;
const goldenCase = (file: string, id: string) => golden(file).cases.find((c) => c.id === id)!;

const config = JSON.parse(readFileSync("data/generated/config.json", "utf-8")) as {
  constants: { name: string; value: unknown }[];
};
const configValue = (name: string) =>
  config.constants.find((constant) => constant.name === name)!.value as number;

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

async function hydrated(page: Page, locator: Locator) {
  await expect
    .poll(() =>
      locator.evaluate((element) =>
        Object.keys(element).some((key) => key.startsWith("__reactProps")),
      ),
    )
    .toBe(true);
  await expect(page.locator("body")).toBeVisible();
}

const show = (value: number) => String(Number(value.toFixed(4)));
const cells = (rows: number[][]) => rows.flat().map(show);

test.describe("Notation & Shapes", () => {
  test("the symbol table shows the values in config.json", async ({ page }) => {
    await page.goto(NOTATION);
    const table = page.getByTestId("notation-table");
    for (const [dimension, name] of [
      ["T", "seq_len"],
      ["Tmax", "context_len"],
      ["d", "emb_dim"],
      ["H", "num_heads"],
      ["dh", "head_dim"],
      ["F", "ff_dim"],
      ["L", "num_layers"],
    ] as const) {
      const row = table.locator(`tr[data-dimension=${dimension}]`);
      await expect(row.getByTestId("notation-value")).toHaveText(String(configValue(name)));
      await expect(row.locator("code").first()).toHaveText(name);
    }
    await expect(table.locator("tr[data-dimension=V]").getByTestId("notation-value")).toHaveText(
      "depends on the corpus",
    );
  });

  test("the matrix example starts at the golden values and is labelled", async ({ page }) => {
    await page.goto(NOTATION);
    await expect(page.getByTestId("matmul-rule")).toContainText("(2, 3) @ (3, 2) → (2, 2)");
    await expect(page.getByTestId("matmul-product").locator("td")).toHaveText(
      cells(goldenCase("ops", "matmul.two_columns.no_bias").outputs.Y as number[][]),
    );
    await expect(page.getByTestId("matmul-output").locator("td")).toHaveText(
      cells(goldenCase("ops", "matmul.two_columns").outputs.Y as number[][]),
    );
    const figure = page.locator("figure").filter({ has: page.getByTestId("matmul-rule") });
    await expect(figure.getByText("Illustrative Example")).toBeVisible();
  });

  test("editing a cell gives the golden result, and three columns change the shape", async ({
    page,
  }) => {
    await page.goto(NOTATION);
    const cell = page.getByRole("spinbutton", { name: "X, row t0, column k0" });
    await hydrated(page, cell);

    await cell.fill("2");
    await expect(page.getByTestId("matmul-product").locator("td")).toHaveText(
      cells(goldenCase("ops", "matmul.edited_cell.no_bias").outputs.Y as number[][]),
    );
    await expect(page.getByTestId("matmul-output").locator("td")).toHaveText(
      cells(goldenCase("ops", "matmul.edited_cell").outputs.Y as number[][]),
    );
    await cell.fill("1");

    await page.getByTestId("matmul-columns").selectOption("3");
    await expect(page.getByTestId("matmul-rule")).toContainText("(2, 3) @ (3, 3) → (2, 3)");
    await expect(page.getByTestId("matmul-output").locator("td")).toHaveText(
      cells(goldenCase("ops", "matmul.three_columns").outputs.Y as number[][]),
    );
  });

  test("the formula column equals count_params in every row", async ({ page }) => {
    await page.goto(NOTATION);
    const rows = page.getByTestId("parameter-example").locator("tbody tr");
    const cases = golden("params").cases;
    await expect(rows).toHaveCount(cases.length);
    for (const item of cases) {
      const row = page.locator(`tr[data-case="${item.id}"]`);
      const expected = (item.outputs.count as number).toLocaleString("en-US");
      await expect(row.getByTestId("formula-count")).toHaveText(expected);
      await expect(row.getByTestId("real-count")).toHaveText(expected);
    }
  });

  test("the real configuration is written with V as a symbol", async ({ page }) => {
    await page.goto(NOTATION);
    const d = configValue("emb_dim");
    const F = configValue("ff_dim");
    const Tmax = configValue("context_len");
    const L = configValue("num_layers");
    const block = 4 * d * d + 2 * d * F + F + 9 * d;
    await expect(page.getByTestId("block-total")).toHaveText(block.toLocaleString("en-US"));
    const annotation = page.getByTestId("parameter-count").locator("annotation");
    await expect(annotation).toHaveCount(1);
    const constant = (Tmax + 2) * d + L * block;
    await expect(annotation).toHaveText(
      `N = ${2 * d + 1}\\,V + ${constant.toLocaleString("en-US").replace(",", "{,}")}`,
    );
  });

  test("the protocol steps stay out of the rail, and the concepts are in it", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto(NOTATION);
    const rail = page.getByRole("navigation", { name: "On this page" });
    await expect(rail.getByRole("link", { name: "Matrix multiplication" })).toBeVisible();
    await expect(rail.getByRole("link", { name: "What is it?" })).toHaveCount(0);
    await expect(page.locator("h3[data-concept-step]").first()).toHaveText("What is it?");
  });
});

type Merge = { pair: [string, string]; merged: string; frequency: number };

test.describe("Tokenization", () => {
  test("shows the number of merges from config.json", async ({ page }) => {
    await page.goto(TOKENIZATION);
    await expect(page.locator("[data-config-value=num_merges]")).toHaveText(
      String(configValue("num_merges")),
    );
  });

  test("the unknown-token table matches the real encode", async ({ page }) => {
    await page.goto(TOKENIZATION);
    const rows = page.getByTestId("encode-example").locator("tr[data-case]");
    await expect(rows).toHaveCount(5);
    for (const row of await rows.all()) {
      const id = (await row.getAttribute("data-case"))!;
      const { outputs } = goldenCase("bpe", id);
      await expect(row.getByTestId("encode-ids")).toHaveText(JSON.stringify(outputs.ids));
      await expect(row.getByTestId("encode-decoded")).toHaveText(outputs.decoded as string);
    }
  });

  test("the vocabulary has the unknown token first", async ({ page }) => {
    await page.goto(TOKENIZATION);
    const { itos } = goldenCase("bpe", "build_tokenizer.stepper").outputs as { itos: string[] };
    await expect(page.getByTestId("vocabulary").locator("tbody tr td:nth-child(2)")).toHaveText(
      itos.map((token) => token.replaceAll(" ", "␠")),
    );
  });

  test("the coverage table names every symbol, with links or reasons", async ({ page }) => {
    await page.goto(TOKENIZATION);
    const table = page.getByTestId("symbol-coverage");
    await expect(table.locator("tr[data-symbol]")).toHaveCount(12);
    const link = table.locator('tr[data-symbol="tokenizer.merge_tokens"] td:last-child a');
    await expect(link).toHaveAttribute("href", "#applying-a-merge");
    await expect(page.locator("#applying-a-merge")).toHaveCount(1);
    await expect(table.locator('tr[data-symbol="tokenizer.Tokenizer.__init__"]')).toContainText(
      "Skipped.",
    );
  });

  test("the concept reference links to this page", async ({ page }) => {
    await page.goto("/reference/mathematics/");
    const row = page.locator('tr[data-concept="tokenizer.train_bpe"]');
    await expect(row.getByRole("link", { name: "Explanation" })).toHaveAttribute(
      "href",
      /\/mathematics\/tokenization\/#concept-tokenizer-train-bpe$/,
    );
  });
});

const windowRows = (rows: number[][]) => rows.map((row) => row.join(", "));

test.describe("Data Pipeline", () => {
  async function checkWindows(page: Page, id: string) {
    const { outputs } = goldenCase("data", id);
    const table = page.getByTestId("window-table");
    await expect(table.locator("tbody tr td:nth-child(1)")).toHaveText(
      (outputs.starts as number[]).map(String),
    );
    await expect(table.locator("tbody tr td:nth-child(2)")).toHaveText(
      windowRows(outputs.inputs as number[][]),
    );
    await expect(table.locator("tbody tr td:nth-child(3)")).toHaveText(
      windowRows(outputs.targets as number[][]),
    );
    await expect(page.getByTestId("window-count")).toContainText(`= ${outputs.count}`);
  }

  test("the window table starts at the golden values and is labelled", async ({ page }) => {
    await page.goto(DATA);
    await checkWindows(page, "make_windows.test_file");
    await expect(page.getByTestId("window-shift")).toContainText("moved one place left");
    const figure = page.locator("figure").filter({ has: page.getByTestId("window-table") });
    await expect(figure.getByText("Illustrative Example")).toBeVisible();
  });

  test("editing stride and ids gives the real make_windows results", async ({ page }) => {
    await page.goto(DATA);
    const stride = page.getByTestId("window-stride");
    await hydrated(page, stride);
    await stride.fill("2");
    await checkWindows(page, "make_windows.stride_two");
    await stride.fill("3");
    await checkWindows(page, "make_windows.stride_three");
    await stride.fill("1");

    const one = goldenCase("data", "make_windows.one_window");
    await page.getByTestId("window-ids").fill((one.inputs.ids as number[]).join(","));
    await checkWindows(page, "make_windows.one_window");
    await expect(page.getByTestId("window-shift")).toContainText("moved one place left");
  });

  test("too few ids shows the message Python raises, and bad input is explained", async ({
    page,
  }) => {
    await page.goto(DATA);
    const ids = page.getByTestId("window-ids");
    await hydrated(page, ids);
    const failing = goldenCase("data", "make_windows.too_few");
    await ids.fill((failing.inputs.ids as number[]).join(", "));
    await expect(page.getByTestId("window-status")).toContainText(
      failing.outputs.message as string,
    );
    await expect(page.getByTestId("window-table")).toHaveCount(0);

    await ids.fill("1, two");
    await expect(page.getByTestId("window-status")).toContainText("whole numbers");
    await ids.fill("1, 2, 3, 4, 5, 6");
    await page.getByTestId("window-length").fill("0");
    await expect(page.getByTestId("window-status")).toContainText("at least 1");
  });

  test("the split, count and batch tables equal the Python results", async ({ page }) => {
    await page.goto(DATA);
    for (const item of golden("data").cases.filter(({ id }) => id.startsWith("split_ids."))) {
      const row = page.locator(`tr[data-case="${item.id}"]`);
      const out = item.outputs as Record<string, number>;
      await expect(row.getByTestId("split-cut")).toHaveText(out.cut!.toLocaleString("en-US"));
      await expect(row.getByTestId("split-train")).toHaveText(
        out.train_windows!.toLocaleString("en-US"),
      );
      await expect(row.getByTestId("split-val")).toHaveText(String(out.val_windows));
      await expect(row.getByTestId("split-crossing")).toHaveText(String(out.crossing));
    }
    const counted = golden("data").cases.filter(
      ({ id, outputs }) => id.startsWith("make_windows.") && outputs.count !== undefined,
    );
    await expect(page.getByTestId("count-example").locator("tbody tr")).toHaveCount(counted.length);
    for (const item of counted) {
      const row = page.locator(`tr[data-case="${item.id}"]`);
      await expect(row.getByTestId("count-formula")).toHaveText(String(item.outputs.count));
      await expect(row.getByTestId("count-real")).toHaveText(String(item.outputs.count));
    }
    const { outputs } = goldenCase("data", "iterate_batches.shuffled");
    await expect(page.getByTestId("batch-sizes")).toContainText(
      (outputs.sizes as number[]).join(", "),
    );
    await expect(page.getByTestId("batch-example").locator("tbody td code")).toHaveText(
      (outputs.epoch_starts as number[][][]).flat().map((batch) => batch.join(", ")),
    );
  });

  test("shapes use the real window length, and the sizes come from config.json", async ({
    page,
  }) => {
    await page.goto(DATA);
    const seqLen = configValue("seq_len");
    await expect(page.getByTestId("tensor-shape-inputs")).toContainText(`(W, ${seqLen})`);
    await expect(page.getByTestId("tensor-shape-batches")).toContainText(
      `(${configValue("batch_size")}, ${seqLen})`,
    );
  });

  test("the train.main lines appear with the split before the windows", async ({ page }) => {
    await page.goto(DATA);
    const text = (await page.locator("main").innerText()).replaceAll("\r", "");
    const split = text.indexOf("train_ids, val_ids = split_ids(ids, config.val_fraction)");
    const windows = text.indexOf("tr_x, tr_y = make_windows(train_ids");
    expect(split).toBeGreaterThan(0);
    expect(windows).toBeGreaterThan(split);
  });

  test("links to pages that do not exist yet are not links", async ({ page }) => {
    await page.goto(DATA);
    const main = page.locator("main");
    await expect(main.getByRole("link", { name: "Training Loop" })).toHaveCount(0);
    await expect(
      main.locator('[aria-disabled="true"]').filter({ hasText: "Training Loop" }),
    ).toBeVisible();
    await expect(main.getByRole("link", { name: "Softmax & Cross-Entropy" })).toBeVisible();
    await expect(main.getByRole("link", { name: "Tokenization" }).first()).toBeVisible();
  });
});

for (const { width, name } of [
  { width: 1280, name: "desktop" },
  { width: 390, name: "phone" },
]) {
  test.describe(`the BPE stepper on a ${name}`, () => {
    test.use({ viewport: { width, height: 844 } });

    test("every step equals the real train_bpe", async ({ page }) => {
      await page.goto(TOKENIZATION);
      const player = page.getByTestId("step-player");
      const next = player.getByRole("button", { name: "Next" });
      await hydrated(page, next);
      const figure = page.locator("figure").filter({ has: player });
      await expect(figure.getByText("Illustrative Example").first()).toBeVisible();
      await expect(figure).toContainText("These are not the merges of the real model");

      const { outputs } = goldenCase("bpe", "train_bpe_on_text.stepper");
      const merges = outputs.merges as Merge[];
      const splits = outputs.splits as [string, string[]][][];
      const splitCells = (index: number) =>
        splits[index]!.map(([, tokens]) => tokens.join("").replaceAll(" ", "␠"));

      await expect(player.getByTestId("step-status")).toContainText("Start");
      await expect(player.getByTestId("bpe-words").locator("tbody tr")).toHaveCount(4);

      for (const [i, merge] of merges.entries()) {
        await next.click();
        await expect(player.getByTestId("step-status")).toContainText(`Merge ${i + 1}`);
        const chosen = player.getByTestId("bpe-chosen");
        await expect(chosen).toContainText(`${merge.pair[0]} + ${merge.pair[1]}`);
        await expect(chosen).toContainText(`frequency ${merge.frequency}`);
        const selected = player.locator("tr[data-chosen]");
        await expect(selected).toHaveCount(1);
        await expect(selected.locator("code")).toHaveText(merge.pair);
        await expect(player.getByTestId("bpe-merge-list").locator("code")).toHaveText(
          merges.slice(0, i + 1).map(({ merged }) => merged),
        );
        await expect(
          player.getByTestId("bpe-words").locator("tbody tr td:nth-child(3)"),
        ).toHaveText(splitCells(i + 1));
        expect(await horizontalOverflow(page)).toBe(0);
      }

      await next.click();
      await expect(player.getByTestId("bpe-stop")).toContainText(`${merges.length} merges`);
      expect(await horizontalOverflow(page)).toBe(0);
    });

    test("the tie is explained on the first merge, and a new text starts over", async ({
      page,
    }) => {
      await page.goto(TOKENIZATION);
      const player = page.getByTestId("step-player");
      const next = player.getByRole("button", { name: "Next" });
      await hydrated(page, next);
      await next.click();
      await expect(player.getByTestId("bpe-chosen")).toContainText("strict >");
      await expect(player.getByTestId("bpe-pairs")).toContainText("Tied, but found later");

      const second = goldenCase("bpe", "train_bpe_on_text.test_file");
      await page.getByTestId("bpe-text").selectOption(second.inputs.text as string);
      await expect(player.getByTestId("step-status")).toContainText("Start");
      await next.click();
      const first = (second.outputs.merges as Merge[])[0]!;
      await expect(player.getByTestId("bpe-chosen")).toContainText(`frequency ${first.frequency}`);
    });
  });
}

for (const { path, name } of [
  { path: NOTATION, name: "Notation & Shapes" },
  { path: TOKENIZATION, name: "Tokenization" },
  { path: DATA, name: "Data Pipeline" },
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

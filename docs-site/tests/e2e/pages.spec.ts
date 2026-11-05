import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { CODE_REFS } from "../../data/code-refs";
import { PIPELINE } from "../../data/pipeline";

const HOW = "/getting-started/how-this-llm-works/";
const ARCHITECTURE = "/getting-started/architecture/";

const initParams = JSON.parse(readFileSync("data/derived/init-params.json", "utf-8")) as {
  parameters: { name: string }[];
};
const codeIndex = JSON.parse(readFileSync("data/generated/code-index.json", "utf-8")) as {
  files: { path: string; lineCount: number }[];
};
const source = (file: string) =>
  readFileSync(`../${file}`, "utf-8").replaceAll("\r\n", "\n").split("\n");

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

test("How This LLM Works quotes each step from the real lines, in code order", async ({ page }) => {
  await page.goto(HOW);
  const ids = Object.keys(CODE_REFS).filter((id) => id.startsWith("train."));
  const trainBlocks = page.locator("figure").filter({ hasText: "train.py" }).filter({
    hasText: "main",
  });
  await expect(trainBlocks).toHaveCount(ids.length);

  const lines = source("train.py");
  let previous = 0;
  for (const block of await trainBlocks.all()) {
    const link = await block.getByRole("link", { name: /GitHub/ }).getAttribute("href");
    const [, start] = /#L(\d+)/.exec(link!)!;
    expect(Number(start)).toBeGreaterThan(previous);
    previous = Number(start);

    // The gutter number and the text agree with the file on disk.
    for (const row of await block.locator("[data-line]").all()) {
      const number = Number(await row.getAttribute("data-line"));
      const text = await row.evaluate((element) =>
        [...element.children]
          .slice(1)
          .map((child) => child.textContent)
          .join(""),
      );
      expect(text.trimEnd()).toBe(lines[number - 1]!.trimEnd());
    }
  }
});

test.describe("Architecture", () => {
  test("every box is drawn, with the call or expression it stands for", async ({ page }) => {
    await page.goto(ARCHITECTURE);
    const diagram = page.getByTestId("pipeline-forward");
    const nodes = PIPELINE.filter((phase) => phase.forward).flatMap((phase) => phase.nodes);
    await expect(diagram.locator("[data-testid^=node-]")).toHaveCount(nodes.length);
    for (const node of nodes) {
      const code = node.inline ?? node.call!.split(".").at(-1)!;
      await expect(diagram.getByTestId(`node-${node.id}`).locator("code").first()).toHaveText(code);
    }
    await expect(diagram.locator("[data-phase=blocks]")).toContainText("num_layers");
  });

  test("the parameter names are those from init_params, in order", async ({ page }) => {
    await page.goto(ARCHITECTURE);
    const rows = page
      .getByTestId("parameter-table")
      .locator("tr[data-parameter] td:first-child code");
    await expect(rows).toHaveText(initParams.parameters.map(({ name }) => name));
  });

  test("the total is written with V as a symbol", async ({ page }) => {
    await page.goto(ARCHITECTURE);
    const count = page.getByTestId("parameter-count");
    await expect(count.getByText("Derived from implementation")).toBeVisible();
    await expect(count.locator("annotation").last()).toContainText("V +");
  });

  test("the repository map lists every real file, with its line count", async ({ page }) => {
    await page.goto(ARCHITECTURE);
    const map = page.getByTestId("repo-map");
    await expect(map.locator("tr[data-file]")).toHaveCount(codeIndex.files.length);
    for (const { path, lineCount } of codeIndex.files) {
      const row = map.locator(`tr[data-file="${path}"]`);
      await expect(row.locator("td").nth(1)).toHaveText(String(lineCount));
      await expect(row.getByRole("link")).toHaveAttribute("href", new RegExp(`${path}$`));
    }
  });
});

for (const width of [1280, 1000, 390]) {
  test.describe(`layout at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } });

    for (const path of [HOW, ARCHITECTURE]) {
      test(`${path}: no horizontal scroll`, async ({ page }) => {
        await page.goto(path);
        expect(await horizontalOverflow(page)).toBe(0);
      });

      for (const theme of ["light", "dark"] as const) {
        test(`${path}: no axe findings in ${theme}`, async ({ page }) => {
          await page.emulateMedia({ colorScheme: theme });
          await page.goto(path);
          const { violations } = await new AxeBuilder({ page }).analyze();
          expect(violations.map((violation) => violation.id)).toEqual([]);
        });
      }
    }
  });
}

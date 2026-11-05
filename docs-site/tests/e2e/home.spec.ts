import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { PIPELINE } from "../../data/pipeline";

const config = JSON.parse(readFileSync("data/generated/config.json", "utf-8")) as {
  constants: { name: string; value: unknown; valueText: string; derived: boolean }[];
};

const HERO =
  "This is a small language model built from scratch. This documentation shows the mathematics behind it, the code that implements the mathematics, and how the pieces work together.";

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

test("the exported home page renders its title and hero text", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "LLM Developer Docs" })).toBeVisible();
  await expect(page.getByText(HERO, { exact: true })).toBeVisible();
});

test("Start learning opens the first written Mathematics page", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Start learning" }).click();
  await expect(page).toHaveURL(/\/mathematics\/notation-and-shapes\/$/);
});

test("the reference page exists, so its question is a link", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "What is the equation?" }).click();
  await expect(page).toHaveURL(/\/reference\/mathematics\/$/);
});

test("Explore the code is disabled until the Code Explorer exists", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Explore the code" })).toBeDisabled();
});

test("the glance table shows the values in config.json", async ({ page }) => {
  await page.goto("/");
  const table = page.getByTestId("glance");
  for (const name of ["num_layers", "num_heads", "emb_dim", "head_dim", "ff_dim", "seq_len"]) {
    const constant = config.constants.find((candidate) => candidate.name === name)!;
    const value = constant.derived ? String(constant.value) : constant.valueText;
    await expect(table.locator(`tr[data-config="${name}"] td`).last()).toHaveText(value);
  }
});

test("the pipeline draws every node, and each links to real lines", async ({ page }) => {
  await page.goto("/");
  const diagram = page.getByTestId("pipeline-full");
  const nodes = PIPELINE.flatMap((phase) => phase.nodes);
  await expect(diagram.locator("[data-testid^=node-]")).toHaveCount(nodes.length);
  for (const node of nodes) {
    const link = diagram.getByTestId(`node-${node.id}`).locator("a.code-link");
    await expect(link).toHaveAttribute("href", /\/blob\/[0-9a-f]{40}\/[\w/.]+\.py#L\d+/);
  }
  // A written page is a link. A page that is not written yet is not.
  await expect(diagram.getByTestId("node-tokenizer").getByRole("link")).toHaveCount(2);
  await expect(diagram.getByTestId("node-attention").getByRole("link")).toHaveCount(2);
  await expect(diagram.getByTestId("node-clipping").getByRole("link")).toHaveCount(2);
  const generation = diagram.getByTestId("node-generation");
  await expect(generation.getByRole("link")).toHaveCount(1);
  await expect(generation.locator('[aria-disabled="true"]')).toContainText("Generation");
});

test("questions link to their pages or show as disabled", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("link", { name: "What is the equation?" })).toHaveAttribute(
    "href",
    "/reference/mathematics/",
  );
  await expect(page.getByRole("link", { name: "How does attention work?" })).toHaveAttribute(
    "href",
    "/mathematics/attention/",
  );
  await expect(page.getByRole("link", { name: "Where is it implemented?" })).toHaveCount(0);
  await expect(page.getByText("Where is it implemented?")).toBeVisible();
});

for (const width of [1280, 1000, 390]) {
  test.describe(`layout at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } });

    test("no horizontal scroll, no clipped boxes", async ({ page }) => {
      await page.goto("/");
      expect(await horizontalOverflow(page)).toBe(0);
      const clipped = await page
        .locator("[data-testid^=node-]")
        .evaluateAll((boxes) =>
          boxes.filter((box) => box.scrollWidth > box.clientWidth).map((box) => box.textContent),
        );
      expect(clipped).toEqual([]);
    });

    test("the sidebar is visible on desktop only", async ({ page }) => {
      await page.goto("/");
      const sidebar = page.getByRole("navigation", { name: "Documentation" });
      if (width >= 860) await expect(sidebar).toBeVisible();
      else await expect(sidebar).toBeHidden();
    });

    for (const theme of ["light", "dark"]) {
      test(`no axe findings in ${theme}`, async ({ page }) => {
        await page.emulateMedia({ colorScheme: theme as "light" | "dark" });
        await page.goto("/");
        const { violations } = await new AxeBuilder({ page }).analyze();
        expect(violations.map((violation) => violation.id)).toEqual([]);
      });
    }
  });
}

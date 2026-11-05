import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { clickWhenHydrated } from "./hydration";

const PATH = "/reference/mathematics/";

const concepts = readdirSync(join("data", "concepts")).filter((name) => name.endsWith(".yaml"));
const inventory = JSON.parse(readFileSync("data/derived/math-inventory.json", "utf-8")) as {
  rows: unknown[];
};

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

test("one row per concept file, and a computed coverage line", async ({ page }) => {
  await page.goto(PATH);
  await expect(page.locator("tr[data-concept]")).toHaveCount(concepts.length);
  await expect(page.getByTestId("coverage")).toHaveText(
    new RegExp(`^\\d+ of ${inventory.rows.length} operations documented$`),
  );
});

test("the row shows its equation and links to a code range", async ({ page }) => {
  await page.goto(PATH);
  const row = page.locator('tr[data-concept="ops.softmax_forward"]');
  await expect(row.locator(".katex").first()).toBeVisible();
  await expect(row.getByText("softmax_forward", { exact: true })).toBeVisible();
  await expect(row.locator("a.code-link")).toHaveAttribute(
    "href",
    /\/blob\/[0-9a-f]{40}\/ops\.py#L\d+-L\d+$/,
  );
  // No page explains this concept yet, so no link pretends one does.
  await expect(row.getByRole("link", { name: "Explanation" })).toHaveCount(0);
  await expect(row.getByText("Explanation not written")).toBeVisible();
});

test("filters show and hide rows", async ({ page }) => {
  await page.goto(PATH);
  const row = page.locator('tr[data-concept="ops.softmax_forward"]');
  const count = page.getByTestId("reference-count");
  const total = concepts.length;
  const category = page.getByTestId("reference-category");
  const direction = page.getByTestId("reference-direction");
  const text = page.getByTestId("reference-text");

  // React must be attached before changes to the controls count.
  await clickWhenHydrated(text);

  await category.selectOption("attention");
  await expect(row).toBeHidden();
  await category.selectOption("operations");
  await expect(row).toBeVisible();
  await category.selectOption("");

  await direction.selectOption("backward");
  await expect(row).toBeHidden();
  await direction.selectOption("forward");
  await expect(row).toBeVisible();
  await direction.selectOption("");

  await text.fill("softmax exp");
  await expect(row).toBeVisible();
  await text.fill("no such words");
  await expect(row).toBeHidden();
  await expect(count).toHaveText(`Showing 0 of ${total}`);
  await expect(page.getByText("No concept matches these filters.")).toBeVisible();
  await text.fill("");
  await expect(count).toHaveText(`Showing ${total} of ${total}`);
});

test.describe("desktop", () => {
  test.use({ viewport: { width: 1280, height: 900 } });

  test("a dense table", async ({ page }) => {
    await page.goto(PATH);
    const display = await page
      .getByTestId("reference-table")
      .evaluate((table) => getComputedStyle(table).display);
    expect(display).toBe("table");
    await expect(page.getByRole("columnheader", { name: "Tensor shapes" })).toBeVisible();
  });
});

test.describe("phone", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("cards that keep their labels, and a hidden row stays hidden", async ({ page }) => {
    await page.goto(PATH);
    const row = page.locator('tr[data-concept="ops.softmax_forward"]');
    expect(await row.evaluate((element) => getComputedStyle(element).display)).toBe("block");
    await expect(row.getByText("Tensor shapes")).toBeVisible();

    await clickWhenHydrated(page.getByTestId("reference-text"));
    await page.getByTestId("reference-category").selectOption("attention");
    await expect(row).toBeHidden();
  });
});

for (const width of [1280, 1000, 390]) {
  test.describe(`layout at ${width}px`, () => {
    test.use({ viewport: { width, height: 900 } });

    test("no horizontal scroll", async ({ page }) => {
      await page.goto(PATH);
      expect(await horizontalOverflow(page)).toBe(0);
    });

    for (const theme of ["light", "dark"] as const) {
      test(`no axe findings in ${theme}`, async ({ page }) => {
        await page.emulateMedia({ colorScheme: theme });
        await page.goto(PATH);
        const { violations } = await new AxeBuilder({ page }).analyze();
        expect(violations.map((violation) => violation.id)).toEqual([]);
      });
    }
  });
}

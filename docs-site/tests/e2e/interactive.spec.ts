import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { clickWhenHydrated } from "./hydration";

const KIT = "/design/interactive/";

async function openHydrated(page: Page) {
  await page.goto(KIT);
  const next = page.getByRole("button", { name: "Next" });
  await expect
    .poll(() =>
      next.evaluate((element) =>
        Object.keys(element).some((key) => key.startsWith("__reactProps")),
      ),
    )
    .toBe(true);
}

async function assertEveryPrimitiveVisible(page: Page) {
  for (const heading of [
    "MatrixInput and ShapeFlow",
    "StepPlayer and Heatmap",
    "Selected row and BarChart",
    "ProbabilityBars",
    "LineChart",
  ]) {
    await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
  }
  await expect(page.getByRole("spinbutton").first()).toBeVisible();
  await expect(page.getByTestId("shape-stage")).toHaveCount(7);
  await expect(page.getByTestId("step-player")).toBeVisible();
  await expect(page.getByTestId("heatmap-selected-row")).toBeVisible();
  await expect(page.getByTestId("bar-row").first()).toBeVisible();
  await expect(page.getByTestId("line-path")).toBeAttached();
  await expect(page.getByTestId("line-label")).toHaveCount(2);
  await expect(page.getByText("Illustrative Example").first()).toBeVisible();
  await expect(page.getByText("Actual Model Output")).toHaveCount(0);
}

// The first heatmap cell uses a --seq step, which differs between the themes.
const firstCellFill = (page: Page) =>
  page.evaluate(() => {
    const value = getComputedStyle(document.documentElement).getPropertyValue("--seq-0");
    return value.trim();
  });

for (const scheme of ["light", "dark"] as const) {
  test(`every primitive renders under a ${scheme} system preference`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto(KIT);
    await assertEveryPrimitiveVisible(page);
    expect(await firstCellFill(page)).toBe(scheme === "light" ? "#eef6f8" : "#1a2a33");
  });
}

test("every primitive renders after the theme toggle", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(KIT);
  await clickWhenHydrated(page.getByRole("button", { name: "Toggle theme" }));
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await assertEveryPrimitiveVisible(page);
  expect(await firstCellFill(page)).toBe("#1a2a33");
});

for (const { name, width, scheme } of [
  { name: "1280px light", width: 1280, scheme: "light" },
  { name: "1280px dark", width: 1280, scheme: "dark" },
  { name: "390px light", width: 390, scheme: "light" },
  { name: "390px dark", width: 390, scheme: "dark" },
] as const) {
  test(`axe finds nothing at ${name}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto(KIT);
    const { violations } = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"])
      .analyze();
    expect(
      violations.map((violation) => ({
        id: violation.id,
        nodes: violation.nodes.map((node) => node.target.join(" ")),
      })),
    ).toEqual([]);
  });
}

test.describe("at 390px", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("the page does not scroll sideways, on any step", async ({ page }) => {
    await openHydrated(page);
    const overflow = () =>
      page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
    expect(await overflow()).toBe(0);
    const next = page.getByRole("button", { name: "Next" });
    for (let i = 0; i < 4; i++) {
      await next.click();
      expect(await overflow()).toBe(0);
    }
  });

  test("all text stays at least 14px", async ({ page }) => {
    await page.goto(KIT);
    const small = await page.evaluate(() =>
      [...document.querySelectorAll<Element>("main text, main code, main label")]
        .filter((element) => parseFloat(getComputedStyle(element).fontSize) < 14)
        .map((element) => element.textContent),
    );
    expect(small).toEqual([]);
  });

  test("controls are at least 44px tall", async ({ page }) => {
    await page.goto(KIT);
    const small = await page.evaluate(() =>
      [
        ...document.querySelectorAll<Element>(
          "main input[type=number], main input[type=range], main select, main button, main summary, main label:has(input[type=radio])",
        ),
      ]
        .filter((element) => element.getBoundingClientRect().height < 44)
        .map((element) => element.outerHTML.slice(0, 80)),
    );
    expect(small).toEqual([]);
  });
});

test.describe("by keyboard", () => {
  const firstCell = (page: Page) =>
    page.getByRole("spinbutton", { name: "Xn, the normalized input, row t0, column d0" });

  test("editing a cell updates the results at once, without a submit", async ({ page }) => {
    await openHydrated(page);
    const before = await page.getByTestId("step-player").textContent();
    await firstCell(page).focus();
    await page.keyboard.press("Control+A");
    await page.keyboard.type("7");
    await expect(firstCell(page)).toHaveValue("7");
    await expect.poll(() => page.getByTestId("step-player").textContent()).not.toBe(before);
    await expect(page.getByRole("button", { name: /submit|apply/i })).toHaveCount(0);
  });

  test("an empty cell is marked and keeps the last valid value", async ({ page }) => {
    await openHydrated(page);
    const before = await page.getByTestId("step-player").textContent();
    await firstCell(page).focus();
    await page.keyboard.press("Control+A");
    await page.keyboard.press("Delete");
    await expect(firstCell(page)).toHaveAttribute("aria-invalid", "true");
    await expect(page.getByTestId("matrix-status")).toContainText("last valid value");
    expect(await page.getByTestId("step-player").textContent()).toBe(before);

    await page.keyboard.type("1");
    await expect(firstCell(page)).not.toHaveAttribute("aria-invalid", "true");
    await expect(page.getByTestId("matrix-status")).toHaveText("");
  });

  test("StepPlayer moves with Enter, stops at both ends and keeps focus", async ({ page }) => {
    await openHydrated(page);
    const status = page.getByTestId("step-status");
    const previous = page.getByRole("button", { name: "Previous" });
    const next = page.getByRole("button", { name: "Next" });
    await expect(status).toHaveText("Step 1 of 5: Scores");
    await expect(previous).toHaveAttribute("aria-disabled", "true");

    await next.focus();
    for (const title of ["Scale", "Mask", "Softmax", "Weighted values"]) {
      await page.keyboard.press("Enter");
      await expect(status).toContainText(title);
    }
    await expect(next).toHaveAttribute("aria-disabled", "true");
    await page.keyboard.press("Enter");
    await expect(status).toHaveText("Step 5 of 5: Weighted values");
    await expect(next).toBeFocused();

    const mask = page.getByRole("button", { name: "Mask" });
    await mask.focus();
    await page.keyboard.press("Enter");
    await expect(status).toHaveText("Step 3 of 5: Mask");
    await expect(mask).toHaveAttribute("aria-current", "step");
    await expect(page.getByTestId("step-player").getByText("-inf").first()).toBeVisible();
  });

  test("arrow keys choose the selected row", async ({ page }) => {
    await openHydrated(page);
    await page.getByRole("radio", { name: "t0" }).focus();
    await page.keyboard.press("ArrowDown");
    await expect(page.getByRole("radio", { name: "t1" })).toBeChecked();
    await expect(page.getByText("Row t1 of A")).toBeVisible();
  });

  test("sliders change the distribution, and temperature 0 is greedy", async ({ page }) => {
    await openHydrated(page);
    const chart = page.getByRole("img", { name: /Distribution given/ });
    await expect(chart).toHaveAttribute("aria-label", /removed/);
    await page.getByRole("slider", { name: "top_p" }).focus();
    await page.keyboard.press("End");
    await expect(chart).not.toHaveAttribute("aria-label", /removed/);

    await page.getByRole("slider", { name: "temperature" }).focus();
    await page.keyboard.press("Home");
    await expect(page.getByTestId("sampling-greedy")).toContainText("id 0");
  });
});

test.describe("motion", () => {
  const cellTransition = (page: Page) =>
    page
      .getByTestId("step-player")
      .locator("svg g rect")
      .first()
      .evaluate((element) => getComputedStyle(element).transitionDuration);

  test("transitions are off under prefers-reduced-motion", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(KIT);
    expect(await cellTransition(page)).toBe("0s");
  });

  test("transitions exist otherwise", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "no-preference" });
    await page.goto(KIT);
    expect(await cellTransition(page)).toBe("0.15s");
  });
});

test("the page is hidden from the navigation and marked noindex", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(KIT);
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute("content", /noindex/);
  const nav = page.getByRole("navigation", { name: "Documentation" });
  await expect(nav.getByRole("link", { name: /Interactive/ })).toHaveCount(0);
});

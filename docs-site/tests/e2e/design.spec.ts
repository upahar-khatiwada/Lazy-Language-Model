import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { clickWhenHydrated } from "./hydration";

const SPECIMENS = [
  "Typography",
  "Colour tokens",
  "Spacing, radius and shadow",
  "Badges",
  "Callouts",
  "Tables",
  "Code block",
  "Math block",
  "Sidebar item states",
  "Focus ring",
  "Buttons",
  "Search field",
];

async function axeViolations(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"])
    .analyze();
  return results.violations.map((violation) => ({
    id: violation.id,
    nodes: violation.nodes.map((node) => node.target.join(" ")),
  }));
}

test("/design shows every element in both themes", async ({ page }) => {
  await page.goto("/design/");
  for (const theme of ["light", "dark"]) {
    const panel = page.getByTestId(`panel-${theme}`);
    for (const title of SPECIMENS) {
      await expect(panel.getByRole("heading", { name: title, exact: true })).toBeVisible();
    }
    await expect(panel.getByText("Illustrative Example").first()).toBeVisible();
    await expect(panel.getByText("Actual Model Output").first()).toBeVisible();
    await expect(panel.getByText("Derived from implementation")).toBeVisible();
    await expect(panel.locator(".katex-display")).toBeVisible();
    await expect(panel.getByRole("button", { name: "Copy" })).toBeVisible();
    await expect(panel.getByRole("link", { name: /GitHub/ })).toHaveAttribute(
      "href",
      /\/blob\/[0-9a-f]{40}\/tests\/test_gradcheck\.py#L\d+-L\d+$/,
    );
    await expect(panel.getByLabel("Search documentation")).toBeVisible();
  }
});

test("each panel uses its own theme and the syntax colours resolve", async ({ page }) => {
  await page.goto("/design/");
  const background = (theme: string) =>
    page
      .getByTestId(`panel-${theme}`)
      .evaluate((element) => getComputedStyle(element).backgroundColor);
  expect(await background("light")).toBe("rgb(245, 247, 250)");
  expect(await background("dark")).toBe("rgb(15, 19, 26)");

  const colours = await page
    .getByTestId("panel-light")
    .locator("pre code span[style]")
    .evaluateAll((spans) => [...new Set(spans.map((span) => getComputedStyle(span).color))]);
  // Keyword, string, number, comment and function colours from the light tokens.
  for (const expected of [
    "rgb(161, 49, 95)",
    "rgb(37, 118, 74)",
    "rgb(163, 90, 0)",
    "rgb(97, 110, 133)",
    "rgb(11, 107, 124)",
  ]) {
    expect(colours).toContain(expected);
  }
});

for (const scheme of ["light", "dark"] as const) {
  test(`axe finds nothing on /design under a ${scheme} system preference`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto("/design/");
    expect(await axeViolations(page)).toEqual([]);
  });
}

test("the theme toggle overrides the system preference and persists", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/design/");
  const html = page.locator("html");
  await expect(html).not.toHaveAttribute("data-theme");

  await clickWhenHydrated(page.getByRole("button", { name: "Toggle theme" }));
  await expect(html).toHaveAttribute("data-theme", "dark");
  expect(await html.evaluate((element) => getComputedStyle(element).backgroundColor)).toBe(
    "rgb(15, 19, 26)",
  );
  expect(await axeViolations(page)).toEqual([]);

  await page.reload();
  await expect(html).toHaveAttribute("data-theme", "dark");

  await clickWhenHydrated(page.getByRole("button", { name: "Toggle theme" }));
  await expect(html).toHaveAttribute("data-theme", "light");
  expect(await axeViolations(page)).toEqual([]);
});

test("fonts are self-hosted and load without layout shift", async ({ page }) => {
  const fontRequests: string[] = [];
  page.on("request", (request) => {
    if (request.resourceType() === "font") fontRequests.push(request.url());
  });
  await page.addInitScript(() => {
    const store = window as unknown as { cls: number };
    store.cls = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as (PerformanceEntry & {
        value: number;
        hadRecentInput: boolean;
      })[]) {
        if (!entry.hadRecentInput) store.cls += entry.value;
      }
    }).observe({ type: "layout-shift", buffered: true });
  });

  await page.goto("/design/");
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(500);

  expect(fontRequests.length).toBeGreaterThan(0);
  for (const url of fontRequests) {
    expect(new URL(url).host).toBe("127.0.0.1:3000");
  }
  const family = await page
    .locator("body")
    .evaluate((element) => getComputedStyle(element).fontFamily);
  expect(family).toContain("IBM Plex Sans");
  expect(await document_fonts_has(page, "IBM Plex Sans")).toBe(true);
  expect(await document_fonts_has(page, "IBM Plex Mono")).toBe(true);

  const cls = await page.evaluate(() => (window as unknown as { cls: number }).cls);
  expect(cls).toBeLessThan(0.01);
});

async function document_fonts_has(page: Page, family: string): Promise<boolean> {
  return page.evaluate(
    (name) =>
      [...document.fonts].some((font) => font.family.includes(name) && font.status === "loaded"),
    family,
  );
}

test("the copy button copies the exact source", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/design/");
  const panel = page.getByTestId("panel-light");
  await clickWhenHydrated(panel.getByRole("button", { name: "Copy" }));
  await expect(panel.getByRole("button", { name: "Copied" })).toBeVisible();
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toContain("def test_key_bias_gradient_is_zero");
});

test("colour transitions take 150ms and stop under reduced motion", async ({ page }) => {
  const duration = () =>
    page
      .getByRole("button", { name: "Toggle theme" })
      .evaluate((element) => getComputedStyle(element).transitionDuration);

  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("/design/");
  expect(await duration()).toContain("0.15s");

  await page.emulateMedia({ reducedMotion: "reduce" });
  expect(await duration()).toBe("0s");
});

test("prose lines stay near 65 characters", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.goto("/design/");
  await page.evaluate(() => document.fonts.ready);

  const lengths = await page
    .getByTestId("panel-light")
    .getByTestId("body-sample")
    .locator("p")
    .evaluate((paragraph) => {
      const text = paragraph.firstChild as Text;
      const range = document.createRange();
      const counts = new Map<number, number>();
      for (let i = 0; i < text.length; i++) {
        range.setStart(text, i);
        range.setEnd(text, i + 1);
        const top = Math.round(range.getClientRects()[0]?.top ?? -1);
        counts.set(top, (counts.get(top) ?? 0) + 1);
      }
      return [...counts.values()];
    });

  // Every full line, leaving out the shorter last line.
  const fullLines = lengths.slice(0, -1);
  const mean = fullLines.reduce((sum, count) => sum + count, 0) / fullLines.length;
  console.log(`prose line lengths: ${lengths.join(", ")}; mean ${mean.toFixed(1)}`);
  expect(mean).toBeGreaterThanOrEqual(60);
  expect(mean).toBeLessThanOrEqual(72);
});

test("/design has no horizontal page scroll at 390px and is marked noindex", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/design/");
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute("content", /noindex/);
});

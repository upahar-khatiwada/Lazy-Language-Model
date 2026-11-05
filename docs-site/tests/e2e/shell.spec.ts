import { readdirSync, readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { NAV } from "../../data/nav";
import { clickWhenHydrated } from "./hydration";

const INTRO = "/getting-started/introduction/";
const TOKENIZATION = "/mathematics/tokenization/";

const WRITTEN = readdirSync("content", { recursive: true, encoding: "utf-8" }).filter(
  (path) => path.endsWith(".mdx") && !path.startsWith("_"),
).length;

const repoMeta = JSON.parse(readFileSync("data/generated/repo-meta.json", "utf-8")) as {
  sha: string;
  shortSha: string;
  githubBaseUrl: string;
};

const sidebar = (page: Page) => page.getByRole("navigation", { name: "Documentation" });
const rail = (page: Page) => page.getByRole("navigation", { name: "On this page" });

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
}

test.describe("desktop, 1280px", () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test("three columns with the requested widths", async ({ page }) => {
    await page.goto(INTRO);
    expect((await sidebar(page).boundingBox())?.width).toBe(250);
    expect((await rail(page).boundingBox())?.width).toBe(210);
    await expect(page.getByRole("button", { name: "Contents" })).toBeHidden();

    const main = page.locator("main");
    const content = await main.evaluate((element) => {
      const style = getComputedStyle(element);
      return (
        element.getBoundingClientRect().width -
        parseFloat(style.paddingLeft) -
        parseFloat(style.paddingRight)
      );
    });
    expect(content).toBeGreaterThanOrEqual(760);
    expect(content).toBeLessThanOrEqual(780);
    expect(await horizontalOverflow(page)).toBe(0);
  });

  test("content width stops at 780px on a very wide screen", async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 900 });
    await page.goto(INTRO);
    const width = await page.locator("main").evaluate((element) => {
      const style = getComputedStyle(element);
      return (
        element.getBoundingClientRect().width -
        parseFloat(style.paddingLeft) -
        parseFloat(style.paddingRight)
      );
    });
    expect(width).toBe(780);
  });

  test("top bar has the brand, search trigger, GitHub link and theme toggle", async ({ page }) => {
    await page.goto(INTRO);
    await expect(page.getByRole("link", { name: "LLM Developer Docs" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Search" })).toBeDisabled();
    await expect(page.getByRole("link", { name: "GitHub" })).toHaveAttribute(
      "href",
      repoMeta.githubBaseUrl,
    );
    await expect(page.getByRole("button", { name: "Toggle theme" })).toBeVisible();
  });

  test("sidebar shows the tree, source files, disabled pages and the active page", async ({
    page,
  }) => {
    await page.goto(TOKENIZATION);
    const nav = sidebar(page);

    const active = nav.locator('[aria-current="page"]');
    await expect(active).toHaveCount(1);
    await expect(active).toHaveAttribute("href", TOKENIZATION);
    await expect(active).toContainText("Tokenization");
    await expect(active).toContainText("tokenizer.py");
    expect(
      await active
        .locator("span")
        .last()
        .evaluate((el) => getComputedStyle(el).fontFamily),
    ).toContain("IBM Plex Mono");

    // Only written pages are links; all the others are disabled.
    await expect(nav.getByRole("link")).toHaveCount(WRITTEN);
    const disabled = nav.locator('[aria-disabled="true"]');
    await expect(disabled.first()).toContainText("Linear / ReLU / Initialisation");
    await expect(nav.getByText("gradcheck.py")).toBeVisible();
    await expect(nav.getByRole("link", { name: /design/i })).toHaveCount(0);
    expect(await disabled.count()).toBe(NAV.flatMap((group) => group.pages).length - WRITTEN);
  });

  test("breadcrumbs and previous/next links work and skip disabled pages", async ({ page }) => {
    await page.goto(INTRO);
    await expect(page.getByRole("navigation", { name: "Breadcrumb" })).toHaveText(
      /Docs.*Getting Started.*Introduction/,
    );
    const pager = page.getByRole("navigation", { name: "Previous and next pages" });
    await expect(pager.getByRole("link")).toHaveCount(1);
    await pager.getByRole("link", { name: /How This LLM Works/ }).click();
    await pager.getByRole("link", { name: /Architecture/ }).click();
    await expect(page).toHaveURL(/\/getting-started\/architecture\/$/);

    await pager.getByRole("link", { name: /Notation & Shapes/ }).click();
    await pager.getByRole("link", { name: /Tokenization/ }).click();
    await expect(page).toHaveURL(new RegExp(`${TOKENIZATION}$`));
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("Tokenization");
    await expect(page.getByRole("navigation", { name: "Breadcrumb" })).toHaveText(
      /Docs.*Mathematics.*Tokenization/,
    );
    await expect(sidebar(page).locator('[aria-current="page"]')).toContainText("Tokenization");

    await pager.getByRole("link", { name: /Notation & Shapes/ }).click();
    await expect(page).toHaveURL(/\/mathematics\/notation-and-shapes\/$/);

    // Linear / ReLU / Initialisation comes next after Data Pipeline, but it is not written, so the
    // pager skips every unwritten page up to the next written one.
    await page.goto("/mathematics/data-pipeline/");
    await expect(pager.getByRole("link", { name: /Linear/ })).toHaveCount(0);
    await pager.getByRole("link", { name: /Mathematics Reference/ }).click();
    await expect(page).toHaveURL(/\/reference\/mathematics\/$/);
  });

  test("the rail lists h2 and h3 headings and follows scrolling in both directions", async ({
    page,
  }) => {
    await page.goto(INTRO);
    const links = rail(page).getByRole("link");
    await expect(links).toHaveText([
      "First section",
      "First subsection",
      "Second subsection",
      "Second section",
      "Third section",
    ]);
    const current = rail(page).locator('[aria-current="location"]');
    await expect(current).toHaveText("First section");

    const scrollTo = (id: string) =>
      page.evaluate((target) => {
        const top = document.getElementById(target)!.getBoundingClientRect().top;
        window.scrollBy(0, top - 120);
      }, id);

    await scrollTo("second-section");
    await expect(current).toHaveText("Second section");
    await scrollTo("first-subsection");
    await expect(current).toHaveText("First subsection");
    // Scrolling up to just above a heading gives the section before it.
    await page.evaluate(() => {
      const top = document.getElementById("first-subsection")!.getBoundingClientRect().top;
      window.scrollBy(0, top - window.innerHeight * 0.5);
    });
    await expect(current).toHaveText("First section");

    await rail(page).getByRole("link", { name: "Third section" }).click();
    await expect(page).toHaveURL(/#third-section$/);
    await expect(current).toHaveText("Third section");
    await expect(page.locator("#third-section")).toBeInViewport();
  });

  test("the sidebar scrolls to keep the active page visible", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 180 });
    await page.goto(TOKENIZATION);
    const nav = sidebar(page);
    expect(await nav.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
    await expect(nav.locator('[aria-current="page"]')).toBeInViewport({ ratio: 1 });
    expect(await page.evaluate(() => window.scrollY)).toBe(0);
  });

  test("the footer shows the pinned commit from repo-meta.json", async ({ page }) => {
    await page.goto(INTRO);
    const footer = page.locator("footer");
    await expect(footer).toHaveText(`Generated from commit ${repoMeta.shortSha}`);
    await expect(footer.getByRole("link")).toHaveAttribute(
      "href",
      `${repoMeta.githubBaseUrl}/commit/${repoMeta.sha}`,
    );
  });

  test("tab order runs nav, content, rail, and every stop has a visible focus ring", async ({
    page,
  }) => {
    await page.goto(INTRO);
    const seen: string[] = [];
    for (let step = 0; step < 60; step++) {
      await page.keyboard.press("Tab");
      const stop = await page.evaluate(() => {
        const element = document.activeElement;
        if (!element || element === document.body) return null;
        const style = getComputedStyle(element);
        const region = element.closest('nav[aria-label="Documentation"]')
          ? "nav"
          : element.closest("main")
            ? "content"
            : element.closest('nav[aria-label="On this page"]')
              ? "rail"
              : element.closest("footer")
                ? "footer"
                : "top";
        return {
          region,
          label: element.textContent?.slice(0, 30),
          outline: style.outlineStyle !== "none" && parseFloat(style.outlineWidth) >= 2,
        };
      });
      if (!stop) break;
      expect(stop.outline, `${stop.region}: ${stop.label}`).toBe(true);
      if (seen.at(-1) !== stop.region) seen.push(stop.region);
    }
    expect(seen).toEqual(["top", "nav", "content", "rail", "footer"]);
  });

  test("/design also shows a visible focus ring at every tab stop", async ({ page }) => {
    await page.goto("/design/");
    let stops = 0;
    for (let step = 0; step < 80; step++) {
      await page.keyboard.press("Tab");
      const stop = await page.evaluate(() => {
        const element = document.activeElement;
        if (!element || element === document.body) return null;
        const style = getComputedStyle(element);
        return {
          label: `${element.tagName} ${element.textContent?.slice(0, 30)}`,
          outline: style.outlineStyle !== "none" && parseFloat(style.outlineWidth) >= 2,
        };
      });
      if (!stop) break;
      stops++;
      expect(stop.outline, stop.label).toBe(true);
    }
    expect(stops).toBeGreaterThan(20);
  });
});

test.describe("tablet, 1000px", () => {
  test.use({ viewport: { width: 1000, height: 800 } });

  test("sidebar and content, with the rail hidden", async ({ page }) => {
    await page.goto(INTRO);
    expect((await sidebar(page).boundingBox())?.width).toBe(250);
    await expect(page.locator('nav[aria-label="On this page"]')).toBeHidden();
    await expect(page.getByRole("button", { name: "Contents" })).toBeHidden();
    const main = await page.locator("main").boundingBox();
    expect(main?.x).toBe(250);
    expect(main?.width).toBe(750);
    expect(await horizontalOverflow(page)).toBe(0);
  });
});

test.describe("mobile, 390px", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("full-width content with a 16px gutter and no horizontal scroll", async ({ page }) => {
    for (const path of ["/", INTRO, TOKENIZATION]) {
      await page.goto(path);
      expect(await horizontalOverflow(page), path).toBe(0);
    }
    await page.goto(INTRO);
    const main = await page.locator("main").boundingBox();
    expect(main?.x).toBe(0);
    expect(main?.width).toBe(390);
    const heading = await page.getByRole("heading", { level: 1 }).boundingBox();
    expect(heading?.x).toBe(16);
    await expect(page.locator('nav[aria-label="On this page"]')).toBeHidden();
    await expect(page.locator('nav[aria-label="Documentation"]')).toBeHidden();
  });

  test("Contents opens the drawer and Escape closes it, returning focus", async ({ page }) => {
    await page.goto(INTRO);
    const button = page.getByRole("button", { name: "Contents" });
    await expect(button).toHaveAttribute("aria-expanded", "false");

    await clickWhenHydrated(button);
    await expect(button).toHaveAttribute("aria-expanded", "true");
    await expect(sidebar(page)).toBeVisible();
    await expect(sidebar(page)).toBeFocused();
    expect(await horizontalOverflow(page)).toBe(0);

    await page.keyboard.press("Escape");
    await expect(sidebar(page)).toBeHidden();
    await expect(button).toBeFocused();
    await expect(button).toHaveAttribute("aria-expanded", "false");
  });

  test("following a link closes the drawer", async ({ page }) => {
    await page.goto(INTRO);
    await clickWhenHydrated(page.getByRole("button", { name: "Contents" }));
    await sidebar(page)
      .getByRole("link", { name: /Tokenization/ })
      .click();
    await expect(page).toHaveURL(new RegExp(`${TOKENIZATION}$`));
    await expect(page.locator('nav[aria-label="Documentation"]')).toBeHidden();
  });

  test("tabbing out of the drawer closes it", async ({ page }) => {
    await page.goto(INTRO);
    await clickWhenHydrated(page.getByRole("button", { name: "Contents" }));
    // One stop per sidebar link, and only written pages are links.
    for (let stop = 0; stop < WRITTEN; stop++) await page.keyboard.press("Tab");
    await expect(sidebar(page)).toBeVisible();
    await page.keyboard.press("Tab");
    await expect(page.locator('nav[aria-label="Documentation"]')).toBeHidden();
  });

  test("controls are at least 44px tall", async ({ page }) => {
    await page.goto(INTRO);
    await clickWhenHydrated(page.getByRole("button", { name: "Contents" }));
    const small = await page.evaluate(() => {
      const selector = "header a, header button, nav a, footer a";
      return [...document.querySelectorAll<HTMLElement>(selector)]
        .filter((element) => {
          const box = element.getBoundingClientRect();
          return box.width > 0 && box.height < 44;
        })
        .map((element) => `${element.tagName} ${element.textContent}`);
    });
    expect(small).toEqual([]);
  });
});

test("layouts switch exactly at 860px and 1100px", async ({ page }) => {
  await page.goto(INTRO);
  const documentation = page.locator('nav[aria-label="Documentation"]');
  const onThisPage = page.locator('nav[aria-label="On this page"]');

  await page.setViewportSize({ width: 859, height: 800 });
  await expect(documentation).toBeHidden();
  await expect(page.getByRole("button", { name: "Contents" })).toBeVisible();

  await page.setViewportSize({ width: 860, height: 800 });
  await expect(documentation).toBeVisible();
  await expect(onThisPage).toBeHidden();
  await expect(page.getByRole("button", { name: "Contents" })).toBeHidden();
  expect(await horizontalOverflow(page)).toBe(0);

  await page.setViewportSize({ width: 1099, height: 800 });
  await expect(onThisPage).toBeHidden();

  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(onThisPage).toBeVisible();
  expect(await horizontalOverflow(page)).toBe(0);
});

for (const { name, width, scheme } of [
  { name: "1280px light", width: 1280, scheme: "light" },
  { name: "1000px dark", width: 1000, scheme: "dark" },
  { name: "390px dark", width: 390, scheme: "dark" },
] as const) {
  test(`axe finds nothing on a page with a source file at ${name}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto(TOKENIZATION);
    const { violations } = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"])
      .analyze();
    expect(violations.map((violation) => violation.id)).toEqual([]);
  });
}

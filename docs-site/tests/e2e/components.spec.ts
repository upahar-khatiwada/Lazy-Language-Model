import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { parse } from "yaml";
import { clickWhenHydrated } from "./hydration";

const SAMPLE = "/design/components/";

const repoMeta = JSON.parse(readFileSync("data/generated/repo-meta.json", "utf-8")) as {
  sha: string;
  githubBaseUrl: string;
};

const realLines = (file: string) =>
  execFileSync("git", ["show", `${repoMeta.sha}:${file}`], {
    encoding: "utf-8",
    maxBuffer: 1 << 24,
  }).split(/\r?\n/);

// The distinct code matches in data/links that point into a symbol.
const linkedMatches = (symbol: string) => {
  const files = readdirSync(join("data", "links")).filter((name) => name.endsWith(".yaml"));
  const links = files.flatMap(
    (name) =>
      (
        parse(readFileSync(join("data", "links", name), "utf-8")) as {
          links: { symbol: string; match: string }[];
        }
      ).links,
  );
  return [...new Set(links.filter((link) => link.symbol === symbol).map((link) => link.match))];
};

const codeBlock = (page: Page, name: string, index = 0) =>
  page.locator("figure", { has: page.getByText(name, { exact: true }) }).nth(index);

async function assertEveryComponentVisible(page: Page) {
  for (const title of [
    "ConceptCard",
    "SourceBadge",
    "Trace",
    "MathBlock",
    "Symbols",
    "MathCodeLink",
    "CodeReference",
    "NumericExample",
    "TensorShape",
    "PipelineDiagram",
    "Callout",
  ]) {
    await expect(page.getByRole("heading", { name: title, exact: true })).toBeVisible();
  }
  await expect(page.getByRole("region", { name: "Concept: Softmax" })).toBeVisible();
  await expect(page.getByTestId("source-badge")).toContainText("ops.py:22-24 softmax_forward");
  await expect(page.getByTestId("trace")).toContainText("Section 4.3, Eq. 23, p. 7");
  await expect(page.getByTestId("trace")).toContainText("generate.sample_next");
  await expect(page.getByTestId("trace")).toContainText("transformer.attention_forward");
  await expect(page.locator("#eq-softmax-shift .katex-display")).toBeVisible();
  await expect(page.getByRole("table", { name: /./ }).or(page.locator("table"))).toBeVisible();
  await expect(page.getByTestId("math-code-link")).toBeVisible();
  await expect(page.getByText("Illustrative Example")).toBeVisible();
  await expect(page.getByTestId("example-output")).toHaveText("[0.6652, 0.2447, 0.0900]");
  await expect(page.getByTestId("tensor-shape-attention")).toContainText("(2, 16, 16)");
  await expect(page.getByTestId("tensor-shape-sampling")).toContainText("(V,)");
  await expect(page.getByTestId("pipeline-softmax-in-attention").locator("li")).toHaveCount(4);
}

for (const scheme of ["light", "dark"] as const) {
  test(`every component renders under a ${scheme} system preference`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto(SAMPLE);
    await assertEveryComponentVisible(page);
    const expected = scheme === "light" ? "rgb(245, 247, 250)" : "rgb(15, 19, 26)";
    expect(await page.locator("html").evaluate((el) => getComputedStyle(el).backgroundColor)).toBe(
      expected,
    );
  });
}

test("every component renders after the theme toggle", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(SAMPLE);
  await clickWhenHydrated(page.getByRole("button", { name: "Toggle theme" }));
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await assertEveryComponentVisible(page);
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
    await page.goto(SAMPLE);
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

async function gutterAndText(block: Locator): Promise<[number, string][]> {
  return block.locator("[data-line]").evaluateAll((lines) =>
    lines.map((line) => {
      const number = line.querySelector('[aria-hidden="true"]')!.textContent!;
      return [Number(number), line.textContent!.slice(number.length)] as [number, string];
    }),
  );
}

test.describe("code references match the file at the pinned commit", () => {
  for (const { file, fn, index, first } of [
    { file: "ops.py", fn: "softmax_forward", index: 1, first: 22 },
    { file: "ops.py", fn: "cross_entropy_forward", index: 0, first: 29 },
    { file: "transformer.py", fn: "attention_forward", index: 0, first: 70 },
  ]) {
    test(`${file} ${fn}`, async ({ page }) => {
      await page.goto(SAMPLE);
      const block = codeBlock(page, fn, index);
      const shown = await gutterAndText(block);
      const real = realLines(file);

      expect(shown[0]?.[0]).toBe(first);
      expect(shown.length).toBeGreaterThan(1);
      for (const [number, text] of shown) expect(text).toBe(real[number - 1]);

      const last = shown.at(-1)![0];
      await expect(block.getByRole("link", { name: /GitHub/ })).toHaveAttribute(
        "href",
        `${repoMeta.githubBaseUrl}/blob/${repoMeta.sha}/${file}#L${first}-L${last}`,
      );
      await expect(block.getByRole("button", { name: "Copy" })).toBeVisible();
      await expect(block.getByText(file, { exact: true })).toBeVisible();
    });
  }

  test("highlighted lines come from links", async ({ page }) => {
    await page.goto(SAMPLE);
    const highlighted = (block: Locator) =>
      block
        .locator('[data-highlighted="true"]')
        .evaluateAll((lines) => lines.map((line) => Number(line.getAttribute("data-line"))));

    const ops = realLines("ops.py");
    const whole = await highlighted(codeBlock(page, "softmax_forward", 1));
    expect(whole.map((line) => ops[line - 1]?.trim())).toEqual([
      "e = np.exp(x - np.max(x, axis=-1, keepdims=True))",
      "return e / e.sum(axis=-1, keepdims=True)",
    ]);

    // MathCodeLink highlights only its own equation.
    const paired = await highlighted(page.getByTestId("math-code-link"));
    expect(paired.map((line) => ops[line - 1]?.trim())).toEqual([
      "return e / e.sum(axis=-1, keepdims=True)",
    ]);

    // Without an equation, every link into the function is highlighted, whichever concept owns it.
    const attention = await highlighted(codeBlock(page, "attention_forward"));
    const shown = attention.map((line) => realLines("transformer.py")[line - 1]!.trim());
    const declared = linkedMatches("transformer.attention_forward");
    expect(shown).toContain("A = softmax_forward(S)");
    // Several links can point at parts of one line, so this compares both ways, not the counts.
    for (const match of declared) {
      expect(shown.some((line) => line.includes(match))).toBe(true);
    }
    for (const line of shown) {
      expect(
        declared.some((match) => line.includes(match)),
        line,
      ).toBe(true);
    }
  });

  test("resolved line links carry the pinned sha", async ({ page }) => {
    await page.goto(SAMPLE);
    const hrefs = await page
      .locator("main a.code-link")
      .evaluateAll((links) => links.map((link) => link.getAttribute("href")!));
    expect(hrefs.length).toBeGreaterThanOrEqual(7);
    const transformer = realLines("transformer.py");
    for (const href of hrefs) {
      expect(href.startsWith(`${repoMeta.githubBaseUrl}/blob/${repoMeta.sha}/`), href).toBe(true);
    }
    const scores = hrefs.filter((href) => /transformer\.py#L\d+$/.test(href));
    const lines = scores.map((href) => transformer[Number(href.split("#L")[1]) - 1]?.trim());
    expect(lines).toContain("S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)");
    expect(lines).toContain("O = merge_heads(A @ Vh)");
  });
});

test("blocks in prose keep their vertical spacing", async ({ page }) => {
  await page.goto(SAMPLE);
  const first = await codeBlock(page, "softmax_forward", 1).boundingBox();
  const second = await codeBlock(page, "cross_entropy_forward").boundingBox();
  expect(second!.y - (first!.y + first!.height)).toBeGreaterThanOrEqual(16);

  const heading = await page.getByRole("heading", { name: "Trace", exact: true }).boundingBox();
  const trace = await page.getByTestId("trace").boundingBox();
  expect(trace!.y - (heading!.y + heading!.height)).toBeGreaterThanOrEqual(16);
});

test("the status badge in Trace keeps its own width", async ({ page }) => {
  await page.goto(SAMPLE);
  const trace = await page.getByTestId("trace").boundingBox();
  const badge = await page.getByTestId("trace").getByText("Implemented").boundingBox();
  expect(badge!.width).toBeLessThan(trace!.width / 2);
});

test("Copy LaTeX copies the equation from the data", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(SAMPLE);
  const block = page.locator("#eq-softmax-shift");
  await clickWhenHydrated(block.getByRole("button", { name: "Copy LaTeX" }));
  await expect(block.getByRole("button", { name: "Copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    String.raw`e_i = \exp\left(x_i - \max_k x_k\right)`,
  );
});

test("the code copy button copies the whole source", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto(SAMPLE);
  await clickWhenHydrated(
    codeBlock(page, "cross_entropy_forward").getByRole("button", { name: "Copy" }),
  );
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied.startsWith("def cross_entropy_forward(logits, targets):")).toBe(true);
  expect(copied).toContain("return float(loss), np.exp(logp)");
});

test("the equation anchor scrolls to its block", async ({ page }) => {
  await page.goto(SAMPLE);
  await page.locator('#eq-softmax-shift a[href="#eq-softmax-shift"]').click();
  await expect(page).toHaveURL(/#eq-softmax-shift$/);
  await expect(page.locator("#eq-softmax-shift")).toBeInViewport();
});

test("the page is hidden from the navigation and marked noindex", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(SAMPLE);
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute("content", /noindex/);
  const nav = page.getByRole("navigation", { name: "Documentation" });
  await expect(nav.getByRole("link", { name: /Component/ })).toHaveCount(0);
});

test.describe("desktop layout", () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test("MathCodeLink is side by side, and long lines scroll inside the block", async ({ page }) => {
    await page.goto(SAMPLE);
    const link = page.getByTestId("math-code-link");
    const math = await link.locator("figure").nth(0).boundingBox();
    const code = await link.locator("figure").nth(1).boundingBox();
    expect(code!.x).toBeGreaterThan(math!.x + math!.width - 1);

    const pre = link.locator("pre");
    expect(await pre.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    ).toBe(0);
  });
});

test.describe("mobile layout", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("no page scroll; equations and code scroll inside their own containers", async ({
    page,
  }) => {
    await page.goto(SAMPLE);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    ).toBe(0);

    const attention = codeBlock(page, "attention_forward").locator("pre");
    expect(await attention.evaluate((el) => getComputedStyle(el).overflowX)).toBe("auto");
    expect(await attention.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);

    const equation = page.locator('#eq-softmax-shift [role="group"]');
    expect(await equation.evaluate((el) => getComputedStyle(el).overflowX)).toBe("auto");
  });

  test("MathCodeLink stacks", async ({ page }) => {
    await page.goto(SAMPLE);
    const link = page.getByTestId("math-code-link");
    const math = await link.locator("figure").nth(0).boundingBox();
    const code = await link.locator("figure").nth(1).boundingBox();
    expect(code!.y).toBeGreaterThanOrEqual(math!.y + math!.height);
    expect(Math.round(code!.x)).toBe(Math.round(math!.x));
  });

  test("code links and figure links are at least 44px tall", async ({ page }) => {
    await page.goto(SAMPLE);
    const small = await page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>("main a.code-link, main figcaption a")]
        .filter((element) => element.getBoundingClientRect().height < 44)
        .map((element) => element.textContent),
    );
    expect(small).toEqual([]);
  });
});

test("no highlighting or KaTeX code ships to the browser", () => {
  // Markers taken from the libraries, checked first so that this test cannot pass by accident.
  const markers = [
    { text: "createHighlighter", library: "node_modules/shiki/dist/bundle-full.mjs" },
    { text: "KaTeX parse error", library: "node_modules/katex/dist/katex.js" },
  ];
  for (const { text, library } of markers) {
    expect(readFileSync(library, "utf-8"), library).toContain(text);
  }

  const dir = join("out", "_next", "static");
  const files = readdirSync(dir, { recursive: true, encoding: "utf-8" }).filter((file) =>
    file.endsWith(".js"),
  );
  expect(files.length).toBeGreaterThan(0);
  for (const file of files) {
    const source = readFileSync(join(dir, file), "utf-8");
    for (const { text } of markers) expect(source.includes(text), `${file}: ${text}`).toBe(false);
    expect(source.includes("vscode-textmate"), file).toBe(false);
  }
});

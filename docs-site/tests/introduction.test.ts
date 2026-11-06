import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import { PageDirectory } from "@/components/PageDirectory";
import { NAV } from "@/data/nav";
import { explorerFiles, explorerHref } from "@/lib/code-explorer";
import { listContentSlugs, readPageBody, readPageMeta } from "@/lib/content";
import { pageHref } from "@/lib/nav";
import { loadIndex, runSearch } from "@/lib/search-engine";
import { buildSearchIndexJson } from "@/lib/search-index";

const SLUG = "getting-started/introduction";
const body = readPageBody(SLUG);

describe("the Introduction page", () => {
  test("is published, so the placeholder notice is gone", () => {
    const meta = readPageMeta(SLUG);
    expect(meta.status).toBe("published");
    expect(meta.description).not.toMatch(/placeholder/i);
    expect(body).not.toContain("only to test page layout");
  });

  test("takes the model's numbers from generated data", () => {
    for (const name of ["num_layers", "num_heads", "emb_dim"]) {
      expect(body).toContain(`<ConfigValue name="${name}" />`);
    }
    expect(body).toContain("<TestTotals />");
    // Prose words only: stripping the components leaves no digit to go stale.
    const prose = body
      .replace(/^---[\s\S]*?---/, "")
      .replace(/<[^>]+>/g, "")
      .replace(/\(https?:[^)]+\)/g, "")
      .replace(/^\d+\. /gm, "");
    expect(prose).not.toMatch(/\d/);
  });

  test("credits both sources with plain links, and says what the documentation describes", () => {
    expect(body).toContain("(https://github.com/w3cj/how-llms-work)");
    expect(body).toContain("(https://youtu.be/YmLp8qe87A0)");
    expect(body).toContain("independent Python and NumPy implementation");
    expect(body).toContain("describes only this repository's code");
  });

  test("explains all three labels and links what it does not cover", () => {
    // Plain text: verify:content allows an Actual Model Output badge only on a panel with provenance.
    for (const label of [
      "Illustrative Example",
      "Actual Model Output",
      "Derived from implementation",
    ]) {
      expect(body).toContain(`**${label}**`);
    }
    expect(body).toContain('page="reference/not-in-this-model"');
    expect(body).toContain('page="reference/sources-discrepancies"');
  });
});

describe("the page list", () => {
  const html = renderToStaticMarkup(createElement(PageDirectory));
  const written = new Set(listContentSlugs());
  // Outside a build next/link drops the trailing slash; the built pages are checked by verify:site.
  const linksTo = (href: string) => html.includes(`href="${href.replace(/\/$/, "")}"`);

  test("links every written page and every source file", () => {
    for (const page of NAV.flatMap((group) => group.pages)) {
      if (written.has(page.slug)) expect(linksTo(pageHref(page.slug)), page.slug).toBe(true);
    }
    for (const { path } of explorerFiles()) {
      expect(linksTo(explorerHref(path)), path).toBe(true);
    }
    expect(html).toContain('href="/"');
  });

  test("shows an unwritten page as disabled text, not a link", () => {
    const unwritten = NAV.flatMap((group) => group.pages).filter((page) => !written.has(page.slug));
    expect(unwritten.length).toBeGreaterThan(0);
    for (const page of unwritten) {
      expect(linksTo(pageHref(page.slug)), page.slug).toBe(false);
      expect(html, page.slug).toContain(`aria-disabled="true"`);
      expect(html, page.slug).toContain(page.title.replace("&", "&amp;"));
    }
  });
});

describe("search", () => {
  test("finds the page and its headings", async () => {
    const { json } = await buildSearchIndexJson();
    const index = loadIndex(json);
    const hits = runSearch(index, "introduction").flat;
    expect(hits[0]).toMatchObject({ title: "Introduction", href: pageHref(SLUG) });
    expect(hits[0]!.detail).not.toMatch(/placeholder/i);

    const credits = runSearch(index, "credits").flat;
    expect(credits.map((hit) => hit.href)).toContain(`${pageHref(SLUG)}#credits`);
    expect(json).not.toContain("Placeholder page used to test");
  }, 60_000);
});

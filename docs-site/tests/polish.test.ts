import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { pageMetadata, SITE_NAME } from "@/lib/metadata";

const tsxFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return tsxFiles(path);
    return entry.name.endsWith(".tsx") ? [path] : [];
  });

const scripts = Object.keys(
  (JSON.parse(readFileSync("package.json", "utf-8")) as { scripts: Record<string, string> })
    .scripts,
);

describe("metadata", () => {
  test("a page's title and description reach link previews", () => {
    const meta = pageMetadata("Attention", "About attention.");
    expect(meta.title).toBe("Attention");
    expect(meta.description).toBe("About attention.");
    expect(meta.openGraph).toMatchObject({
      title: "Attention",
      description: "About attention.",
      siteName: SITE_NAME,
    });
    expect(meta.twitter).toMatchObject({
      card: "summary",
      title: "Attention",
      description: "About attention.",
    });
  });

  test("the home page does not repeat the site name in its title", () => {
    expect(readFileSync("app/(docs)/page.tsx", "utf-8")).toContain(
      "title: { absolute: SITE_NAME }",
    );
  });

  test("the app has a favicon and a 404 page that is not indexed", () => {
    expect(readFileSync("app/icon.svg", "utf-8")).toContain("<svg");
    const notFound = readFileSync("app/not-found.tsx", "utf-8");
    expect(notFound).toContain("robots: { index: false }");
    expect(notFound).toContain("DocMain");
  });
});

describe("performance", () => {
  // Automatic prefetch cost 60 requests on every page, and 20 of them were 404s: the static export
  // does not write the segment files it asks for under a route group.
  test.each([...tsxFiles("components"), ...tsxFiles("app")])("%s prefetches no link", (file) => {
    const tags = readFileSync(file, "utf-8").match(/<Link\b[^<]*?>/gs) ?? [];
    for (const tag of tags) expect(tag).toContain("prefetch=");
  });

  test("the search index and engine load only on first open", () => {
    const dialog = readFileSync("components/SearchDialog.tsx", "utf-8");
    expect(dialog).toContain('import("@/lib/search-engine")');
    expect(dialog).not.toMatch(/^import .* from "minisearch"/m);
    expect(dialog).not.toMatch(/^import .* from "@\/lib\/search-engine"/m);
  });
});

describe("maintenance documents", () => {
  const readme = readFileSync("README.md", "utf-8");
  const maintenance = readFileSync("MAINTENANCE.md", "utf-8");

  test.each([
    ["README.md", readme],
    ["MAINTENANCE.md", maintenance],
  ])("every command in %s exists", (_name, text) => {
    const named = [...text.matchAll(/bun run ([a-z0-9:]+)/g)].map((match) => match[1]!);
    expect(named.length).toBeGreaterThan(5);
    for (const name of named) expect(scripts).toContain(name);
  });

  test("the README lists every script", () => {
    for (const script of scripts) expect(readme, script).toContain(`bun run ${script}`);
  });

  test("the README covers clone to running site, data flow, a concept and a page", () => {
    for (const heading of [
      "## From clone to running site",
      "## Data flow",
      "## How to add a concept",
      "## How to add a page",
    ]) {
      expect(readme).toContain(heading);
    }
    expect(readme).toContain("git clone ");
  });

  test("the maintenance guide covers the order after a code change, and the deployment", () => {
    const order = [
      "bun run sync:repo",
      "bun run sync:golden",
      "bun run verify:generated",
      "bun run verify:content",
      "bun run build",
    ].map((command) => maintenance.indexOf(command, maintenance.indexOf("### 1. Regenerate")));
    expect(order.every((at) => at >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));

    expect(maintenance).toContain("NEXT_PUBLIC_BASE_PATH");
    expect(maintenance).toContain("actions/deploy-pages");
    expect(maintenance).toContain("path: docs-site/out");
    expect(maintenance).toContain(".nojekyll");
  });

  test("the workflow in the guide runs commands that exist", () => {
    const workflow = maintenance.slice(maintenance.indexOf("```yaml"));
    const named = [...workflow.matchAll(/run: bun run ([a-z0-9:]+)/g)].map((match) => match[1]!);
    expect(named).toContain("build");
    expect(named).toContain("verify:site");
    for (const name of named) expect(scripts).toContain(name);
  });
});

import { describe, expect, test } from "vitest";
import { NAV } from "@/data/nav";
import { listContentSlugs, loadPage, readPageMeta } from "@/lib/content";
import { breadcrumbs, flattenNav, prevNext, resolveNav } from "@/lib/nav";
import { loadCodeIndex } from "@/lib/repo";

const sample = resolveNav(
  [
    {
      title: "A",
      pages: [
        { title: "One", slug: "a/one" },
        { title: "Two", slug: "a/two" },
      ],
    },
    {
      title: "B",
      pages: [
        { title: "Three", slug: "b/three" },
        { title: "Four", slug: "b/four" },
      ],
    },
  ],
  ["a/one", "b/three", "b/four"],
);

describe("navigation helpers", () => {
  test("a page is enabled only when its content exists", () => {
    expect(flattenNav(sample).map((page) => page.enabled)).toEqual([true, false, true, true]);
    expect(flattenNav(sample)[0]?.href).toBe("/a/one/");
  });

  test("previous and next follow nav order and skip disabled pages", () => {
    expect(prevNext(sample, "a/one").prev).toBeUndefined();
    expect(prevNext(sample, "a/one").next?.slug).toBe("b/three");
    expect(prevNext(sample, "b/three").prev?.slug).toBe("a/one");
    expect(prevNext(sample, "b/four").next).toBeUndefined();
  });

  test("breadcrumbs name the group and the page", () => {
    expect(breadcrumbs(sample, "b/three")).toEqual([
      { title: "Docs", href: "/" },
      { title: "B" },
      { title: "Three" },
    ]);
  });

  test("unknown pages fail loudly", () => {
    expect(() => breadcrumbs(sample, "missing")).toThrow();
    expect(() => prevNext(sample, "a/two")).toThrow();
  });
});

describe("the real navigation tree", () => {
  const pages = NAV.flatMap((group) => group.pages);

  test("slugs are unique and /design is not listed", () => {
    expect(new Set(pages.map((page) => page.slug)).size).toBe(pages.length);
    expect(pages.some((page) => page.slug.includes("design"))).toBe(false);
  });

  test("Gradient Checking is outside the Mathematics pipeline (decision D2)", () => {
    const mathematics = NAV.find((group) => group.title === "Mathematics")!;
    expect(mathematics.pages.some((page) => page.title === "Gradient Checking")).toBe(false);
    const testing = NAV.find((group) => group.title === "Testing")!;
    expect(testing.pages.map((page) => page.title)).toEqual(["Gradient Checking"]);
  });

  test("every source file exists in the generated code index", () => {
    const paths = new Set(loadCodeIndex().files.map((file) => file.path));
    for (const page of pages) {
      if (page.sourceFile) expect(paths, page.title).toContain(page.sourceFile);
    }
  });

  test("every content file is listed in the navigation", () => {
    const slugs = new Set(pages.map((page) => page.slug));
    for (const slug of listContentSlugs()) expect(slugs, slug).toContain(slug);
  });

  test("frontmatter agrees with the navigation", () => {
    for (const slug of listContentSlugs()) {
      const page = pages.find((candidate) => candidate.slug === slug)!;
      const meta = readPageMeta(slug);
      expect(meta.title, slug).toBe(page.title);
      expect(meta.sourceFile, slug).toBe(page.sourceFile);
    }
  });
});

test("MDX pages compile and report unique h2 and h3 headings", async () => {
  const { headings, Content } = await loadPage("getting-started/introduction");
  expect(typeof Content).toBe("function");
  expect(headings.map((heading) => [heading.depth, heading.id])).toEqual([
    [2, "first-section"],
    [3, "first-subsection"],
    [3, "second-subsection"],
    [2, "second-section"],
    [2, "third-section"],
  ]);
});

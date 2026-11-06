// @vitest-environment happy-dom

import { describe, expect, test } from "vitest";
import {
  auditAccessibility,
  auditMetadata,
  auditPermalink,
  auditSkipLink,
  collectIds,
  collectLinks,
  parsePermalink,
  resolveInternal,
} from "@/lib/site-audit";

const doc = (body: string, head = "") =>
  new DOMParser().parseFromString(
    `<!doctype html><html lang="en"><head>${head}</head><body>${body}</body></html>`,
    "text/html",
  );

const GOOD = `<a href="#main-content">Skip</a><main id="main-content"><h1>T</h1><h2>A</h2><h3>B</h3><h2>C</h2></main>`;

describe("auditAccessibility", () => {
  test("a good page has no problems", () => {
    expect(auditAccessibility(doc(GOOD))).toEqual([]);
    expect(auditSkipLink(doc(GOOD))).toEqual([]);
  });

  test.each([
    ["no main", `<h1>T</h1>`, "exactly one <main>"],
    ["two mains", `<main><h1>T</h1></main><main></main>`, "exactly one <main>"],
    ["no h1", `<main><h2>A</h2></main>`, "exactly one h1"],
    ["two h1", `<main><h1>A</h1><h1>B</h1></main>`, "exactly one h1"],
    ["skipped level", `<main><h1>A</h1><h3>B</h3></main>`, "heading skips a level"],
    ["empty heading", `<main><h1>A</h1><h2></h2></main>`, "empty h2"],
    ["duplicate id", `<main id="x"><h1>A</h1><p id="x"></p></main>`, 'duplicate id "x"'],
    ["bad aria target", `<main><h1>A</h1><p aria-describedby="n"></p></main>`, 'missing id "n"'],
    ["empty button", `<main><h1>A</h1><button></button></main>`, "<button> has no accessible name"],
    ["empty link", `<main><h1>A</h1><a href="/x"></a></main>`, "<a> has no accessible name"],
    ["unlabelled input", `<main><h1>A</h1><input></main>`, "<input> has no label"],
    ["unlabelled select", `<main><h1>A</h1><select></select></main>`, "<select> has no label"],
    ["positive tabindex", `<main><h1>A</h1><p tabindex="2"></p></main>`, "positive tabindex"],
    [
      "focusable in hidden",
      `<main><h1>A</h1><div aria-hidden="true"><button>x</button></div></main>`,
      "focusable element inside aria-hidden",
    ],
    ["unnamed svg", `<main><h1>A</h1><svg></svg></main>`, "<svg> has neither"],
    [
      "role img without label",
      `<main><h1>A</h1><svg role="img"></svg></main>`,
      "<svg> has neither",
    ],
    ["image without alt", `<main><h1>A</h1><img src="x"></main>`, "<img> has no alt"],
    ["KaTeX error", `<main><h1>A</h1><span class="katex-error"></span></main>`, "KaTeX error"],
    ["KaTeX without MathML", `<main><h1>A</h1><span class="katex"></span></main>`, "no MathML"],
    [
      "summary not first",
      `<main><h1>A</h1><details><p>x</p><summary>y</summary></details></main>`,
      "not the first child",
    ],
  ])("%s", (_name, body, expected) => {
    expect(auditAccessibility(doc(body)).join("\n")).toContain(expected);
  });

  test.each([
    ["aria-label", `<input aria-label="x">`],
    ["wrapping label", `<label>x <input></label>`],
    ["label for", `<label for="a">x</label><input id="a">`],
    ["aria-labelledby", `<span id="a">x</span><input aria-labelledby="a">`],
    ["hidden input", `<input type="hidden">`],
    ["named button by aria", `<button aria-label="x"></button>`],
    ["named button by title", `<button title="x"></button>`],
    ["named button by text", `<button>x</button>`],
    ["labelled picture", `<svg role="img" aria-label="x"></svg>`],
    ["hidden picture", `<svg aria-hidden="true"></svg>`],
    ["picture in hidden", `<span aria-hidden="true"><svg></svg></span>`],
    ["svg in a labelled picture", `<svg role="img" aria-label="x"><svg></svg></svg>`],
    ["equation", `<span class="katex"><math></math></span>`],
  ])("accepts %s", (_name, inner) => {
    expect(auditAccessibility(doc(`<main><h1>A</h1>${inner}</main>`))).toEqual([]);
  });
});

test("the skip link has to come first and point at main", () => {
  expect(auditSkipLink(doc(`<main><h1>A</h1></main>`)).join()).toContain("no id");
  expect(
    auditSkipLink(doc(`<a href="/">Home</a><a href="#m">Skip</a><main id="m"></main>`)).join(),
  ).toContain("not a skip link");
});

describe("auditMetadata", () => {
  const head = [
    "<title>T</title>",
    '<meta name="description" content="D">',
    '<meta property="og:title" content="T">',
    '<meta property="og:description" content="D">',
    '<meta name="twitter:card" content="summary">',
    '<link rel="icon" href="/icon.svg">',
  ];

  test("a complete head has no problems", () => {
    expect(auditMetadata(doc("", head.join("")))).toEqual([]);
  });

  test.each([
    [0, "no <title>"],
    [1, "no meta description"],
    [2, "no og:title"],
    [3, "no og:description"],
    [4, "no twitter:card"],
    [5, "no favicon link"],
  ])("removing part %i reports it", (index, expected) => {
    const without = head.filter((_, i) => i !== index).join("");
    expect(auditMetadata(doc("", without))).toEqual([expected]);
  });
});

describe("links", () => {
  test("collects hrefs and ids", () => {
    const page = doc(`<a href="/x#y">Text</a><p id="p"></p>`);
    expect(collectLinks(page)).toEqual([{ href: "/x#y", text: "Text" }]);
    expect(collectIds(page)).toEqual(new Set(["p"]));
  });

  test.each([
    ["/a/b/", "/from/", "", { path: "/a/b/", hash: "" }],
    ["/a/b/#x", "/from/", "", { path: "/a/b/", hash: "x" }],
    ["#x", "/from/", "", { path: "/from/", hash: "x" }],
    ["/a/?eq=y#x", "/from/", "", { path: "/a/", hash: "x" }],
    ["../c/", "/a/b/", "", { path: "/a/c/", hash: "" }],
    ["/base/a/", "/from/", "/base", { path: "/a/", hash: "" }],
    ["https://example.com/", "/from/", "", null],
    ["mailto:x@example.com", "/from/", "", null],
    ["//example.com/", "/from/", "", null],
    ["/base/", "/from/", "/base", { path: "/", hash: "" }],
    ["/a/", "/from/", "/base", { path: "/a/", hash: "", outsideBase: true }],
    ["/basement/", "/from/", "/base", { path: "/basement/", hash: "", outsideBase: true }],
  ])("resolves %s", (href, from, base, expected) => {
    expect(resolveInternal(href, from, base)).toEqual(expected);
  });

  const BASE = "https://github.com/o/r";

  test("parses a permalink", () => {
    expect(parsePermalink(`${BASE}/blob/abc/ops.py#L3-L9`, BASE)).toEqual({
      sha: "abc",
      file: "ops.py",
      start: 3,
      end: 9,
    });
    expect(parsePermalink(`${BASE}/blob/abc/d/x.py#L3`, BASE)).toEqual({
      sha: "abc",
      file: "d/x.py",
      start: 3,
      end: 3,
    });
    expect(parsePermalink(`${BASE}/blob/abc/x.py`, BASE)?.start).toBeNull();
    expect(parsePermalink(`${BASE}/commit/abc`, BASE)).toBeNull();
    expect(parsePermalink("https://example.com/", BASE)).toBeNull();
  });

  test("a permalink must be pinned, name a real file and stay inside it", () => {
    const lines = (file: string) => (file === "ops.py" ? 100 : null);
    const link = (sha: string, file: string, start: number | null, end: number | null) => ({
      sha,
      file,
      start,
      end,
    });
    expect(auditPermalink(link("abc", "ops.py", 3, 9), "abc", lines)).toBeNull();
    expect(auditPermalink(link("abc", "ops.py", null, null), "abc", lines)).toBeNull();
    expect(auditPermalink(link("old", "ops.py", 3, 9), "abc", lines)).toContain("pinned to old");
    expect(auditPermalink(link("abc", "no.py", 3, 9), "abc", lines)).toContain("does not exist");
    expect(auditPermalink(link("abc", "ops.py", 3, 101), "abc", lines)).toContain("outside");
    expect(auditPermalink(link("abc", "ops.py", 9, 3), "abc", lines)).toContain("outside");
    expect(auditPermalink(link("abc", "ops.py", 0, 3), "abc", lines)).toContain("outside");
  });
});

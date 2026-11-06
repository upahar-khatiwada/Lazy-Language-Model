// @vitest-environment happy-dom

import { createProcessor } from "@mdx-js/mdx";
import type { Root, RootContent } from "mdast";
import { toString } from "mdast-util-to-string";
import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import remarkGfm from "remark-gfm";
import { afterEach, describe, expect, test } from "vitest";
import { Collapsible } from "@/components/Collapsible";
import { DataTable } from "@/components/DataTable";
import { mdxComponents } from "@/components/mdx";
import { listContentSlugs, readPageBody } from "@/lib/content";
import { FOLDED_STEPS, remarkCollapse } from "@/lib/remark-collapse";
import {
  COLLAPSIBLE_SELECTOR,
  openAllForPrint,
  openAncestors,
  revealEquation,
  revealHash,
  revealHref,
} from "@/lib/reveal";

type Jsx = { type: string; name?: string; attributes?: { name: string; value: unknown }[] };

// The tree of a page after the plugins, as the MDX compiler sees it.
async function treeOf(body: string, collapse: boolean): Promise<Root> {
  let captured: Root | undefined;
  const processor = createProcessor({
    remarkPlugins: [
      remarkGfm,
      ...(collapse ? [remarkCollapse] : []),
      () => (tree: Root) => {
        captured = structuredClone(tree);
      },
    ],
  });
  await processor.process(body);
  return captured!;
}

const jsx = (node: RootContent) => node as unknown as Jsx & { children: RootContent[] };
const isCollapsible = (node: RootContent) => jsx(node).name === "Collapsible";
const isStep = (node: RootContent) => jsx(node).name === "ConceptStep";
const attr = (node: RootContent, name: string) =>
  jsx(node).attributes?.find((candidate) => candidate.name === name);

// Everything a reader could read, in order, with the folds taken out again.
function flatten(children: RootContent[]): string[] {
  return children.flatMap((node) => {
    if (isCollapsible(node)) {
      const step = attr(node, "step")?.value as string | undefined;
      return [...(step ? [`step:${step}`] : []), ...flatten(jsx(node).children)];
    }
    if (isStep(node)) return [`step:${toString(node)}`];
    return [
      JSON.stringify([
        node.type,
        jsx(node).name ?? null,
        jsx(node).attributes?.map(({ name, value }) => [name, value]) ?? null,
        toString(node),
      ]),
    ];
  });
}

const PAGES = listContentSlugs().map((slug) => ({ slug, body: readPageBody(slug) }));
const PROTOCOL_PAGES = PAGES.filter(({ body }) => body.includes("<ConceptStep>"));

describe("the collapse plugin", () => {
  test("covers the long pages", () => {
    expect(PROTOCOL_PAGES.length).toBeGreaterThanOrEqual(13);
    expect(PROTOCOL_PAGES.map(({ slug }) => slug)).toContain("mathematics/attention");
  });

  test.each(PROTOCOL_PAGES)("$slug loses nothing and changes no order", async ({ body }) => {
    const before = await treeOf(body, false);
    const after = await treeOf(body, true);
    expect(flatten(after.children)).toEqual(flatten(before.children));
  });

  test.each(PROTOCOL_PAGES)("$slug keeps headings outside the folds", async ({ body }) => {
    const after = await treeOf(body, true);
    for (const node of after.children.filter(isCollapsible)) {
      expect(jsx(node).children.filter(({ type }) => type === "heading")).toEqual([]);
      expect(jsx(node).children.filter(isStep)).toEqual([]);
    }
  });

  test.each(PROTOCOL_PAGES)(
    "$slug opens its first folded section and only that",
    async ({ body }) => {
      const after = await treeOf(body, true);
      let section = -1;
      const opened = new Set<number>();
      const folded: number[] = [];
      for (const node of after.children) {
        if (node.type === "heading" && node.depth === 2) section++;
        if (isCollapsible(node)) {
          folded.push(section);
          if (attr(node, "open")) opened.add(section);
        }
      }
      expect(folded.length).toBeGreaterThan(0);
      expect([...opened]).toEqual([folded[0]]);
    },
  );

  test("the Attention page folds what the task names, and keeps the nine steps", async () => {
    const after = await treeOf(readPageBody("mathematics/attention"), true);
    const folds = after.children.filter(isCollapsible);
    const steps = folds.map((node) => attr(node, "step")?.value);
    for (const step of FOLDED_STEPS) expect(steps).toContain(step);
    // The backward pass derivation folds too, and the full listing, which has no step.
    expect(steps).toContain("Mathematics");
    expect(folds.map((node) => attr(node, "title")?.value)).toContain(
      "Code listing: transformer.attention_forward",
    );

    const names = flatten(after.children).filter((line) => line.startsWith("step:"));
    expect(names.slice(0, 9)).toEqual([
      "step:What is it?",
      "step:Why does the model need it?",
      "step:Mathematics",
      "step:What does each symbol mean?",
      "step:Tiny numerical example",
      "step:Tensor shapes",
      "step:Actual implementation",
      "step:Code and math connection",
      "step:Where is it used?",
    ]);
  });

  test("a page without the protocol is left as written", async () => {
    const body = readPageBody("getting-started/how-this-llm-works");
    const after = await treeOf(body, true);
    expect(after.children.filter(isCollapsible)).toEqual([]);
  });

  test("a step with nothing under it stays a plain step", async () => {
    const body = ["## A", "", "<ConceptStep>Actual implementation</ConceptStep>", "", "## B"].join(
      "\n",
    );
    const after = await treeOf(body, true);
    expect(after.children.filter(isCollapsible)).toEqual([]);
    expect(after.children.filter(isStep).length).toBe(1);
  });

  test("MDX can use the component the plugin writes", () => {
    expect(Object.keys(mdxComponents)).toContain("Collapsible");
  });

  test("no page names the component itself", () => {
    for (const { slug, body } of PAGES) expect(body, slug).not.toContain("<Collapsible");
  });
});

describe("Collapsible", () => {
  const render = (props: Parameters<typeof Collapsible>[0]) =>
    renderToStaticMarkup(createElement(Collapsible, props));

  test("the summary holds the step's real h3, and the content is in the page while closed", () => {
    const html = render({ step: "Actual implementation", children: "the body text" });
    expect(html).toMatch(/^<details[^>]*data-collapsible=""/);
    expect(html).not.toMatch(/^<details[^>]* open/);
    expect(html).toContain('<h3 data-concept-step="">Actual implementation</h3>');
    expect(html).toContain("the body text");
  });

  test("open and titled", () => {
    const html = render({ title: "Code listing: x", open: true, children: "body" });
    expect(html).toMatch(/^<details[^>]* open=""/);
    expect(html).toContain("Code listing: x");
    expect(html).not.toContain("<h3");
  });
});

describe("opening what a link points at", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  function page() {
    document.body.innerHTML = `
      <h2 id="outside">Outside</h2>
      <details id="first" data-collapsible="" open><summary>a</summary><p id="in-open">a</p></details>
      <details id="second" data-collapsible="">
        <summary>b</summary>
        <figure id="eq-softmax-shift" data-eq="softmax-shift">equation</figure>
        <details id="inner"><summary>c</summary><p id="deep">c</p></details>
      </details>
      <details id="third" data-collapsible="">
        <summary>d</summary>
        <span data-eqs="attention-mask softmax-shift" data-line="83">line</span>
      </details>
      <details id="fourth" data-collapsible=""><summary>e</summary></details>`;
  }
  const open = (id: string) => (document.getElementById(id) as HTMLDetailsElement).open;

  test("an anchor opens its section, and every section around it", () => {
    page();
    expect(revealHash("#deep")?.id).toBe("deep");
    expect(open("second")).toBe(true);
    expect(open("inner")).toBe(true);
    expect(open("third")).toBe(false);
  });

  test("nothing needs doing for a target that is already visible", () => {
    page();
    expect(revealHash("#outside")).toBeNull();
    expect(revealHash("#in-open")).toBeNull();
    expect(revealHash("#missing")).toBeNull();
    expect(revealHash("")).toBeNull();
    expect(revealHash("#%E0%A4%A")).toBeNull();
    expect(open("second")).toBe(false);
  });

  test("a search result's address opens its section", () => {
    page();
    revealHref("/mathematics/attention/#eq-softmax-shift");
    expect(open("second")).toBe(true);
    revealHref("/no/hash/");
    expect(open("third")).toBe(false);
  });

  test("?eq= opens the equation's section and the sections of its code lines", () => {
    page();
    revealEquation("softmax-shift");
    expect(open("second")).toBe(true);
    expect(open("third")).toBe(true);
    expect(open("fourth")).toBe(false);
  });

  test("a slug matches a whole word of data-eqs, not part of one", () => {
    page();
    revealEquation("attention");
    expect(open("third")).toBe(false);
  });

  test("printing opens every folded section, and can be undone", () => {
    page();
    const opened = openAllForPrint();
    expect(opened.map(({ id }) => id)).toEqual(["second", "third", "fourth"]);
    expect(
      [...document.querySelectorAll<HTMLDetailsElement>(COLLAPSIBLE_SELECTOR)].every((d) => d.open),
    ).toBe(true);
    // A chart's own data table is not a folded section.
    expect(open("inner")).toBe(false);
  });

  test("openAncestors reports whether anything changed", () => {
    page();
    const deep = document.getElementById("deep")!;
    expect(openAncestors(deep)).toBe(true);
    expect(openAncestors(deep)).toBe(false);
  });
});

describe("the wiring that uses it", () => {
  test("the docs layout mounts SectionReveal, and the compiler runs the plugin", () => {
    expect(readFileSync("app/(docs)/layout.tsx", "utf-8")).toContain("<SectionReveal />");
    expect(readFileSync("lib/content.ts", "utf-8")).toContain("remarkCollapse");
  });

  test("a pin and a search result open folded sections", () => {
    expect(readFileSync("components/MathCodeNavigator.tsx", "utf-8")).toContain("revealEquation");
    expect(readFileSync("components/SearchDialog.tsx", "utf-8")).toContain("revealHref");
  });

  test("the skip link and the landmark agree, and a Python function cannot take the id", () => {
    const chrome = readFileSync("components/DocsChrome.tsx", "utf-8");
    const main = readFileSync("components/DocMain.tsx", "utf-8");
    const id = main.match(/<main[^>]*id="([^"]+)"/s)?.[1];
    expect(id).toBeDefined();
    expect(chrome).toContain(`href="#${id}"`);
    // train.py and talk.py both define main, which becomes an anchor in the Code Explorer.
    expect(id).not.toBe("main");
  });
});

describe("tables stay in their own container", () => {
  test("a DataTable can take keyboard focus to scroll, and has a label", () => {
    const html = renderToStaticMarkup(createElement(DataTable, { label: "Gradient check" }));
    expect(html).toContain('tabindex="0"');
    expect(html).toContain('aria-label="Gradient check"');
    const css = readFileSync("components/DataTable.module.css", "utf-8");
    expect(css).toContain("overflow: auto");
    expect(css).toContain("max-height");
    expect(css).toContain("position: sticky");
  });
});

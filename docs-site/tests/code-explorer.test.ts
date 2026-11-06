import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import { CodeFileBody, SourceListing } from "@/components/CodeExplorer";
import { NAV } from "@/data/nav";
import {
  buildFileTree,
  callableSymbols,
  explorerFiles,
  explorerHref,
  fileSlug,
  filePurpose,
  fileSymbols,
  findExplorerFile,
  fileUsedBy,
  isTestFile,
  locateSymbol,
  mathForFile,
  pipelineSteps,
  reachCounts,
  reachedFrom,
  sourceTree,
  symbolPurpose,
  usedBy,
  type ExplorerFile,
  type SourceNode,
} from "@/lib/code-explorer";
import { listContentSlugs } from "@/lib/content";
import { loadLinks } from "@/lib/links";
import { activeHref, resolveNav } from "@/lib/nav";
import { loadCallGraph, loadCodeIndex, loadTestsIndex } from "@/lib/repo";

const files = explorerFiles();
const allSymbols = files.flatMap((file) => fileSymbols(file).map((symbol) => ({ file, symbol })));

// Walks rendered HTML and returns, for each id on a <span>, the data-line values inside it.
function anchoredLines(html: string): Map<string, number[]> {
  const found = new Map<string, number[]>();
  const stack: (string | null)[] = [];
  for (const match of html.matchAll(/<(\/?)span\b([^>]*)>/g)) {
    if (match[1]) {
      stack.pop();
      continue;
    }
    const attributes = match[2]!;
    const id = attributes.match(/\bid="([^"]+)"/)?.[1] ?? null;
    stack.push(id);
    if (id && !found.has(id)) found.set(id, []);
    const line = attributes.match(/\bdata-line="(\d+)"/)?.[1];
    if (line) {
      for (const open of stack) if (open) found.get(open)!.push(Number(line));
    }
  }
  return found;
}

function linesOf(nodes: SourceNode[]): number[] {
  return nodes.flatMap((node) => (node.kind === "line" ? [node.line] : linesOf(node.children)));
}

function findNode(nodes: SourceNode[], anchor: string): SourceNode | undefined {
  for (const node of nodes) {
    if (node.kind !== "symbol") continue;
    if (node.anchor === anchor) return node;
    const inside = findNode(node.children, anchor);
    if (inside) return inside;
  }
  return undefined;
}

describe("the file tree", () => {
  test("holds exactly the files in code-index.json, with tests in their folder", () => {
    const tree = buildFileTree(files.map((file) => file.path));
    const paths = (nodes: typeof tree): string[] =>
      nodes.flatMap((node) => (node.type === "file" ? [node.path] : paths(node.children)));
    expect(paths(tree).sort()).toEqual(files.map((file) => file.path).sort());
    const folder = tree.find((node) => node.type === "folder");
    expect(folder?.name).toBe("tests");
  });

  test("is built from the paths given, not from names known in advance", () => {
    expect(buildFileTree(["b.py", "x/y/z.py", "a.py", "x/w.py"])).toEqual([
      { type: "file", name: "b.py", path: "b.py" },
      { type: "file", name: "a.py", path: "a.py" },
      {
        type: "folder",
        name: "x",
        children: [
          { type: "file", name: "w.py", path: "x/w.py" },
          {
            type: "folder",
            name: "y",
            children: [{ type: "file", name: "z.py", path: "x/y/z.py" }],
          },
        ],
      },
    ]);
  });

  test("labels as tests exactly the files in tests-index.json", () => {
    const tests = files.filter((file) => isTestFile(file.path)).map((file) => file.path);
    expect(tests).toEqual(loadTestsIndex().files.map((file) => file.path));
    expect(tests.length).toBeGreaterThan(0);
  });
});

describe("every file has a page", () => {
  test("slugs are unique and do not hide a written page", () => {
    const slugs = files.map((file) => fileSlug(file.path));
    expect(new Set(slugs).size).toBe(slugs.length);
    const pages = listContentSlugs().filter((slug) => slug.startsWith("code/"));
    for (const slug of slugs) expect(pages).not.toContain(`code/${slug}`);
    const planned = NAV.flatMap((group) => group.pages)
      .map((page) => page.slug)
      .filter((slug) => slug.startsWith("code/"));
    for (const slug of slugs) expect(planned).not.toContain(`code/${slug}`);
  });

  test.each(files.map((file) => [file.path, file] as const))("%s", (_path, file) => {
    expect(findExplorerFile(fileSlug(file.path))).toBe(file);
    expect(explorerHref(file.path)).toBe(`/code/${fileSlug(file.path)}/`);
  });

  test("deep links have the requested form", () => {
    expect(explorerHref("ops.py", "softmax_forward")).toBe("/code/ops/#softmax_forward");
    expect(explorerHref("tests/test_ops.py", "test_relu")).toBe("/code/test_ops/#test_relu");
  });

  test("the Code Explorer entry stays current on a file page, and other pages are unaffected", () => {
    const nav = resolveNav(NAV, listContentSlugs());
    expect(activeHref(nav, "/code/ops/")).toBe("/code/");
    expect(activeHref(nav, "/code/")).toBe("/code/");
    expect(activeHref(nav, "/code/checkpoints-tests/")).toBe("/code/checkpoints-tests/");
    expect(activeHref(nav, "/mathematics/attention/")).toBe("/mathematics/attention/");
    expect(activeHref(nav, "/design/")).toBeUndefined();
  });
});

describe("purposes are not hand-written", () => {
  test("the docstring wins over the contract text", () => {
    const file = { ...files[0]!, docstring: "From the docstring." };
    expect(filePurpose(file)).toBe("From the docstring.");
  });

  test.each(files.map((file) => [file.path, file] as const))("%s has contract text", (_p, file) => {
    expect(file.docstring).toBeNull();
    expect(filePurpose(file)).not.toContain("Not available");
  });

  test("every function, class and method has a purpose from the contract", () => {
    const missing = files
      .flatMap((file) => callableSymbols(file))
      .filter((symbol) => symbolPurpose(symbol.id) === null)
      .map((symbol) => symbol.id);
    expect(missing).toEqual([]);
  });
});

describe("source and anchors", () => {
  test.each(files.map((file) => [file.path, file] as const))(
    "%s: the stored source is the whole file",
    (_path, file) => {
      expect(file.source.split("\n").length).toBe(file.lineCount);
    },
  );

  test("anchors are unique within a file and cannot collide with sections or lines", () => {
    for (const file of files) {
      const anchors = fileSymbols(file).map((symbol) => symbol.anchor);
      expect(new Set(anchors).size, file.path).toBe(anchors.length);
      for (const anchor of anchors) {
        expect(anchor, file.path).not.toMatch(/^L\d+$|^file-/);
      }
    }
  });

  test.each(allSymbols.map(({ file, symbol }) => [symbol.id, file, symbol] as const))(
    "%s: its anchor covers exactly its lines, and they are its source",
    (_id, file, symbol) => {
      const tree = sourceTree(file.lineCount, fileSymbols(file));
      const node = findNode(tree, symbol.anchor);
      expect(node?.kind).toBe("symbol");
      const lines = linesOf((node as Extract<SourceNode, { kind: "symbol" }>).children);
      const expected = Array.from(
        { length: symbol.endLine - symbol.startLine + 1 },
        (_, index) => symbol.startLine + index,
      );
      expect(lines).toEqual(expected);
      const text = file.source.split("\n").slice(symbol.startLine - 1, symbol.endLine);
      expect(text.join("\n")).toBe(symbol.source);
    },
  );

  test("every line appears once, in order, including those outside any symbol", () => {
    for (const file of files) {
      const lines = linesOf(sourceTree(file.lineCount, fileSymbols(file)));
      expect(lines, file.path).toEqual(Array.from({ length: file.lineCount }, (_, i) => i + 1));
    }
  });

  test("symbols that overlap without nesting are refused", () => {
    expect(() =>
      sourceTree(5, [
        { anchor: "a", startLine: 1, endLine: 3 },
        { anchor: "b", startLine: 2, endLine: 4 },
      ]),
    ).toThrow("crosses the end");
  });

  test.each(files.map((file) => [file.path, file] as const))(
    "%s: the rendered listing shows every anchor on its lines",
    async (_path, file) => {
      const html = renderToStaticMarkup(await SourceListing({ file, permalink: "https://x/" }));
      const anchors = anchoredLines(html);
      for (const symbol of fileSymbols(file)) {
        const expected = Array.from(
          { length: symbol.endLine - symbol.startLine + 1 },
          (_, index) => symbol.startLine + index,
        );
        expect(anchors.get(symbol.anchor), symbol.id).toEqual(expected);
      }
      expect(anchors.get("L1")).toEqual([1]);
      expect(anchors.get(`L${file.lineCount}`)).toEqual([file.lineCount]);
    },
  );
});

describe("used-by lists equal the call graph", () => {
  const graph = loadCallGraph();

  test("every symbol the call graph knows is shown from the same record", () => {
    for (const { symbol } of allSymbols) {
      const expected = graph.usedBy[symbol.id] ?? { production: [], tests: [] };
      expect(usedBy(symbol.id), symbol.id).toEqual(expected);
    }
  });

  test("every symbol in the call graph's usedBy has a page and an anchor", () => {
    for (const id of Object.keys(graph.usedBy)) {
      const { path, anchor } = locateSymbol(id);
      expect(findExplorerFile(fileSlug(path)), id).toBeDefined();
      expect(anchor, id).toBeDefined();
    }
  });

  test("every caller can be linked, including module-level code", () => {
    expect(locateSymbol("train.<module>")).toEqual({ path: "train.py" });
    for (const entry of Object.values(graph.usedBy)) {
      for (const id of [...entry.production, ...entry.tests]) {
        expect(() => locateSymbol(id), id).not.toThrow();
      }
    }
  });

  test.each(files.map((file) => [file.path, file] as const))(
    "%s: the rendered page lists exactly the call graph's callers",
    (_path, file) => {
      const html = renderToStaticMarkup(CodeFileBody({ file, children: null }));
      for (const symbol of callableSymbols(file)) {
        const start = html.indexOf(`data-symbol-card="${symbol.anchor}"`);
        expect(start, symbol.id).toBeGreaterThan(-1);
        const card = html.slice(start, html.indexOf("</article>", start));
        for (const group of ["production", "tests"] as const) {
          const section = card.match(
            new RegExp(`data-used-by="${group}">(.*?)</dd>`),
          )?.[1] as string;
          const shown = [...section.matchAll(/data-symbol-ref="([^"]+)"/g)].map((m) =>
            m[1]!.replaceAll("&lt;", "<").replaceAll("&gt;", ">"),
          );
          expect(shown, `${symbol.id} ${group}`).toEqual(graph.usedBy[symbol.id]?.[group] ?? []);
        }
      }
    },
  );

  test("known edges, read from the code", () => {
    expect(usedBy("ops.softmax_forward").production).toEqual([
      "generate.sample_next",
      "transformer.attention_forward",
    ]);
    expect(usedBy("checkpoint.load_checkpoint").production).toEqual(["talk.main"]);
    expect(usedBy("checkpoint.save_checkpoint").production).toEqual(["train.main"]);
    expect(usedBy("ops.linear_forward").production).toEqual([]);
  });

  test("a file's used-by is the union of its symbols' callers in other files", () => {
    const find = (path: string) => files.find((file) => file.path === path) as ExplorerFile;
    expect(fileUsedBy(find("checkpoint.py")).production).toEqual(["talk.py", "train.py"]);
    expect(fileUsedBy(find("gradcheck.py")).production).toEqual([]);
    expect(fileUsedBy(find("gradcheck.py")).tests).toEqual(["tests/test_gradcheck.py"]);
  });
});

describe("mathematics comes from link records", () => {
  test("every link record's file shows that concept", () => {
    for (const file of files) {
      const links = loadLinks().filter((link) => link.file === file.path);
      expect(mathForFile(file.path).length > 0, file.path).toBe(links.length > 0);
    }
  });

  test("a line in a math row is a line in the file", () => {
    for (const file of files) {
      for (const use of mathForFile(file.path)) {
        for (const { start, end } of use.lines) {
          expect(start).toBeGreaterThanOrEqual(1);
          expect(end).toBeLessThanOrEqual(file.lineCount);
        }
      }
    }
  });

  test("softmax in ops.py links to its explanation", () => {
    const softmax = mathForFile("ops.py").find((use) => use.conceptId === "ops.softmax_forward")!;
    expect(softmax.roles).toContain("implements");
    expect(softmax.href).toMatch(/^\/mathematics\/softmax-cross-entropy\/#/);
  });
});

describe("where a file fits", () => {
  test("the reach counts add up to the contract's 53 of 58", () => {
    const symbols = files
      .filter((file) => !isTestFile(file.path))
      .flatMap((file) => callableSymbols(file));
    expect(symbols.length).toBe(58);
    const unreached = symbols.filter((symbol) => reachedFrom(symbol.id).length === 0);
    expect(unreached.map((symbol) => symbol.id).sort()).toEqual([
      "gradcheck.check_params",
      "gradcheck.numerical_grad",
      "gradcheck.relative_error",
      "ops.linear_forward",
      "transformer.count_params",
    ]);
  });

  test("gradcheck.py is reached from neither entry point", () => {
    const gradcheck = files.find((file) => file.path === "gradcheck.py")!;
    expect(reachCounts(gradcheck)).toMatchObject({ train: 0, talk: 0 });
  });

  test("pipeline steps follow data/pipeline.ts", () => {
    expect(pipelineSteps("data.py").map((step) => step.label)).toEqual(["Text", "Ids", "Windows"]);
    expect(pipelineSteps("gradcheck.py")).toEqual([]);
    expect(pipelineSteps("train.py").some((step) => step.label === "Clipping")).toBe(true);
  });

  test("the code index has no file the tree would miss", () => {
    expect(files.length).toBe(loadCodeIndex().files.length);
  });
});

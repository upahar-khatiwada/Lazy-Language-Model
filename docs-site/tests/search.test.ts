import { beforeAll, describe, expect, test } from "vitest";
import {
  groupHits,
  highlightParts,
  isExactName,
  nextActive,
  opensSearch,
  RESULT_TYPES,
  tokenize,
  TYPE_LABELS,
  type SearchDoc,
  type SearchHit,
  type SearchResults,
} from "@/lib/search";
import { loadIndex, RESULTS_PER_TYPE, runSearch } from "@/lib/search-engine";
import {
  buildSearchDocuments,
  buildSearchIndexJson,
  SEARCH_INDEX_LIMIT_BYTES,
} from "@/lib/search-index";

let documents: SearchDoc[];
let json: string;
let search: (query: string) => SearchResults;

beforeAll(async () => {
  documents = await buildSearchDocuments();
  ({ json } = await buildSearchIndexJson());
  // Through the serialized form, the way the dialog loads it.
  const index = loadIndex(json);
  search = (query) => runSearch(index, query);
}, 60_000);

const titles = (results: SearchResults, type: SearchDoc["type"]) =>
  results.flat.filter((hit) => hit.type === type).map((hit) => hit.title);

describe("the softmax example, from real data", () => {
  test.each([
    ["concept", "Softmax"],
    ["concept", "Softmax, backward"],
    ["concept", "Cross-entropy"],
    ["concept", "Softmax per row"],
    ["page", "Attention"],
    ["file", "ops.py"],
    ["function", "softmax_forward"],
    ["function", "softmax_backward"],
  ] as const)("%s %s is returned", (type, title) => {
    expect(titles(search("softmax"), type).map((found) => found.toLowerCase())).toContain(
      title.toLowerCase(),
    );
  });

  test("the concept named exactly what was typed comes first", () => {
    expect(titles(search("softmax"), "concept")[0]).toBe("Softmax");
    expect(titles(search("softmax_forward"), "function")[0]).toBe("softmax_forward");
  });

  test("results are grouped in the order of the type list, each with its label", () => {
    const { groups } = search("softmax");
    expect(groups.map((group) => group.type)).toEqual(
      RESULT_TYPES.filter((type) => groups.some((group) => group.type === type)),
    );
    for (const group of groups) expect(group.label).toBe(TYPE_LABELS[group.type]);
    expect(groups.map((group) => group.type)).toEqual([...RESULT_TYPES]);
  });

  test("each result links to a real route, not a guess", () => {
    const forward = search("softmax_forward").flat.find((hit) => hit.title === "softmax_forward")!;
    expect(forward.href).toBe("/code/ops/#softmax_forward");
    const file = search("softmax").flat.find((hit) => hit.title === "ops.py")!;
    expect(file.href).toBe("/code/ops/");
    const concept = search("softmax").flat.find((hit) => hit.title === "Softmax")!;
    expect(concept.href).toContain("/mathematics/softmax-cross-entropy/");
    const equation = search("softmax.shift").flat.find((hit) => hit.type === "equation")!;
    expect(equation.href).toContain("eq=softmax-shift");
  });
});

describe("a query that is exactly a function or file name", () => {
  const first = (query: string) => search(query).flat[0]!;

  test.each([
    ["softmax_forward", "function", "softmax_forward"],
    ["SOFTMAX_FORWARD", "function", "softmax_forward"],
    ["  softmax_forward  ", "function", "softmax_forward"],
    ["Tokenizer.encode", "function", "Tokenizer.encode"],
    ["encode", "function", "Tokenizer.encode"],
    ["ops.py", "file", "ops.py"],
    ["tests/test_ops.py", "file", "tests/test_ops.py"],
    ["test_ops.py", "file", "tests/test_ops.py"],
  ] as const)("%j opens the %s %s", (query, type, title) => {
    expect(first(query)).toMatchObject({ type, title });
  });

  test("every function and file in the index ranks first for its own name", () => {
    for (const doc of documents) {
      if (doc.type !== "function" && doc.type !== "file") continue;
      expect(first(doc.title), doc.title).toMatchObject({ type: doc.type, title: doc.title });
    }
  });

  test("its group is listed first, so the first row on screen is the one Enter opens", () => {
    const { groups, flat } = search("softmax_forward");
    expect(groups[0]!.type).toBe("function");
    expect(groups[0]!.hits[0]).toBe(flat[0]);
    expect(groups.map((group) => group.type).slice(1)).toEqual(
      RESULT_TYPES.filter((type) => type !== "function").filter((type) =>
        groups.some((group) => group.type === type),
      ),
    );
  });

  test("a partial name or a phrase keeps the usual order", () => {
    expect(first("softmax").type).toBe("concept");
    expect(first("softmax forward").type).toBe("concept");
    expect(first("softmax_forw").type).toBe("concept");
  });

  test("isExactName takes a file's path or base name, and a function's title or method", () => {
    const hit = (type: SearchDoc["type"], title: string) => ({ type, title });
    expect(isExactName(hit("file", "tests/test_ops.py"), "TEST_OPS.PY")).toBe(true);
    expect(isExactName(hit("file", "tests/test_ops.py"), "test_ops")).toBe(false);
    expect(isExactName(hit("function", "Adam.step"), "step")).toBe(true);
    expect(isExactName(hit("function", "Adam.step"), "Adam.step")).toBe(true);
    expect(isExactName(hit("function", "Adam.step"), "Adam")).toBe(false);
    expect(isExactName(hit("concept", "Softmax"), "softmax")).toBe(false);
    expect(isExactName(hit("function", "main"), "   ")).toBe(false);
  });
});

describe("what is indexed", () => {
  test("LaTeX and equation text can be found", () => {
    expect(search("sqrt").flat.some((hit) => hit.type === "equation")).toBe(true);
    const equation = search("frac sum").flat.find((hit) => hit.type === "equation")!;
    expect(equation.formula).toContain("\\frac");
  });

  test("only an Equation result carries a formula to show", () => {
    for (const hit of search("softmax").flat) {
      if (hit.type === "equation") expect(hit.formula).not.toBe("");
      else expect(hit.formula).toBe("");
    }
  });

  test("a symbol's meaning finds its concept", () => {
    expect(titles(search("shifted exponentials"), "concept")).toContain("Softmax");
  });

  test("page descriptions, headings and navigation labels are found", () => {
    expect(titles(search("hand-written backward"), "page").length).toBeGreaterThan(0);
    expect(titles(search("Encode and decode"), "page")).toContain("Encode and decode");
    expect(titles(search("Reference"), "page")).toContain("Not in This Model");
  });

  test("function leading comments and purposes are found", () => {
    const withComment = documents.find(
      (doc) => doc.type === "function" && doc.body.includes("Samples a"),
    );
    expect(withComment?.title).toBe("xavier_uniform");
    expect(titles(search("Samples uniform distribution"), "function")).toContain("xavier_uniform");
  });

  test("every file in the code index and every written page has a document", async () => {
    const { explorerFiles } = await import("@/lib/code-explorer");
    const { listContentSlugs } = await import("@/lib/content");
    const ids = new Set(documents.map((doc) => doc.id));
    for (const file of explorerFiles()) expect(ids).toContain(`file:${file.path}`);
    for (const slug of listContentSlugs()) expect(ids).toContain(`page:${slug}`);
  });

  test("document ids are unique", () => {
    expect(new Set(documents.map((doc) => doc.id)).size).toBe(documents.length);
  });

  test("titles and function names outrank a word in the text", () => {
    const pages = search("softmax").flat.filter((hit) => hit.type === "page");
    expect(pages[0]!.title.toLowerCase()).toContain("softmax");
  });

  test("the index is under the limit", () => {
    expect(Buffer.byteLength(json, "utf-8")).toBeLessThan(SEARCH_INDEX_LIMIT_BYTES);
  });

  test("a typo of one letter still finds softmax", () => {
    expect(titles(search("softmaxx"), "concept")).toContain("Softmax");
  });

  test("an empty query or only punctuation returns nothing", () => {
    expect(search("").total).toBe(0);
    expect(search("  \\ {} ").total).toBe(0);
  });

  test("a word that is nowhere returns nothing", () => {
    expect(search("zzzzqqqq").total).toBe(0);
  });
});

describe("search helpers", () => {
  test("tokenize splits code names and LaTeX on everything but letters and digits", () => {
    expect(tokenize("softmax_forward")).toEqual(["softmax", "forward"]);
    expect(tokenize("\\frac{e_i}{2}")).toEqual(["frac", "e", "i", "2"]);
    expect(tokenize("  ")).toEqual([]);
  });

  test("groupHits limits each type and keeps the true total", () => {
    const hit = (id: number, type: SearchDoc["type"]): SearchHit => ({
      id: String(id),
      type,
      title: "t",
      detail: "",
      formula: "",
      href: "/",
      terms: [],
    });
    const hits = [hit(1, "function"), hit(2, "concept"), hit(3, "concept"), hit(4, "concept")];
    const grouped = groupHits(hits, 2);
    expect(grouped.total).toBe(4);
    expect(grouped.groups.map((group) => [group.type, group.total, group.hits.length])).toEqual([
      ["concept", 3, 2],
      ["function", 1, 1],
    ]);
    expect(grouped.flat.map((item) => item.id)).toEqual(["2", "3", "1"]);
  });

  test("the limit per type is applied by runSearch", () => {
    for (const group of search("softmax").groups) {
      expect(group.hits.length).toBeLessThanOrEqual(RESULTS_PER_TYPE);
    }
  });

  test("highlightParts marks matches in any case and treats regex characters as text", () => {
    expect(highlightParts("Softmax backward", ["softmax"])).toEqual([
      { text: "Softmax", match: true },
      { text: " backward", match: false },
    ]);
    expect(highlightParts("softmax_forward", ["softmax", "forward"])).toEqual([
      { text: "softmax", match: true },
      { text: "_", match: false },
      { text: "forward", match: true },
    ]);
    expect(highlightParts("a.b", ["."])).toEqual([
      { text: "a", match: false },
      { text: ".", match: true },
      { text: "b", match: false },
    ]);
    expect(highlightParts("plain", [])).toEqual([{ text: "plain", match: false }]);
  });

  test("nextActive wraps at both ends and stays at 0 with no options", () => {
    expect(nextActive(0, 1, 3)).toBe(1);
    expect(nextActive(2, 1, 3)).toBe(0);
    expect(nextActive(0, -1, 3)).toBe(2);
    expect(nextActive(0, 1, 0)).toBe(0);
  });

  test.each([
    [{ key: "k", ctrlKey: true }, true],
    [{ key: "K", metaKey: true }, true],
    [{ key: "k", ctrlKey: true, tagName: "INPUT" }, true],
    [{ key: "/" }, true],
    [{ key: "/", tagName: "A" }, true],
    [{ key: "/", tagName: "INPUT" }, false],
    [{ key: "/", tagName: "TEXTAREA" }, false],
    [{ key: "/", tagName: "SELECT" }, false],
    [{ key: "/", isContentEditable: true }, false],
    [{ key: "/", ctrlKey: true }, false],
    [{ key: "k" }, false],
    [{ key: "k", ctrlKey: true, altKey: true }, false],
    [{ key: "a", ctrlKey: true }, false],
  ])("opensSearch(%o) is %s", (event, expected) => {
    expect(
      opensSearch({ ctrlKey: false, metaKey: false, altKey: false, ...event } as Parameters<
        typeof opensSearch
      >[0]),
    ).toBe(expected);
  });
});

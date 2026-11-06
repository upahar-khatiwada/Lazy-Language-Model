import { beforeAll, describe, expect, test } from "vitest";
import { loadAbsentTechniques, plainText } from "@/lib/contract-pages";
import {
  exactRank,
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
    expect(groups.map((group) => group.type)).toEqual(
      RESULT_TYPES.filter((type) => type !== "absent"),
    );
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

  test("a partial name keeps the usual order", () => {
    expect(first("softmax").type).toBe("concept");
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

describe("the order of exact matches", () => {
  const first = (query: string) => search(query).flat[0]!;
  const isPageTitle = (doc: SearchDoc) => doc.type === "page" && !doc.id.includes("#");

  test.each(["not in this model", "Not in This Model", "NOT IN THIS MODEL", "not-in-this-model!"])(
    "%j opens the Not in This Model page",
    (query) => {
      const { groups, flat } = search(query);
      expect(flat[0]).toMatchObject({
        type: "page",
        title: "Not in This Model",
        href: "/reference/not-in-this-model/",
      });
      expect(groups[0]!.hits[0]).toBe(flat[0]);
    },
  );

  test.each([
    ["function name", "softmax_forward", "function", "softmax_forward"],
    ["function name without punctuation", "Softmax Forward", "function", "softmax_forward"],
    ["file name", "ops.py", "file", "ops.py"],
    ["file name without punctuation", "OPS PY", "file", "ops.py"],
    ["concept name", "softmax", "concept", "Softmax"],
    ["concept name without punctuation", "top k", "concept", "Top-k"],
    ["concept name with other punctuation", "q k v projections", "concept", "Q, K, V projections"],
  ] as const)("%s: %j opens the %s %s", (_case, query, type, title) => {
    const { groups, flat } = search(query);
    expect(flat[0]).toMatchObject({ type, title });
    expect(groups[0]!.hits[0]).toBe(flat[0]);
  });

  test("a page title beats a concept of the same name", () => {
    expect(documents.some((doc) => isPageTitle(doc) && doc.title === "LayerNorm")).toBe(true);
    expect(documents.some((doc) => doc.type === "concept" && doc.title === "LayerNorm")).toBe(true);
    expect(first("layernorm")).toMatchObject({ type: "page", title: "LayerNorm" });
    expect(
      search("layernorm")
        .groups.map((group) => group.type)
        .slice(0, 2),
    ).toEqual(["page", "concept"]);
  });

  test("a function name beats a concept of the same name", () => {
    expect(titles(search("relative_error"), "concept")).toContain("Relative error");
    expect(first("relative_error")).toMatchObject({ type: "function", title: "relative_error" });
    expect(titles(search("Softmax, backward"), "concept")).toContain("Softmax, backward");
    expect(first("Softmax, backward")).toMatchObject({
      type: "function",
      title: "softmax_backward",
    });
  });

  test("the ranks run page title, then function or file name, then concept name", () => {
    const rank = (type: SearchDoc["type"]) => exactRank({ id: "a", type, title: "same" }, "same");
    expect(exactRank({ id: "page:a", type: "page", title: "same" }, "same")).toBeGreaterThan(
      rank("function"),
    );
    expect(rank("function")).toBe(rank("file"));
    expect(rank("file")).toBeGreaterThan(rank("concept"));
    expect(rank("concept")).toBeGreaterThan(0);
  });

  test("a heading is not a page title", () => {
    expect(first("perplexity").type).toBe("function");
    expect(first("softmax").type).toBe("concept");
    expect(exactRank({ id: "page:a", type: "page", title: "Softmax" }, "softmax")).toBe(3);
    expect(exactRank({ id: "page:a#b", type: "page", title: "Softmax" }, "softmax")).toBe(0);
  });

  test("every page opens for its own title", () => {
    for (const doc of documents.filter(isPageTitle)) {
      expect(first(doc.title), doc.title).toMatchObject({ type: "page", title: doc.title });
    }
  });

  test("every concept opens for its own name, unless a page or function has that name", () => {
    for (const doc of documents.filter((candidate) => candidate.type === "concept")) {
      const outranked = documents.some(
        (other) => exactRank({ id: other.id, type: other.type, title: other.title }, doc.title) > 1,
      );
      if (!outranked) {
        expect(first(doc.title), doc.title).toMatchObject({ type: "concept", title: doc.title });
      }
    }
  });

  test("a partial match moves no group and takes no exact rank", () => {
    expect(first("not in this").type).toBe("concept");
    expect(exactRank({ id: "page:a", type: "page", title: "Not in This Model" }, "not in")).toBe(0);
    for (const query of ["softmax_forw", "not in this", "layer"]) {
      const types = search(query).groups.map((group) => group.type);
      expect(types, query).toEqual(RESULT_TYPES.filter((type) => types.includes(type)));
    }
  });

  test("an equation or a table row of the same name takes no exact rank", () => {
    expect(exactRank({ id: "equation:a", type: "equation", title: "Scores" }, "scores")).toBe(0);
    expect(exactRank({ id: "absent:a", type: "absent", title: "Dropout" }, "dropout")).toBe(0);
  });
});

describe("a technique this code does not use", () => {
  const absent = (query: string) => search(query).flat.filter((hit) => hit.type === "absent");

  test.each([
    ["rotary", "Rotary positions"],
    ["RMSNorm", "RMSNorm"],
    ["KV cache", "KV cache"],
    ["dropout", "Dropout"],
    ["early stopping", "Stop when validation rises"],
    ["BOS", "Pad, BOS, EOS tokens"],
    ["chat.py", "ELIZA chat (chat.py)"],
  ])("%j finds the row %s, which opens Not in This Model", (query, title) => {
    expect(absent(query)[0]).toMatchObject({
      title,
      href: "/reference/not-in-this-model/",
    });
  });

  test("every row of the page can be found by its own name", () => {
    const rows = loadAbsentTechniques().rows;
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      const title = plainText(row.technique);
      expect(
        absent(title).map((hit) => hit.title),
        title,
      ).toContain(title);
    }
  });

  test("the result says the code does not use it, and the group says so too", () => {
    const { groups, flat } = search("rotary");
    expect(flat[0]!.detail).toContain("Not used by this code.");
    expect(groups.at(-1)!.label).toBe("Not in This Model");
  });

  test("the note is the one the page shows, not the contract's note to the author", () => {
    expect(absent("dropout")[0]!.detail).toBe("Not used by this code.");
    expect(absent("weight tying")[0]!.detail).toContain("separate matrix");
  });

  test.each(["positions", "gradient", "loss", "tokens", "embedding", "cache", "encoding"])(
    "%j opens something the code has, with the rows listed after it",
    (query) => {
      const { flat } = search(query);
      expect(flat[0]!.type).not.toBe("absent");
      const firstAbsent = flat.findIndex((hit) => hit.type === "absent");
      if (firstAbsent >= 0) {
        expect(flat.slice(firstAbsent).every((hit) => hit.type === "absent")).toBe(true);
      }
    },
  );

  test("an exact function or file name still opens that function or file", () => {
    expect(search("softmax_forward").flat[0]!.type).toBe("function");
    expect(search("ops.py").flat[0]!.type).toBe("file");
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

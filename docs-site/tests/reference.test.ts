import { describe, expect, test } from "vitest";
import { loadSymbolIndex } from "@/lib/code-targets";
import { conceptAnchor, exampleAnchor, loadConcepts, type MathConcept } from "@/lib/concepts";
import { navSlugs } from "@/lib/pages";
import { loadInitParams } from "@/lib/parameters";
import {
  buildReferenceRow,
  conceptLinks,
  coverage,
  FILTER_LABELS,
  loadInventory,
} from "@/lib/reference";
import { loadCallGraph, loadCodeIndex, loadConfig, loadRepoMeta } from "@/lib/repo";
import { verifySiteData } from "@/lib/site-verify";

const sample = loadConcepts().find((concept) => concept.id === "ops.softmax_forward")!;

const concept = (changes: Partial<MathConcept>): MathConcept => ({ ...sample, ...changes });

describe("rows", () => {
  test("a concept becomes a row with its source, shapes and direction", () => {
    const row = buildReferenceRow(sample, []);
    expect(row.title).toBe("Softmax");
    expect(row.filter).toBe("operations");
    expect(row.directions).toEqual(["forward"]);
    expect(row.equationIds).toEqual(["softmax.shift", "softmax.normalize"]);
    expect(row.sources[0]).toMatchObject({ file: "ops.py", name: "softmax_forward" });
    expect(row.shapes.map((shape) => shape.id)).toEqual(["attention", "sampling"]);
    expect(row.shapes[0]?.stages[0]).toEqual({
      name: "S",
      symbolic: "(H, T, T)",
      values: "(2, 16, 16)",
    });
    // V has no value, so there is nothing to show next to the symbol.
    expect(row.shapes[1]?.stages[0]?.values).toBeNull();
  });

  test("the source link is a valid range in the pinned commit", () => {
    const code = loadSymbolIndex().get("ops.softmax_forward")!;
    const { sha } = loadRepoMeta();
    const [source] = buildReferenceRow(sample, []).sources;
    expect(source?.permalink).toContain(`/blob/${sha}/ops.py#L${code.startLine}-L${code.endLine}`);
    expect(source?.lines).toBe(`${code.startLine}-${code.endLine}`);
  });

  test("a row with an unknown source cannot be built", () => {
    const bad = concept({ source: [{ file: "ops.py", symbol: "ops.nope" }] });
    expect(() => buildReferenceRow(bad, [])).toThrow("unknown source symbol ops.nope");
  });

  test("backward and undifferentiated equations are reported as such", () => {
    const [first] = sample.equations;
    const row = buildReferenceRow(
      concept({
        equations: [
          { ...first!, role: "backward" },
          { ...first!, id: "other", role: "none" },
        ],
      }),
      [],
    );
    expect(row.directions).toEqual(["backward", "none"]);
  });

  test.each([
    ["loss", "loss"],
    ["whole-model", "transformer"],
    ["initialization", "operations"],
    ["sampling", "generation"],
    ["gradient-checking", "testing"],
  ] as const)("category %s is filtered as %s", (category, filter) => {
    expect(buildReferenceRow(concept({ category }), []).filter).toBe(filter);
    expect(FILTER_LABELS[filter]).toBeTruthy();
  });

  test("the text filter covers names, equations, files and functions", () => {
    const { search } = buildReferenceRow(sample, []);
    for (const word of ["softmax", "normalize", "ops.py", "softmax_forward", "\\exp"]) {
      expect(search).toContain(word);
    }
  });
});

describe("links to pages", () => {
  const id = sample.id;

  test("no page means no link", () => {
    expect(
      conceptLinks(id, [{ slug: "a/b", body: 'text <ConceptCard concept="other" />' }]),
    ).toEqual({ explanation: undefined, example: undefined });
  });

  test("a page with the concept and its example gets both links", () => {
    const body = `<ConceptCard concept="${id}" />\n<NumericExample concept="${id}" kind="illustrative" />`;
    expect(conceptLinks(id, [{ slug: "a/b", body }])).toEqual({
      explanation: `/a/b/#${conceptAnchor(id)}`,
      example: `/a/b/#${exampleAnchor(id)}`,
    });
  });

  test("any component naming the concept counts as an explanation", () => {
    const pages = [{ slug: "a/b", body: `<Trace concept="${id}" />` }];
    expect(conceptLinks(id, pages).explanation).toBe("/a/b/");
    expect(conceptLinks(id, pages).example).toBeUndefined();
  });
});

describe("coverage", () => {
  const { rows } = loadInventory();

  test("the inventory snapshot has the rows of the contract", () => {
    expect(rows.length).toBeGreaterThan(0);
    expect(new Set(rows.map((row) => row.id)).size).toBe(rows.length);
    expect(rows.find((row) => row.id === "M21")?.name).toBe("Softmax");
  });

  test("is computed from the concepts, and follows them", () => {
    expect(coverage([], rows)).toEqual({ documented: 0, total: rows.length });
    expect(coverage([sample], rows).documented).toBe(sample.inventory.length);
    const added = concept({ id: "x", inventory: ["M19"] });
    expect(coverage([sample, added], rows).documented).toBe(sample.inventory.length + 1);
  });

  test("two concepts on one row count once, and ids outside the inventory do not count", () => {
    expect(coverage([sample, concept({ id: "x" })], rows).documented).toBe(sample.inventory.length);
    const unknown = concept({ inventory: ["M999"] });
    expect(coverage([unknown], rows).documented).toBe(0);
  });

  test("a concept naming a row that does not exist fails verification", () => {
    const problems = verifySiteData({
      symbols: loadSymbolIndex(),
      callGraph: loadCallGraph(),
      config: loadConfig(),
      navSlugs: navSlugs(),
      parameterNames: loadInitParams().parameters.map((parameter) => parameter.name),
      repoFiles: loadCodeIndex().files.map((file) => file.path),
      concepts: [concept({ inventory: ["M999"] })],
      inventoryIds: rows.map((row) => row.id),
    });
    expect(problems.join("\n")).toContain("M999 is not a row of the mathematics inventory");
  });
});

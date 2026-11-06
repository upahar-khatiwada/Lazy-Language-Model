import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import { CODE_REFS } from "@/data/code-refs";
import { ABSENT_TERMS, READER_NOTES, SCAN_EXEMPT, TASK_NAMED_TERMS } from "@/data/not-in-model";
import { SHAPE_STAGE_ALIASES } from "@/data/shape-stages";
import { loadSymbolIndex } from "@/lib/code-targets";
import { loadConcepts, type MathConcept } from "@/lib/concepts";
import {
  checkCoverage,
  checkEquations,
  checkForbiddenTopics,
  checkFreshness,
  checkLabels,
  checkShapes,
  checkSymbolsAndLinks,
  exitCode,
  formatResults,
  isPublic,
  jsxTagAround,
  termPattern,
  type CheckResult,
  type ShapeRuns,
} from "@/lib/content-checks";
import { publicSymbols, runContentChecks } from "@/lib/content-input";
import { inlineParts, loadAbsentTechniques, loadDiscrepancies } from "@/lib/contract-pages";
import { loadLinks } from "@/lib/links";
import { readPageSources } from "@/lib/reference";

const concepts = loadConcepts();
const symbols = loadSymbolIndex();
const softmax = concepts.find((concept) => concept.id === "ops.softmax_forward")!;
const copy = <T>(value: T): T => structuredClone(value);
const problems = (check: CheckResult) => check.problems.join("\n");

describe("the real documentation", () => {
  const results = runContentChecks();

  test("runs seven checks, and none fails", () => {
    expect(results.map((check) => check.id)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(results.flatMap((check) => check.problems)).toEqual([]);
    expect(exitCode(results)).toBe(0);
  });

  test("the coverage report states the uncovered functions", () => {
    const coverage = results.find((check) => check.id === 2)!;
    expect(coverage.report.at(-1)).toBe("Uncovered functions: none.");
    expect(coverage.report).toContain("ops: 12 of 12 referenced");
  });

  test("most shape stages are compared, in both runs", () => {
    const [line] = results.find((check) => check.id === 5)!.report;
    const [, checked, total] = /^(\d+) of (\d+) stages/.exec(line!)!;
    expect(Number(checked)).toBeGreaterThan(190);
    expect(Number(total) - Number(checked)).toBeLessThanOrEqual(10);
  });
});

describe("check 1: source symbols and link matches", () => {
  const good = { file: "ops.py", symbol: "ops.softmax_forward", equationId: "softmax.normalize" };
  const run = (concept: MathConcept, match: string, occurrence?: number) =>
    checkSymbolsAndLinks({
      concepts: [concept],
      symbols,
      linkFiles: [
        {
          path: "data/links/x.yaml",
          links: [{ ...good, match, ...(occurrence === undefined ? {} : { occurrence }) }],
        },
      ],
    });

  test("passes on a match that resolves", () => {
    expect(run(softmax, "return e / e.sum(axis=-1, keepdims=True)").problems).toEqual([]);
  });

  test("names a concept whose source symbol does not exist", () => {
    const broken = copy(softmax);
    broken.source[0]!.symbol = "ops.softmax_nope";
    expect(problems(run(broken, "return e"))).toContain(
      "data/concepts/ops.softmax_forward.yaml: source symbol does not exist: ops.softmax_nope",
    );
  });

  test("names a source symbol in the wrong file", () => {
    const broken = copy(softmax);
    broken.source[0]!.file = "train.py";
    expect(problems(run(broken, "return e"))).toContain("is in ops.py, not train.py");
  });

  test("points to the file, the equation and the text of a broken match", () => {
    const text = problems(run(softmax, "return e / e.total()"));
    expect(text).toContain("data/links/x.yaml");
    expect(text).toContain("softmax.normalize");
    expect(text).toContain('match "return e / e.total()" not found');
  });

  test("rejects a match with several ranges and no declared occurrence", () => {
    expect(problems(run(softmax, "e"))).toContain("no occurrence index");
  });

  test("accepts a declared occurrence, and rejects one out of range", () => {
    expect(run(softmax, "keepdims", 1).problems).toEqual([]);
    expect(problems(run(softmax, "keepdims", 9))).toContain("occurrence 9 is out of range");
  });
});

describe("check 2: coverage", () => {
  const real = publicSymbols();
  const base = {
    symbols: real,
    concepts,
    links: loadLinks() as { symbol: string }[],
    pages: readPageSources(),
    codeRefs: CODE_REFS as Record<string, { symbol: string }>,
    allowlist: [] as { symbol: string; reason: string }[],
  };
  const ghost = { id: "ops.ghost_function", name: "ghost_function", module: "ops" };

  test("leaves out names with a leading underscore", () => {
    expect(isPublic("ops.softmax_forward")).toBe(true);
    expect(isPublic("optim.Adam.step")).toBe(true);
    expect(isPublic("optim.Adam.__init__")).toBe(false);
    expect(real.map((symbol) => symbol.id)).not.toContain("optim.Adam.__init__");
    expect(real.map((symbol) => symbol.id)).toContain("optim.Adam.step");
  });

  test("fails on a public function nothing references, and lists it in the report", () => {
    const check = checkCoverage({ ...base, symbols: [...real, ghost] });
    expect(problems(check)).toContain(
      "ops.ghost_function is not referenced by any concept or page",
    );
    expect(check.report.at(-1)).toBe("Uncovered functions (1): ops.ghost_function");
    expect(exitCode([check])).toBe(1);
  });

  test("counts a concept, a link, a symbol attribute, a quoted line and inline code", () => {
    const fake = (name: string) => ({ id: `ops.${name}`, name, module: "ops" });
    const run = (name: string, parts: Partial<typeof base>) =>
      checkCoverage({
        ...base,
        symbols: [fake(name)],
        concepts: [],
        links: [],
        pages: [],
        ...parts,
      }).problems;
    const conceptUsing = copy(softmax);
    conceptUsing.source = [{ file: "ops.py", symbol: "ops.a" }];
    expect(run("a", { concepts: [conceptUsing] })).toEqual([]);
    expect(run("b", { links: [{ symbol: "ops.b" }] })).toEqual([]);
    expect(run("c", { pages: [{ slug: "p", body: '<CodeReference symbol="ops.c" />' }] })).toEqual(
      [],
    );
    expect(
      run("d", {
        pages: [{ slug: "p", body: '<CodeLine id="x" />' }],
        codeRefs: { x: { symbol: "ops.d" } },
      }),
    ).toEqual([]);
    expect(run("e", { pages: [{ slug: "p", body: "calls `e(x)` here" }] })).toEqual([]);
    expect(run("f", { pages: [{ slug: "p", body: "the word f in prose" }] })).toHaveLength(1);
  });

  test("a bare name that two modules share counts for neither", () => {
    const both = [
      { id: "train.main", name: "main", module: "train" },
      { id: "talk.main", name: "main", module: "talk" },
    ];
    const run = (body: string) =>
      checkCoverage({
        ...base,
        symbols: both,
        concepts: [],
        links: [],
        pages: [{ slug: "p", body }],
      });
    expect(run("`main`").problems).toHaveLength(2);
    expect(run("`train.main` and `talk.main`").problems).toEqual([]);
  });

  test("an allowlisted function passes, and the report prints its reason", () => {
    const check = checkCoverage({
      ...base,
      symbols: [...real, ghost],
      allowlist: [{ symbol: ghost.id, reason: "Kept for a test." }],
    });
    expect(check.problems).toEqual([]);
    expect(check.report).toContain("allowlisted ops.ghost_function: Kept for a test.");
  });

  test("fails on an allowlist entry with no reason, or an unknown symbol", () => {
    expect(
      problems(
        checkCoverage({
          ...base,
          symbols: [...real, ghost],
          allowlist: [{ symbol: ghost.id, reason: " " }],
        }),
      ),
    ).toContain("has no written reason");
    expect(
      problems(checkCoverage({ ...base, allowlist: [{ symbol: "ops.nope", reason: "x" }] })),
    ).toContain("unknown public function: ops.nope");
  });
});

describe("check 3: equations", () => {
  test("passes on an equation that exists", () => {
    const pages = [{ slug: "mathematics/x", body: 'one\n<MathBlock id="softmax.shift" />\n' }];
    expect(checkEquations({ concepts, pages }).problems).toEqual([]);
  });

  test("fails on an unknown id, with the page and line", () => {
    const pages = [{ slug: "mathematics/x", body: 'one\n<MathBlock id="softmax.nope" />\n' }];
    expect(problems(checkEquations({ concepts, pages }))).toBe(
      'content/mathematics/x.mdx:2: equation id "softmax.nope" does not exist',
    );
  });

  test("also reads the equation attribute of other components", () => {
    const pages = [{ slug: "x", body: '<MathCodeLink equation="softmax.nope" />' }];
    expect(problems(checkEquations({ concepts, pages }))).toContain("softmax.nope");
  });

  test("fails on the same LaTeX in two concepts, even with different spacing", () => {
    const other = copy(concepts.find((concept) => concept.id === "ops.relu_forward")!);
    other.equations[0]!.latex = softmax.equations[0]!.latex.replaceAll(" ", "  ");
    const text = problems(checkEquations({ concepts: [softmax, other], pages: [] }));
    expect(text).toContain("duplicated equation LaTeX");
    expect(text).toContain("ops.softmax_forward (softmax.shift)");
    expect(text).toContain("ops.relu_forward");
  });
});

describe("check 4: labels", () => {
  const run = (text: string, path = "content/x.mdx", extra = {}) =>
    checkLabels({
      concepts,
      sources: [{ path, text }],
      actualBadgeFiles: [],
      provenance: null,
      ...extra,
    });

  test("passes on a NumericExample that agrees with its data", () => {
    expect(
      run('<NumericExample concept="ops.softmax_forward" kind="illustrative" />').problems,
    ).toEqual([]);
  });

  test("fails on a NumericExample with no kind", () => {
    expect(problems(run('\n<NumericExample concept="ops.softmax_forward" />'))).toBe(
      "content/x.mdx:2: NumericExample does not declare a kind",
    );
  });

  test("fails on an unknown kind, and on a kind the data contradicts", () => {
    expect(
      problems(run('<NumericExample concept="ops.softmax_forward" kind="made-up" />')),
    ).toContain('unknown kind "made-up"');
    expect(
      problems(run('<NumericExample concept="ops.softmax_forward" kind="actual" />')),
    ).toContain("says actual, the data says illustrative");
  });

  test("fails on an Actual panel with no provenance, and passes with one", () => {
    const bad = `<Heatmap\n  title="x"\n  kind="actual"\n  values={[1 > 0]}\n/>`;
    expect(problems(run(bad, "components/X.tsx"))).toBe(
      "components/X.tsx:3: Actual Model Output panel <Heatmap> has no provenance",
    );
    const good = `<Heatmap title="x" kind="actual" values={[1 > 0]} provenance={line} />`;
    expect(run(good, "components/X.tsx").problems).toEqual([]);
  });

  test("a bare Actual badge passes only in a file that is listed", () => {
    const text = '<Badge kind="actual" />';
    expect(problems(run(text, "components/X.tsx"))).toContain("badge outside a panel");
    const allowed = { actualBadgeFiles: [{ path: "components/X.tsx", reason: "r" }] };
    expect(run(text, "components/X.tsx", allowed).problems).toEqual([]);
  });

  test("fails on an actual example in data with no provenance", () => {
    const broken = copy(softmax);
    broken.example = { ...broken.example!, kind: "actual", provenance: " " } as never;
    expect(problems(run("", "content/x.mdx", { concepts: [broken] }))).toContain(
      "data/concepts/ops.softmax_forward.yaml: Actual Model Output example has no provenance",
    );
  });

  test("fails on exported provenance with no commit or no hash", () => {
    const provenance = { gitSha: "abc", checkpointFiles: [{ name: "m", sha256: "h" }], seed: 42 };
    expect(run("", "content/x.mdx", { provenance }).problems).toEqual([]);
    expect(
      problems(run("", "content/x.mdx", { provenance: { ...provenance, gitSha: "" } })),
    ).toContain("provenance has no commit");
    expect(
      problems(run("", "content/x.mdx", { provenance: { ...provenance, checkpointFiles: [] } })),
    ).toContain("provenance has no checkpoint hash");
  });

  test("jsxTagAround finds the whole tag past a > inside braces, and ignores plain text", () => {
    const text = '<Chart a={b > c} kind="actual" d="e" /> after';
    expect(jsxTagAround(text, text.indexOf("kind"))).toEqual({
      name: "Chart",
      tag: '<Chart a={b > c} kind="actual" d="e" />',
    });
    expect(jsxTagAround('no tag kind="actual"', 8)).toBeNull();
  });
});

describe("check 5: tensor shapes", () => {
  const { runs } = JSON.parse(readFileSync("data/golden/shapes.json", "utf-8")) as ShapeRuns;
  const attention = concepts.find((concept) => concept.id === "attention.softmax")!;
  const configDims = { d: runs.config!.dims.d!, T: runs.config!.dims.T! };
  const base = {
    runs,
    configDims,
    aliases: SHAPE_STAGE_ALIASES,
    requiredFunctions: ["transformer.attention_forward"],
  };

  test("the tiny run's sizes are all different, so a swap cannot go unnoticed", () => {
    const { V, Tmax, T, d, dh, F, H, B } = runs.tiny!.dims;
    const sizes = [V, Tmax, T, d, dh, F, H, B];
    expect(new Set(sizes).size).toBe(sizes.length);
  });

  test("passes on the stored shapes", () => {
    expect(checkShapes({ ...base, concepts: [attention] }).problems).toEqual([]);
  });

  test("fails when a stage's dims disagree with the run, naming the values", () => {
    const broken = copy(attention);
    broken.shapes[0]!.stages[0]!.dims = ["T", "H", "T"];
    const text = problems(checkShapes({ ...base, concepts: [broken] }));
    expect(text).toContain("data/concepts/attention.softmax.yaml");
    expect(text).toContain("(T, H, T) at the tiny config is (5, 2, 5)");
    expect(text).toContain("transformer.attention_forward recorded S as (2, 5, 5)");
    expect(text).toContain("(T, H, T) at the config config is (16, 2, 16)");
  });

  test("fails when config.py and the stored run have drifted apart", () => {
    const check = checkShapes({ ...base, concepts: [], configDims: { d: 64 } });
    expect(problems(check)).toContain("config.py gives d = 64, the run used 32");
  });

  test("fails when a run or a required function is missing", () => {
    expect(
      problems(checkShapes({ ...base, concepts: [], runs: { config: runs.config! } })),
    ).toContain("has no tiny run");
    const empty = { tiny: { ...runs.tiny!, functions: {} }, config: runs.config! };
    expect(problems(checkShapes({ ...base, concepts: [], runs: empty }))).toContain(
      "tiny run has no transformer.attention_forward",
    );
  });

  test("reports a stage with no recorded tensor, and does not pretend to have checked it", () => {
    const made = copy(attention);
    made.shapes[0]!.stages[0]!.name = "no such tensor";
    const check = checkShapes({ ...base, concepts: [made] });
    expect(check.report.join("\n")).toContain("no such tensor");
    expect(check.report[0]).toContain("1 of 2 stages");
  });
});

describe("check 6: freshness", () => {
  const sha = "a".repeat(40);
  const other = "b".repeat(40);
  const base = { repoMetaSha: sha, codeIndexSha: sha, actualSha: sha, headSha: sha };

  test("has no warning when everything is at HEAD", () => {
    expect(checkFreshness(base).warnings).toEqual([]);
  });

  test("warns on generated data behind HEAD, and does not fail the command", () => {
    const check = checkFreshness({ ...base, headSha: other });
    expect(check.warnings.join("\n")).toContain("data/generated is stale: pinned to aaaaaaa");
    expect(check.warnings.join("\n")).toContain("data/generated/actual is stale");
    expect(check.problems).toEqual([]);
    expect(exitCode([check])).toBe(0);
    expect(formatResults([check])[0]).toMatch(/^\[WARN\]/);
  });

  test("warns when the generated files disagree, or when git cannot be asked", () => {
    expect(checkFreshness({ ...base, codeIndexSha: other }).warnings.join()).toContain(
      "disagrees with itself",
    );
    expect(checkFreshness({ ...base, headSha: null }).warnings.join()).toContain("Could not read");
  });

  test("a site with no export is not reported as stale", () => {
    expect(checkFreshness({ ...base, actualSha: null }).warnings).toEqual([]);
  });
});

describe("check 7: forbidden topics", () => {
  const absentRows = [{ technique: "GELU" }, { technique: "Dropout" }];
  const base = {
    absentRows,
    absentTerms: { GELU: ["GELU"], Dropout: ["dropout"] },
    taskNamedTerms: { RMSNorm: ["RMSNorm"] },
    sources: [],
    repoFiles: [],
    exempt: [{ path: "content/reference/not-in-this-model.mdx", reason: "r" }],
  };
  const scan = (text: string, path = "content/x.mdx") =>
    checkForbiddenTopics({ ...base, sources: [{ path, text }] });

  test("passes on text with none of the words", () => {
    expect(scan("The model uses ReLU.").problems).toEqual([]);
  });

  test("fails with the file, the line, the word and the technique", () => {
    expect(problems(scan("one\ntwo\nAfter a gelu layer"))).toBe(
      'content/x.mdx:3: "gelu" names GELU, which this repository does not use. Move it to the Not in This Model page',
    );
  });

  test("also catches a word the task names that the contract does not", () => {
    expect(problems(scan("uses rmsnorm"))).toContain('"rmsnorm" names RMSNorm');
  });

  test("allows the words on the Not in This Model page", () => {
    expect(scan("GELU and dropout", "content/reference/not-in-this-model.mdx").problems).toEqual(
      [],
    );
  });

  test("a folder can be exempt, but a file with a similar name is not", () => {
    const folder = { ...base, exempt: [{ path: "data/derived/", reason: "r" }] };
    const run = (path: string) =>
      checkForbiddenTopics({ ...folder, sources: [{ path, text: "GELU" }] }).problems;
    expect(run("data/derived/x.json")).toEqual([]);
    expect(run("data/derived.ts")).toHaveLength(1);
  });

  test("fails when a word turns up in the repository, so the row may be wrong", () => {
    const check = checkForbiddenTopics({
      ...base,
      repoFiles: [{ path: "ops.py", text: "\ndef gelu(x): ..." }],
    });
    expect(problems(check)).toContain('ops.py:2: "gelu" is in the repository');
  });

  test("fails when a contract row has no words, or words have no contract row", () => {
    expect(
      problems(
        checkForbiddenTopics({ ...base, absentRows: [...absentRows, { technique: "New one" }] }),
      ),
    ).toContain('lists "New one" as absent');
    expect(
      problems(checkForbiddenTopics({ ...base, absentRows: [{ technique: "GELU" }] })),
    ).toContain('scan words for "Dropout"');
  });

  test("words match whole words, and tolerate a space or a hyphen", () => {
    const hits = (term: string, text: string) => [...text.matchAll(termPattern(term))].length;
    expect(hits("BOS", "the bos token")).toBe(1);
    expect(hits("BOS", "bosom")).toBe(0);
    expect(hits("lower-cases", "lower cases")).toBe(1);
    expect(hits("lower-cases", "toLowerCase")).toBe(0);
    expect(hits("XOR", "np.bitwise_xor")).toBe(0);
    expect(hits("N(0, 0.02", "drawn from N(0, 0.02^2)")).toBe(1);
  });

  test("the words cover exactly the contract's confirmed-absent rows", () => {
    expect(Object.keys(ABSENT_TERMS).sort()).toEqual(
      loadAbsentTechniques()
        .rows.map((row) => row.technique)
        .sort(),
    );
  });

  test("the terms the task named but the contract does not stay off that page's data", () => {
    const rows = loadAbsentTechniques().rows.map((row) => row.technique);
    for (const technique of Object.keys(TASK_NAMED_TERMS)) expect(rows).not.toContain(technique);
    expect(SCAN_EXEMPT.every((entry) => entry.reason.length > 0)).toBe(true);
  });

  test("the confirmed-absent list holds only rows that say In code: no", () => {
    const rows = loadAbsentTechniques().rows.map((row) => row.technique);
    expect(rows).toContain("Dropout");
    expect(rows).toContain("GELU");
    expect(rows).toContain("Beam search, repetition penalty");
    // `partly` and `test` rows of the contract.
    expect(rows).not.toContain("Plain Adam at constant LR");
    expect(rows).not.toContain("Overfit one batch");
    expect(rows).not.toContain("ReLU");
  });
});

describe("the command's result", () => {
  const pass: CheckResult = { id: 1, title: "A", problems: [], warnings: [], report: ["r"] };
  const fail: CheckResult = { id: 2, title: "B", problems: ["bad"], warnings: [], report: [] };

  test("any failure gives a non-zero exit", () => {
    expect(exitCode([pass])).toBe(0);
    expect(exitCode([pass, fail])).toBe(1);
  });

  test("the output names each check and its problems", () => {
    expect(formatResults([pass, fail])).toEqual([
      "[PASS] 1. A",
      "       r",
      "[FAIL] 2. B",
      "       problem: bad",
      "FAILED: 1 of 2 checks failed.",
    ]);
    expect(formatResults([pass]).at(-1)).toBe("OK: 1 checks passed, 0 with warnings.");
  });
});

describe("the two reference pages", () => {
  const pages = readPageSources();
  const sources = pages.find((page) => page.slug === "reference/sources-discrepancies")!;
  const absent = pages.find((page) => page.slug === "reference/not-in-this-model")!;

  test("Sources & Discrepancies has each contract section under the contract's own heading", () => {
    const { sections } = loadDiscrepancies();
    expect(sections.map((section) => section.number)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    for (const section of sections) {
      expect(sources.body).toContain(
        `## ${section.title}\n\n<DiscrepancySection number={${section.number}} />`,
      );
    }
  });

  test("the snapshot keeps every table row of the contract, and escaped pipes in cells", () => {
    const formulas = loadDiscrepancies().sections.find((section) => section.number === 4)!;
    const table = formulas.blocks.find((block) => block.type === "table");
    if (table?.type !== "table") throw new Error("section 4 has no table");
    expect(table.rows[0]![2]).toContain("\\w+|\\s|[^\\w\\s]");
    expect(table.rows.every((row) => row.length === table.headers.length)).toBe(true);
  });

  test("every reader note belongs to a contract row, and none tells the author what to do", () => {
    const rows = loadAbsentTechniques().rows;
    for (const technique of Object.keys(READER_NOTES)) {
      expect(rows.map((row) => row.technique)).toContain(technique);
    }
    for (const row of rows) {
      expect(READER_NOTES[row.technique] ?? row.note).not.toMatch(/^(Do not|May )/);
    }
  });

  test("Not in This Model renders the list and does not retype it", () => {
    expect(absent.body).toContain("<AbsentTechniques />");
    expect(absent.body).not.toContain("| ");
  });

  test("inlineParts reads code and bold, and drops no text", () => {
    expect(inlineParts("a `b**c` **d** e")).toEqual([
      { text: "a ", style: "plain" },
      { text: "b**c", style: "code" },
      { text: " ", style: "plain" },
      { text: "d", style: "bold" },
      { text: " e", style: "plain" },
    ]);
  });
});

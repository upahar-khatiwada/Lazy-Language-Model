import { execFileSync } from "node:child_process";
import { describe, expect, test } from "vitest";
import {
  loadSymbolIndex,
  requireTarget,
  resolveTarget,
  type CodeSymbol,
  type SymbolIndex,
} from "@/lib/code-targets";
import { loadConcepts, loadPipelines, type MathConcept } from "@/lib/concepts";
import { loadStoredExamples } from "@/lib/examples";
import { highlightedLines, LinkSchema, loadLinks, type Link } from "@/lib/links";
import { loadRepoMeta } from "@/lib/repo";
import { verifyData, type VerifyInput } from "@/lib/verify-links";

const fn: CodeSymbol = {
  id: "m.fn",
  name: "fn",
  file: "m.py",
  startLine: 10,
  endLine: 14,
  source: "def fn(x):\n    y = x + 1\n    y = x + 1\n    z = (y +\n         1)\n",
  permalink: "https://example.test/blob/abc/m.py#L10-L14",
};
const symbols: SymbolIndex = new Map([[fn.id, fn]]);

const target = (overrides: Partial<Parameters<typeof resolveTarget>[0]>) => ({
  file: "m.py",
  symbol: "m.fn",
  match: "def fn",
  ...overrides,
});

function problemFor(overrides: Partial<Parameters<typeof resolveTarget>[0]>): string {
  const result = resolveTarget(target(overrides), symbols);
  if (result.ok) throw new Error("expected a problem");
  return result.problem;
}

describe("resolving a code target", () => {
  test("a substring resolves to the real line in the file", () => {
    const { startLine, endLine, permalink } = requireTarget(target({ match: "def fn" }), symbols);
    expect([startLine, endLine]).toEqual([10, 10]);
    expect(permalink).toBe("https://example.test/blob/abc/m.py#L10");
  });

  test("an occurrence index selects one of several matches", () => {
    const pick = (occurrence: number) =>
      requireTarget(target({ match: "y = x + 1", occurrence }), symbols).startLine;
    expect([pick(0), pick(1)]).toEqual([11, 12]);
  });

  test("a regex may span several lines and gives a range", () => {
    const { startLine, endLine, permalink } = requireTarget(
      target({ match: String.raw`z = \(y \+\s+1\)`, regex: true }),
      symbols,
    );
    expect([startLine, endLine]).toEqual([13, 14]);
    expect(permalink).toMatch(/#L13-L14$/);
  });

  test("a symbol that does not exist fails", () => {
    expect(problemFor({ symbol: "m.missing" })).toContain("symbol does not exist");
  });

  test("a file that does not hold the symbol fails", () => {
    expect(problemFor({ file: "other.py" })).toContain("is in m.py, not other.py");
  });

  test("a match that is not in the code fails", () => {
    expect(problemFor({ match: "not in the code" })).toContain("not found");
  });

  test("text outside the symbol does not count as a match", () => {
    expect(problemFor({ match: "def other" })).toContain("not found");
  });

  test("an ambiguous match without an occurrence fails", () => {
    expect(problemFor({ match: "y = x + 1" })).toContain("ambiguous: 2 matches");
  });

  test("an occurrence out of range fails", () => {
    expect(problemFor({ match: "y = x + 1", occurrence: 2 })).toContain("out of range");
  });

  test("an invalid regex or one that can match nothing fails", () => {
    expect(problemFor({ match: "(", regex: true })).toContain("invalid regex");
    expect(problemFor({ match: "q*", regex: true })).toContain("can match empty text");
  });
});

const concept = {
  id: "c",
  title: "C",
  category: "basic-operations",
  status: "derived",
  summary: "s",
  why: "w",
  equations: [{ id: "e1", latex: "x", role: "forward", label: "l" }],
  symbols: [],
  example: null,
  shapes: [],
  source: [{ file: "m.py", symbol: "m.fn" }],
  usedBy: [],
  pdfRef: null,
  notes: [],
} satisfies MathConcept;

const link: Link = {
  equationId: "e1",
  file: "m.py",
  symbol: "m.fn",
  match: "def fn",
  role: "implements",
};

const base: VerifyInput = {
  concepts: [concept],
  links: [link],
  pipelines: [],
  symbols,
  examples: {},
};

describe("verifying the data", () => {
  test("consistent data has no problems", () => {
    expect(verifyData(base)).toEqual([]);
  });

  test.each<[string, Partial<VerifyInput>, string]>([
    ["missing symbol", { links: [{ ...link, symbol: "m.nope" }] }, "symbol does not exist"],
    ["missing match", { links: [{ ...link, match: "nope" }] }, "not found"],
    ["ambiguous match", { links: [{ ...link, match: "y = x + 1" }] }, "ambiguous"],
    ["unknown equation", { links: [link, { ...link, equationId: "no" }] }, "equation does not"],
    ["unlinked equation", { links: [] }, "has no link to code"],
    ["call site only", { links: [{ ...link, role: "call" }] }, "has no link to code"],
    ["duplicate equation", { concepts: [concept, { ...concept, id: "d" }] }, "duplicate equation"],
    [
      "bad LaTeX",
      {
        concepts: [
          { ...concept, equations: [{ id: "e1", latex: "\\frac{", role: "forward", label: "l" }] },
        ],
      },
      "equation e1",
    ],
    [
      "bad source symbol",
      { concepts: [{ ...concept, source: [{ file: "m.py", symbol: "m.nope" }] }] },
      "source symbol does not exist",
    ],
    [
      "bad usedBy",
      {
        concepts: [
          { ...concept, usedBy: [{ file: "m.py", symbol: "m.fn", match: "nope", note: "n" }] },
        ],
      },
      "usedBy",
    ],
    [
      "bad pipeline target",
      {
        pipelines: [
          {
            id: "p",
            title: "P",
            steps: [
              { label: "a", target: { file: "m.py", symbol: "m.fn", match: "def fn" } },
              { label: "b", target: { file: "m.py", symbol: "m.fn", match: "nope" } },
            ],
          },
        ],
      },
      "pipeline p step 2",
    ],
    [
      "orphan example",
      {
        examples: {
          other: { kind: "illustrative", call: "m.fn", input: {}, steps: {}, output: 1 },
        },
      },
      "has no concept",
    ],
  ])("%s is reported", (_name, overrides, expected) => {
    const problems = verifyData({ ...base, ...overrides });
    expect(problems.join("\n")).toContain(expected);
  });

  const withExample: MathConcept = {
    ...concept,
    example: {
      kind: "illustrative",
      call: "m.fn",
      input: { x: [1] },
      steps: [{ id: "s", label: "l", equationId: "e1", capture: "y" }],
      output: { label: "o" },
      precision: 2,
    },
  };
  const stored = {
    kind: "illustrative" as const,
    call: "m.fn",
    input: { x: [1] },
    steps: { s: [2] },
    output: [2],
  };

  test("a stored example must agree with its concept", () => {
    const check = (examples: VerifyInput["examples"]) =>
      verifyData({ ...base, concepts: [withExample], examples }).join("\n");

    expect(check({ c: stored })).toBe("");
    expect(check({})).toContain("no computed example");
    expect(check({ c: { ...stored, input: { x: [9] } } })).toContain("stale");
    expect(check({ c: { ...stored, kind: "actual" } })).toContain("stale");
    expect(check({ c: { ...stored, steps: { other: [2] } } })).toContain("stale");
  });

  test("the real repository data is consistent", () => {
    expect(
      verifyData({
        concepts: loadConcepts(),
        links: loadLinks(),
        pipelines: loadPipelines(),
        symbols: loadSymbolIndex(),
        examples: loadStoredExamples(),
      }),
    ).toEqual([]);
  });
});

test("a link record cannot store a line number", () => {
  expect(LinkSchema.safeParse({ ...link, line: 23 }).success).toBe(false);
  expect(LinkSchema.safeParse({ ...link, startLine: 23 }).success).toBe(false);
});

describe("real links against the pinned commit", () => {
  const { sha } = loadRepoMeta();
  const lineAt = (file: string, line: number) =>
    execFileSync("git", ["show", `${sha}:${file}`], { encoding: "utf-8", maxBuffer: 1 << 24 })
      .split(/\r?\n/)
      [line - 1]!.trim();

  test("every resolved link points at a line holding its match", () => {
    for (const entry of loadLinks()) {
      const { startLine, symbol } = requireTarget(entry);
      expect(lineAt(symbol.file, startLine), entry.match).toContain(entry.match);
    }
  });

  test("highlighted lines hold exactly the linked code", () => {
    const lines = [...highlightedLines("ops.softmax_forward")].map((line) =>
      lineAt("ops.py", line),
    );
    expect(lines).toEqual([
      "e = np.exp(x - np.max(x, axis=-1, keepdims=True))",
      "return e / e.sum(axis=-1, keepdims=True)",
    ]);
  });

  test("an equation filter narrows the highlight, and an inline use is highlighted", () => {
    const only = [...highlightedLines("ops.softmax_forward", "softmax.shift")];
    expect(only.map((line) => lineAt("ops.py", line))).toEqual([
      "e = np.exp(x - np.max(x, axis=-1, keepdims=True))",
    ]);
    const inline = [...highlightedLines("ops.cross_entropy_forward")];
    expect(inline.map((line) => lineAt("ops.py", line))).toEqual([
      "z = logits - logits.max(axis=-1, keepdims=True)",
    ]);
  });
});

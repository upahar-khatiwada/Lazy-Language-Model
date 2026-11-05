import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { parse } from "yaml";
import { equationAnchor, getConcept, getEquation, MathConceptSchema } from "@/lib/concepts";
import { listContentSlugs } from "@/lib/content";
import { formatShape, resolveDimension } from "@/lib/dimensions";
import { formatNumbers, getStoredExample } from "@/lib/examples";

const sample: Record<string, unknown> = parse(
  readFileSync(join(process.cwd(), "data", "concepts", "ops.softmax_forward.yaml"), "utf-8"),
);

const isValid = (changes: Record<string, unknown>) =>
  MathConceptSchema.safeParse({ ...sample, ...changes }).success;

describe("the MathConcept schema", () => {
  test("accepts the sample concept", () => {
    expect(isValid({})).toBe(true);
  });

  test("rejects stored line numbers anywhere", () => {
    expect(isValid({ line: 23 })).toBe(false);
    expect(isValid({ source: [{ file: "ops.py", symbol: "ops.x", startLine: 22 }] })).toBe(false);
  });

  test("an implemented concept must cite the PDF; a derived one need not", () => {
    expect(isValid({ pdfRef: null })).toBe(false);
    expect(isValid({ pdfRef: null, status: "derived" })).toBe(true);
  });

  test("an Actual Model Output example must state its provenance", () => {
    const example = sample.example as Record<string, unknown>;
    expect(isValid({ example: { ...example, kind: "actual" } })).toBe(false);
    expect(isValid({ example: { ...example, kind: "actual", provenance: "a run" } })).toBe(true);
    expect(isValid({ example: { ...example, kind: undefined } })).toBe(false);
    expect(isValid({ example: { ...example, kind: "guess" } })).toBe(false);
  });

  test("an unknown dimension name is rejected", () => {
    const shapes = [
      {
        id: "x",
        label: "x",
        stages: [
          { name: "a", dims: ["batch"] },
          { name: "b", dims: ["d"] },
        ],
        target: { file: "ops.py", symbol: "ops.x", match: "x" },
      },
    ];
    expect(isValid({ shapes })).toBe(false);
  });
});

test("lookups fail loudly on unknown ids", () => {
  expect(() => getConcept("nope")).toThrow("Unknown concept");
  expect(() => getEquation("nope")).toThrow("Unknown equation");
  expect(getEquation("softmax.shift").concept.id).toBe("ops.softmax_forward");
  expect(equationAnchor("softmax.shift")).toBe("eq-softmax-shift");
});

test("hand-written data contains no line-number keys", () => {
  for (const folder of ["concepts", "links", "pipelines"]) {
    const dir = join(process.cwd(), "data", folder);
    for (const name of readdirSync(dir).filter((file) => file.endsWith(".yaml"))) {
      const text = readFileSync(join(dir, name), "utf-8");
      expect(text, name).not.toMatch(/^\s*-?\s*(line|lines|startLine|endLine|lineNumber)\s*:/m);
    }
  }
});

describe("tensor dimensions", () => {
  test("sizes come from config.json", () => {
    expect(resolveDimension("d")).toMatchObject({ configName: "emb_dim", value: 32 });
    expect(resolveDimension("H")).toMatchObject({ configName: "num_heads", value: 2 });
    expect(resolveDimension("dh")).toMatchObject({ configName: "head_dim", value: 16 });
    expect(resolveDimension("T")).toMatchObject({ configName: "seq_len", value: 16 });
  });

  test("V has no invented value", () => {
    const vocabulary = resolveDimension("V");
    expect(vocabulary.value).toBeNull();
    expect(vocabulary.note).toContain("depends on the corpus");
  });

  test("shapes print as Python tuples", () => {
    expect(formatShape(["H", "T", "T"])).toBe("(H, T, T)");
    expect(formatShape(["V"])).toBe("(V,)");
  });
});

test("the stored example agrees with test_softmax_values_and_stability", () => {
  const { output, input } = getStoredExample("ops.softmax_forward");
  expect(input).toEqual({ x: [2, 1, 0] });
  expect(formatNumbers(output, 4)).toBe("[0.6652, 0.2447, 0.0900]");
});

test("samples get no route of their own", () => {
  expect(listContentSlugs().some((slug) => slug.includes("_samples"))).toBe(false);
});

test("MDX content cannot inline equations", () => {
  const dir = join(process.cwd(), "content");
  const files = readdirSync(dir, { recursive: true, encoding: "utf-8" }).filter((file) =>
    file.endsWith(".mdx"),
  );
  expect(files.length).toBeGreaterThan(0);
  for (const file of files) {
    const text = readFileSync(join(dir, file), "utf-8");
    expect(text, file).not.toMatch(/\$\$|\btex=|<TexDisplay|<CodeBlock|\\frac/);
  }
});

import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import {
  LAB_STEPS,
  LAB_TOKENS,
  maskedCells,
  runLab,
  stepMatrix,
  weightedTerms,
  type LabInputs,
} from "@/lib/attention-lab";
import { getSymbol, targetLines } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { resolveDimension, type DimensionName } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { highlightedLines, linksForEquation } from "@/lib/links";
import type { Matrix, Tensor3, Vector } from "@/lib/math/tensor";
import { loadConfig } from "@/lib/repo";

const body = readPageBody("mathematics/attention");
const SYMBOL = "transformer.attention_forward";
const golden = getGoldenCase("attention", "attention_forward.cat_sat");
const run = runLab(golden.inputs as LabInputs);
const out = golden.outputs as Record<string, Tensor3 & Matrix>;

const FORWARD_CONCEPTS = [
  "attention.projections",
  "attention.split",
  "attention.scores",
  "attention.scale",
  "attention.mask",
  "attention.softmax",
  "attention.weighted",
  "attention.merge",
  "attention.output",
];
const CONCEPTS = [...FORWARD_CONCEPTS, "transformer.attention_backward"];

const sum = (row: Vector) => row.reduce((total, value) => total + value, 0);

function expectClose(actual: number[][], expected: number[][]) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((row, i) =>
    row.forEach((value, j) => {
      const want = expected[i]![j]!;
      if (want === -Infinity) expect(value).toBe(-Infinity);
      else expect(Math.abs(value - want)).toBeLessThan(1e-9);
    }),
  );
}

// The text of a real line of the file a symbol lives in.
function lineText(symbolId: string, line: number): string {
  const symbol = getSymbol(symbolId);
  return symbol.source.split(/\r?\n/)[line - symbol.startLine]!;
}

describe("the lab equals the real attention_forward", () => {
  test("there is one token per row of the input", () => {
    expect(LAB_TOKENS).toEqual("The cat sat on the mat".split(" "));
    expect(LAB_TOKENS.length).toBe((golden.inputs.Xn as Matrix).length);
  });

  test.each([0, 1])("every step of head %i", (head) => {
    expectClose(stepMatrix(run, "scale", head), out.scaled![head]!);
    expectClose(stepMatrix(run, "mask", head), out.masked![head]!);
    expectClose(stepMatrix(run, "softmax", head), out.A![head]!);
    expectClose(stepMatrix(run, "output", head), out.A![head]!);
  });

  test.each([0, 1])("scores divided by the scale give the scaled scores, head %i", (head) => {
    const dh = run.Vh[head]![0]!.length;
    const raw = stepMatrix(run, "scores", head).map((row) => row.map((v) => v / Math.sqrt(dh)));
    expectClose(raw, out.scaled![head]!);
  });

  test("O and out equal the stored results", () => {
    expectClose(run.O, out.O!);
    expectClose(run.out, out.out!);
  });

  test.each([0, 1])("rows of head %i sum to 1", (head) => {
    for (const row of stepMatrix(run, "softmax", head)) {
      expect(Math.abs(sum(row) - 1)).toBeLessThan(1e-12);
    }
  });

  test.each([0, 1])("masking in head %i: above the diagonal only", (head) => {
    const masked = maskedCells(run, head);
    masked.forEach((row, i) => row.forEach((off, j) => expect(off).toBe(j > i)));
  });

  test.each([0, 1])("masked cells of head %i are -inf, then exactly 0", (head) => {
    const before = stepMatrix(run, "mask", head);
    const after = stepMatrix(run, "softmax", head);
    before.forEach((row, i) =>
      row.forEach((value, j) => {
        if (j > i) {
          expect(value).toBe(-Infinity);
          expect(after[i]![j]).toBe(0);
        } else {
          expect(Number.isFinite(value)).toBe(true);
          expect(after[i]![j]!).toBeGreaterThan(0);
        }
      }),
    );
  });

  test("the first token only reads itself, in each head", () => {
    expect(run.A[0]![0]![0]).toBe(1);
    expect(run.A[1]![0]![0]).toBe(1);
  });

  test("the example is not degenerate: later rows spread their weight", () => {
    for (const head of run.A) {
      for (const row of head.slice(2)) expect(Math.max(...row)).toBeLessThan(0.9);
    }
  });

  test.each([0, 1])("the weighted terms of head %i add up to the merged values", (head) => {
    LAB_TOKENS.forEach((_, row) => {
      const { terms, result } = weightedTerms(run, head, row);
      expect(terms.length).toBe(row + 1);
      result.forEach((value, c) => {
        const total = sum(terms.map((term) => term.weight * term.value[c]!));
        expect(Math.abs(total - value)).toBeLessThan(1e-12);
      });
    });
  });
});

describe("each step highlights the right lines", () => {
  const lines = Object.fromEntries(
    LAB_STEPS.map((step) => [
      step.id,
      step.equations.flatMap((id) => [...highlightedLines(SYMBOL, id)]),
    ]),
  ) as Record<(typeof LAB_STEPS)[number]["id"], number[]>;
  const text = (id: keyof typeof lines) => lines[id].map((line) => lineText(SYMBOL, line).trim());

  test("the steps are in the required order", () => {
    expect(LAB_STEPS.map((step) => step.title)).toEqual([
      "Scores",
      "Scale",
      "Mask",
      "Softmax",
      "Output",
    ]);
  });

  test("scores and scale are the division line, and the mask is the next line", () => {
    expect(text("scores")).toEqual(["S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)"]);
    expect(text("scale")).toEqual(["S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)"]);
    expect(text("mask")).toEqual([
      "S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, S)",
    ]);
    expect(lines.mask[0]).toBe(lines.scale[0]! + 1);
  });

  test("softmax and output", () => {
    expect(text("softmax")).toEqual(["A = softmax_forward(S)"]);
    expect(new Set(text("output"))).toEqual(
      new Set(["O = merge_heads(A @ Vh)", 'out = O @ params[p + "wo"] + params[p + "bo"]']),
    );
  });
});

describe("every equation maps to the right lines", () => {
  const FORWARD: [string, string][] = [
    ["attention.q", 'Q = Xn @ params[p+"wq"] + params[p+"bq"]'],
    ["attention.k", 'K = Xn @ params[p+"wk"] + params[p+"bk"]'],
    ["attention.v", 'V = Xn @ params[p+"wv"] + params[p+"bv"]'],
    ["attention.split-qkv", "Qh, Kh, Vh = split_heads(Q, H), split_heads(K, H), split_heads(V, H)"],
    ["attention.scores", "S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)"],
    ["attention.scale", "S = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dh)"],
    ["attention.mask", "S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, S)"],
    ["attention.softmax", "A = softmax_forward(S)"],
    ["attention.masked-zero", "A = softmax_forward(S)"],
    ["attention.weighted", "O = merge_heads(A @ Vh)"],
    ["attention.weighted-row", "O = merge_heads(A @ Vh)"],
    ["attention.merge", "O = merge_heads(A @ Vh)"],
    ["attention.output", 'out = O @ params[p + "wo"] + params[p + "bo"]'],
  ];
  const BACKWARD: [string, string][] = [
    ["attention.back.wo", 'grads[p + "wo"] += c["O"].T @ dOut'],
    ["attention.back.bo", 'grads[p + "bo"] += dOut.sum(axis=0)'],
    ["attention.back.do", 'dOh = split_heads(dOut @ params[p + "wo"].T, H)'],
    ["attention.back.dv", 'dVh = c["A"].transpose(0, 2, 1) @ dOh'],
    ["attention.back.da", 'dA = dOh @ c["Vh"].transpose(0, 2, 1)'],
    ["attention.back.ds", 'dS = softmax_backward(c["A"], dA)'],
    ["attention.back.dq", 'dQh = dS @ c["Kh"] / np.sqrt(dh)'],
    ["attention.back.dk", 'dKh = dS.transpose(0, 2, 1) @ c["Qh"] / np.sqrt(dh)'],
    ["attention.back.merge", "dQ, dK, dV = merge_heads(dQh), merge_heads(dKh), merge_heads(dVh)"],
    ["attention.back.dw", 'grads[p + "w" + n] += c["Xn"].T @ dM'],
    ["attention.back.db", 'grads[p + "b" + n] += dM.sum(axis=0)'],
    ["attention.back.dxn", 'dXn += dM @ params[p + "w" + n].T'],
    ["attention.back.dbk", 'grads[p + "b" + n] += dM.sum(axis=0)'],
  ];
  const linked = (id: string, symbol: string) =>
    linksForEquation(id)
      .filter((link) => link.symbol === symbol)
      .map((link) =>
        targetLines(link.resolved)
          .replace(/\s+#.*$/, "")
          .trim(),
      );

  test.each(FORWARD)("%s", (id, line) => {
    expect(linked(id, SYMBOL)).toContain(line);
  });

  test.each(BACKWARD)("%s", (id, line) => {
    expect(linked(id, "transformer.attention_backward")).toContain(line);
  });

  test("the masked-gradient equation points at the multiplication in softmax_backward", () => {
    expect(linked("attention.back.masked", "ops.softmax_backward")).toEqual([
      "return y * (dY - np.sum(y * dY, axis=-1, keepdims=True))",
    ]);
  });

  test("every equation of these concepts is listed above", () => {
    const listed = new Set([...FORWARD, ...BACKWARD, ["attention.back.masked"]].map(([id]) => id));
    for (const concept of CONCEPTS) {
      for (const { id } of getConcept(concept).equations) expect(listed.has(id), id).toBe(true);
    }
  });
});

describe("the backward pass", () => {
  const back = getGoldenCase("attention", "attention_backward.cat_sat").outputs as Record<
    string,
    Tensor3
  > & { grads: Record<string, Vector> };

  test("a masked cell can have a non-zero dA, but its dS is exactly 0", () => {
    expect(Math.abs(back.dA![0]![1]![2]!)).toBeGreaterThan(0.01);
    back.dS!.forEach((head) =>
      head.forEach((row, i) =>
        row.forEach((value, j) => {
          if (j > i) expect(value === 0).toBe(true);
        }),
      ),
    );
  });

  test("rows of dS sum to 0, which is why the key bias gets no gradient", () => {
    back.dS!.forEach((head) =>
      head.forEach((row) => expect(Math.abs(sum(row))).toBeLessThan(1e-12)),
    );
    for (const value of back.grads.bk!) expect(Math.abs(value)).toBeLessThan(1e-12);
  });

  test("the other biases do get a gradient", () => {
    for (const name of ["bq", "bv", "bo"]) {
      expect(Math.max(...back.grads[name]!.map(Math.abs))).toBeGreaterThan(1e-3);
    }
  });
});

describe("shapes use the real config", () => {
  const sizes = getGoldenCase("attention", "attention_forward.config_shapes");
  const constant = (name: string) =>
    loadConfig().constants.find((item) => item.name === name)!.value as number;
  const { forward, backward } = sizes.outputs as Record<string, Record<string, number[]>>;

  test("the run used the values in config.py", () => {
    expect(sizes.inputs.seq_len).toBe(constant("seq_len"));
    expect(sizes.inputs.emb_dim).toBe(constant("emb_dim"));
    expect(sizes.inputs.num_heads).toBe(constant("num_heads"));
    expect(sizes.inputs.head_dim).toBe(constant("head_dim"));
  });

  test("concept shapes resolve to what the real functions returned", () => {
    const known = { ...forward, ...backward, "A @ Vh": forward!.AVh! } as Record<string, number[]>;
    let compared = 0;
    for (const id of CONCEPTS) {
      for (const shape of getConcept(id).shapes) {
        for (const stage of shape.stages) {
          const real = known[stage.name];
          if (!real) continue;
          const resolved = stage.dims.map((name) => resolveDimension(name as DimensionName).value);
          expect(resolved, `${shape.id} ${stage.name}`).toEqual(real);
          compared++;
        }
      }
    }
    expect(compared).toBeGreaterThan(15);
  });

  test("no attention shape uses V as a dimension, since V is the value matrix here", () => {
    for (const id of CONCEPTS) {
      for (const shape of getConcept(id).shapes) {
        for (const stage of shape.stages) expect(stage.dims).not.toContain("V");
      }
    }
  });
});

describe("the page", () => {
  const sections = body.split(/^## /m).slice(1);
  const STEPS = [
    "What is it?",
    "Why does the model need it?",
    "Mathematics",
    "What does each symbol mean?",
    "Tiny numerical example",
    "Tensor shapes",
    "Actual implementation",
    "Code and math connection",
    "Where is it used?",
  ];

  test("sections are in the required order", () => {
    expect(sections.map((text) => text.split("\n")[0])).toEqual([
      "Why attention?",
      "Q, K, V projections",
      "Splitting into heads",
      "Scores",
      "Scaling by sqrt(d_h)",
      "Causal mask",
      "Softmax per row",
      "Weighted values",
      "Merging heads",
      "Output projection",
      "The transformer.py implementation",
      "Tensor shapes with the real config",
      "Backward pass",
      "Role in the model",
    ]);
  });

  test.each(CONCEPTS)("%s follows the nine protocol steps, in order", (id) => {
    const section = sections.find((text) => text.includes(`<ConceptCard concept="${id}"`))!;
    expect(section).toBeDefined();
    const found = [...section.matchAll(/<ConceptStep>([^<]+)<\/ConceptStep>/g)].map((m) => m[1]);
    expect(found).toEqual(STEPS);
    for (const part of ["ConceptWhy", "Symbols", "TensorShape", "EquationCode", "Trace"]) {
      expect(section, part).toContain(`<${part} concept="${id}"`);
    }
    expect(section).toMatch(new RegExp(`<Attention\\w+Example concept="${id}"`));
  });

  test.each(CONCEPTS)("%s has every shape and equation on the page", (id) => {
    const { shapes, equations } = getConcept(id);
    for (const shape of shapes) expect(body).toContain(`shape="${shape.id}"`);
    for (const equation of equations) expect(body).toContain(`<MathBlock id="${equation.id}"`);
  });

  test.each(CONCEPTS)("%s records its PDF reference and status", (id) => {
    const concept = getConcept(id);
    expect(concept.status).toBe("implemented");
    expect(concept.pdfRef).not.toBeNull();
  });

  test("the real head count and width come from config, not typed numbers", () => {
    expect(sections[0]).toContain('<ConfigValue name="num_heads" />');
    expect(sections[0]).toContain('<ConfigValue name="head_dim" />');
    expect(body).toContain("sub-word tokens");
    expect(body).not.toMatch(/d_h = 16|H = 2 heads|2 heads/);
  });

  test("the sizes typed for the example are the sizes it runs at", () => {
    const { Xn, H } = golden.inputs as { Xn: Matrix; H: number };
    expect(sections[0]).toContain(
      `\`T = ${Xn.length}\`, \`d = ${Xn[0]!.length}\` and \`H = ${H}\`, so \`d_h = ${Xn[0]!.length / H}\``,
    );
  });

  test("the example is labelled, and the actual slot holds no numbers", () => {
    expect(sections[0]).toContain("Illustrative Example");
    expect(sections[0]).toContain("<ActualModelOutputSlot />");
    const source = readFileSync("components/AttentionExamples.tsx", "utf-8");
    expect(source.match(/kind="actual"/g)).toHaveLength(1);
    expect(source).toContain('<Badge kind="actual" />');
    expect(readFileSync("components/AttentionLab.tsx", "utf-8")).not.toContain('kind="actual"');
  });

  test("it makes no claim about the trained model and stays on this implementation", () => {
    const text = [body, ...CONCEPTS.map((id) => JSON.stringify(getConcept(id)))].join("\n");
    expect(text).not.toMatch(/multi-query|grouped.query|rotary|flash.attention/i);
    expect(text).not.toMatch(/model (focuses|looks at|learns to attend)/i);
    expect(text.match(/attends to/g)).toHaveLength(1);
    expect(text).toContain("Nothing on this page says what the trained model attends to.");
  });

  test("the q, k, v intuition is labelled as general", () => {
    expect(sections[0]).toMatch(
      /In general, these three names[^]*They are not something this project computes/,
    );
    expect(getConcept("attention.scale").why).toMatch(
      /^In general,[^]*This is a general explanation/,
    );
  });

  test("the page explains why masked positions get no gradient", () => {
    expect(body).toContain("**Why masked positions get no gradient.**");
    expect(body).toContain("`dS` is exactly 0 there");
  });
});

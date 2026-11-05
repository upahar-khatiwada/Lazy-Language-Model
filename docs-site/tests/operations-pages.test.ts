import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import { CODE_REFS } from "@/data/code-refs";
import { MathBlock } from "@/components/MathBlock";
import { requireTarget, targetLines } from "@/lib/code-targets";
import { getEquation, loadConcepts } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { resolveDimension } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { linksForEquation } from "@/lib/links";
import { numericalGrad } from "@/lib/math/gradcheck";
import {
  crossEntropy,
  crossEntropyBackward,
  layerNormBackward,
  layerNormForward,
  layerNormJacobian,
  linearBackward,
  linearForward,
  perplexity,
  relu,
  reluBackward,
  softmax,
  softmaxBackwardRows,
  xavierLimit,
} from "@/lib/math/ops";
import type { Matrix, Vector } from "@/lib/math/tensor";
import { loadInitParams } from "@/lib/parameters";
import { loadCallGraph, loadCodeIndex, loadConfig } from "@/lib/repo";

const LINEAR_PAGE = "mathematics/linear-relu-initialisation";
const SOFTMAX_PAGE = "mathematics/softmax-cross-entropy";
const PAGES = [LINEAR_PAGE, SOFTMAX_PAGE];

const OPS_PART_A = [
  "ops.xavier_uniform",
  "ops.linear_forward",
  "ops.linear_backward",
  "ops.relu_forward",
  "ops.relu_backward",
  "ops.softmax_forward",
  "ops.softmax_backward",
  "ops.cross_entropy_forward",
  "ops.cross_entropy_backward",
  "ops.perplexity",
];

const ops = loadCodeIndex().files.find(({ path }) => path === "ops.py")!;
const graph = loadCallGraph();
const section = (slug: string, title: string) =>
  readPageBody(slug)
    .split(/^## /m)
    .find((text) => text.startsWith(`${title}\n`))!;

function closeTo(actual: number[], expected: number[], tolerance: number) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, i) => {
    expect(Math.abs(value - expected[i]!), `${i}: ${value} and ${expected[i]}`).toBeLessThanOrEqual(
      tolerance,
    );
  });
}

describe("Part A: every function in ops.py", () => {
  test("the list here is every function except the layer norm pair", () => {
    expect(ops.functions.map(({ id }) => id).sort()).toEqual(
      [...OPS_PART_A, "ops.layer_norm_forward", "ops.layer_norm_backward"].sort(),
    );
  });

  test.each(OPS_PART_A)("%s is shown with its real code and has a concept", (symbol) => {
    const body = PAGES.map(readPageBody).join("\n");
    expect(body).toContain(`<CodeReference symbol="${symbol}"`);
    expect(loadConcepts().find(({ id }) => id === symbol)).toBeDefined();
    expect(body).toContain(`<ConceptCard concept="${symbol}"`);
  });

  test.each([
    [LINEAR_PAGE, "Linear backward"],
    [LINEAR_PAGE, "ReLU backward"],
    [SOFTMAX_PAGE, "Softmax backward"],
    [SOFTMAX_PAGE, "Cross-entropy backward"],
  ])("%s, %s says how the gradient reaches an update", (slug, title) => {
    expect(section(slug, title)).toContain("How this reaches a parameter update.");
  });

  test("the pages use no activation or loss that the code does not", () => {
    const body = PAGES.map(readPageBody).join("\n");
    const concepts = loadConcepts()
      .filter(({ id }) => OPS_PART_A.includes(id))
      .map((concept) => JSON.stringify(concept))
      .join("\n");
    for (const text of [body, concepts]) {
      expect(text).not.toMatch(/gelu|tanh|sigmoid|leaky|swish|label smoothing|focal|mean squared/i);
    }
  });

  test("every example is labelled illustrative: no page claims a model output", () => {
    for (const slug of PAGES) expect(readPageBody(slug)).not.toContain('kind="actual"');
  });
});

describe("Part A: the helper-versus-inline note", () => {
  const note = section(LINEAR_PAGE, "Helper or inline?");
  const cited = [...note.matchAll(/<CodeLine id="([^"]+)"/g)].map((match) => match[1]!);
  const lineOf = (id: string) => requireTarget(CODE_REFS[id]!);

  test("no pipeline code calls linear_forward, and transformer.py does not import it", () => {
    expect(graph.usedBy["ops.linear_forward"]!.production).toEqual([]);
    const file = loadCodeIndex().files.find(({ path }) => path === "transformer.py")!;
    const imported = JSON.stringify(file);
    expect(imported).not.toContain("linear_forward");
    expect(note).toContain("only `test_linear_worked_example` calls it");
  });

  test("the forward sites cited are exactly the lines that write the formula", () => {
    const file = loadCodeIndex().files.find(({ path }) => path === "transformer.py")!;
    const fromSource = new Set<number>();
    for (const symbol of file.functions) {
      symbol.source.split("\n").forEach((text, offset) => {
        if (/@ params\[[^\]]+\] \+ params\[/.test(text)) fromSource.add(symbol.startLine + offset);
      });
    }
    const forward = cited.filter((id) => !id.startsWith("linear.back"));
    const lines = forward.map((id) => lineOf(id).startLine);
    expect(lines.sort((a, b) => a - b)).toEqual([...fromSource].sort((a, b) => a - b));
    expect(lines).toHaveLength(7);
  });

  test("the backward calls cited are every call in the call graph", () => {
    const callers = graph.usedBy["ops.linear_backward"]!.production;
    expect([...callers].sort()).toEqual(["transformer.ffn_backward", "transformer.loss_and_grads"]);
    const callLines = callers.flatMap(
      (caller) =>
        graph.production[caller]!.calls.find(({ target }) => target === "ops.linear_backward")!
          .lines,
    );
    const citedLines = ["linear.back-ff2", "linear.back-ff1", "linear.back-head"].map(
      (id) => lineOf(id).startLine,
    );
    expect(citedLines.sort((a, b) => a - b)).toEqual([...callLines].sort((a, b) => a - b));
    for (const id of ["linear.back-ff2", "linear.back-ff1", "linear.back-head"]) {
      expect(cited).toContain(id);
    }
  });

  test("attention_backward does not call it, and the cited lines hold the three formulas", () => {
    const calls = graph.production["transformer.attention_backward"]!.calls;
    expect(calls.map(({ target }) => target)).not.toContain("ops.linear_backward");
    const out = targetLines(lineOf("linear.back-out"));
    const qkv = targetLines(lineOf("linear.back-qkv"));
    expect(out).toContain(".T @ dOut");
    expect(out).toContain("dOut.sum(axis=0)");
    expect(out).toContain('dOut @ params[p + "wo"].T');
    expect(qkv).toContain('c["Xn"].T @ dM');
    expect(qkv).toContain("dM.sum(axis=0)");
    expect(qkv).toContain('dM @ params[p + "w" + n].T');
    expect(cited).toContain("linear.back-out");
    expect(cited).toContain("linear.back-qkv");
  });
});

describe("Part A: results against tests/test_ops.py", () => {
  const linear = getGoldenCase("ops", "linear.test_file").inputs as {
    X: Matrix;
    W: Matrix;
    b: Vector;
  };
  const { dY } = getGoldenCase("ops", "linear.test_file.backward").inputs as { dY: Matrix };

  test("the playground starts from test_linear_worked_example", () => {
    expect(linearForward(linear.X, linear.W, linear.b)).toEqual([
      [-1, 3, 1],
      [-1, 2, -1],
    ]);
    const { dX, dW, db } = linearBackward(linear.X, linear.W, dY);
    expect(dW).toEqual([
      [1, 0, 1],
      [2, 2, 1],
    ]);
    expect(db).toEqual([1, 2, 0]);
    expect(dX).toEqual([
      [3, -1],
      [-2, 2],
    ]);
  });

  test("ReLU gives no gradient at exactly 0", () => {
    const { Z, dA } = getGoldenCase("ops", "relu_backward.test_file").inputs as {
      Z: Matrix;
      dA: Matrix;
    };
    expect(relu(Z)).toEqual([[0, 2, 0, 3]]);
    expect(reluBackward(Z, dA)).toEqual([[0, 1, 0, 1]]);
  });

  test("softmax backward has the values the test states, and each row sums to 0", () => {
    const { y, dY: dp } = getGoldenCase("ops", "softmax_backward.test_file").inputs as {
      y: Matrix;
      dY: Matrix;
    };
    const dx = softmaxBackwardRows(y, dp);
    closeTo(dx[0]!, [0.2227, -0.1628, -0.0599], 1e-3);
    for (const row of dx) expect(Math.abs(row.reduce((a, b) => a + b))).toBeLessThan(1e-12);
  });

  test("cross-entropy has the values the test states", () => {
    const { logits, targets } = getGoldenCase("ops", "cross_entropy.test_file").inputs as {
      logits: Matrix;
      targets: number[];
    };
    const { loss, probs } = crossEntropy(logits, targets);
    expect(Math.abs(loss - 0.4076)).toBeLessThan(1e-3);
    closeTo(crossEntropyBackward(probs, targets)[0]!, [-0.3348, 0.2447, 0.09], 1e-3);
  });

  test("the mean over positions gives the 1/T of the test", () => {
    const { logits, targets } = getGoldenCase("ops", "cross_entropy.mean_over_positions")
      .inputs as { logits: Matrix; targets: number[] };
    const { loss, probs } = crossEntropy(logits, targets);
    expect(loss).toBeCloseTo(Math.log(5), 12);
    expect(crossEntropyBackward(probs, targets)[0]![0]).toBeCloseTo((0.2 - 1) / 4, 12);
  });

  test("an extreme logit gives a finite loss, with a probability of exactly 0", () => {
    const { logits, targets } = getGoldenCase("ops", "cross_entropy.extreme").inputs as {
      logits: Matrix;
      targets: number[];
    };
    const { loss, probs } = crossEntropy(logits, targets);
    expect(Number.isFinite(loss)).toBe(true);
    expect(probs[0]![targets[0]!]).toBe(0);
  });

  test("perplexity of ln 300 is 300", () => {
    expect(perplexity(Math.log(300))).toBeCloseTo(300, 9);
  });
});

describe("Part A: the numerical checks on the page", () => {
  const TOLERANCE = 1e-6;

  test("linear: central differences from the real function equal the backward", () => {
    const { outputs } = getGoldenCase("ops", "gradcheck.linear");
    const { dX, dW, db } = getGoldenCase("ops", "linear.test_file.backward").outputs as {
      dX: Matrix;
      dW: Matrix;
      db: Vector;
    };
    closeTo((outputs.dX as Matrix).flat(), dX.flat(), TOLERANCE);
    closeTo((outputs.dW as Matrix).flat(), dW.flat(), TOLERANCE);
    closeTo(outputs.db as Vector, db, TOLERANCE);
  });

  test("cross-entropy: central differences from the real function equal the backward", () => {
    const { outputs } = getGoldenCase("ops", "gradcheck.cross_entropy");
    const { dZ } = getGoldenCase("ops", "cross_entropy.basic.backward").outputs as { dZ: Matrix };
    closeTo((outputs.dZ as Matrix).flat(), dZ.flat(), TOLERANCE);
  });

  test("the engine's numericalGrad leaves its input alone", () => {
    const input = [
      [1, 2],
      [3, 4],
    ];
    numericalGrad((m) => m[0]![0]! * 2, input);
    expect(input).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});

describe("Part A: why the softmax shift is safe", () => {
  const x = (id: string) => getGoldenCase("ops", id).inputs.x as Vector;
  const y = (id: string) => getGoldenCase("ops", id).outputs.y as Vector;

  test("moving every score by one constant changes nothing", () => {
    closeTo(y("softmax_forward.basic_moved"), y("softmax_forward.basic"), 1e-12);
    closeTo(y("softmax_forward.large_positive"), y("softmax_forward.small_same_gaps"), 1e-12);
    closeTo(softmax(x("softmax_forward.basic_moved")), y("softmax_forward.basic"), 1e-12);
  });

  test("without the shift, the largest score of the page's last row overflows", () => {
    expect(Math.exp(Math.max(...x("softmax_forward.large_positive")))).toBe(Infinity);
    const shifted = x("softmax_forward.large_positive");
    const top = Math.max(...shifted);
    expect(Math.max(...shifted.map((value) => Math.exp(value - top)))).toBe(1);
  });

  test("the line that shifts is tied to the equation", () => {
    const [link] = linksForEquation("softmax.safe");
    expect(targetLines(link!.resolved)).toContain("np.max(x, axis=-1, keepdims=True)");
  });
});

describe("Part A: Xavier and starting values", () => {
  test("the limit in the table is the local a of the real function", () => {
    for (const id of [
      "xavier.tiny",
      "xavier.attention",
      "xavier.feed_forward_in",
      "xavier.feed_forward_out",
      "xavier.position_table",
    ]) {
      const { inputs, outputs } = getGoldenCase("ops", id);
      const { fan_in, fan_out } = inputs as { fan_in: number; fan_out: number };
      expect(Math.abs(xavierLimit(fan_in, fan_out) - (outputs.a as number))).toBeLessThan(1e-12);
      expect(outputs.within).toBe(true);
    }
  });

  test("the sizes come from config.py", () => {
    const d = resolveDimension("d").value;
    const F = resolveDimension("F").value;
    const Tmax = resolveDimension("Tmax").value;
    const sizes = (id: string) => {
      const { fan_in, fan_out } = getGoldenCase("ops", id).inputs as Record<string, number>;
      return [fan_in, fan_out];
    };
    expect(sizes("xavier.attention")).toEqual([d, d]);
    expect(sizes("xavier.feed_forward_in")).toEqual([d, F]);
    expect(sizes("xavier.feed_forward_out")).toEqual([F, d]);
    expect(sizes("xavier.position_table")).toEqual([Tmax, d]);
  });

  test("the variance equation holds for the coded limit", () => {
    for (const [fanIn, fanOut] of [
      [2, 3],
      [32, 128],
    ] as const) {
      expect(xavierLimit(fanIn, fanOut) ** 2 / 3).toBeCloseTo(2 / (fanIn + fanOut), 12);
    }
  });

  test("the table's grouping matches what init_params did", () => {
    const { rows } = getGoldenCase("ops", "init_params.tiny").outputs as {
      rows: { name: string; shape: number[]; low: number; high: number }[];
    };
    const { parameters } = loadInitParams();
    expect(rows.length).toBeGreaterThan(0);
    for (const { name, shape } of rows) {
      const generic = name.replace(/^b\d+\./, "b{l}.");
      const declared = parameters.find((parameter) => parameter.name === generic);
      expect(declared, name).toBeDefined();
      expect(shape.length).toBe(declared!.dims.length);
      expect(declared!.init === "xavier_uniform").toBe(shape.length === 2);
    }
  });

  test("only train.main calls init_params in the pipeline", () => {
    expect(graph.usedBy["transformer.init_params"]!.production).toEqual(["train.main"]);
    expect(graph.usedBy["ops.xavier_uniform"]!.production).toEqual(["transformer.init_params"]);
  });
});

describe("Part A: perplexity", () => {
  test("it is printed once in train.main, on the validation loss", () => {
    expect(graph.usedBy["ops.perplexity"]!.production).toEqual(["train.main"]);
    const calls = graph.production["train.main"]!.calls.find(
      ({ target }) => target === "ops.perplexity",
    )!;
    expect(calls.lines).toHaveLength(1);
    expect(targetLines(requireTarget(CODE_REFS["train.validation"]!))).toContain("perplexity(v)");
  });
});

describe("Part B: LayerNorm", () => {
  const LN_PAGE = "mathematics/layernorm";
  const body = readPageBody(LN_PAGE);
  const CASES = ["layer_norm.test_file", "layer_norm.edited", "layer_norm.constant_row"];
  const forwardCase = (id: string) =>
    getGoldenCase("ops", id) as unknown as {
      inputs: { X: Matrix; gamma: Vector; beta: Vector };
      outputs: Record<string, Matrix>;
    };
  const backwardCase = (id: string) => getGoldenCase("ops", `${id}.backward`);

  test("both functions are shown with their real code, each with a concept", () => {
    for (const symbol of ["ops.layer_norm_forward", "ops.layer_norm_backward"]) {
      expect(body).toContain(`<CodeReference symbol="${symbol}"`);
      expect(body).toContain(`<ConceptCard concept="${symbol}"`);
    }
    expect(body).toContain('<ConceptCard concept="ops.layer_norm_derivation"');
  });

  test.each([
    ["ln.mean", ["mu = X.mean(axis=-1, keepdims=True)"]],
    ["ln.var", ["var = ((X - mu) ** 2).mean(axis=-1, keepdims=True)"]],
    ["ln.s", ["s = 1.0 / np.sqrt(var + epsilon)"]],
    ["ln.xhat", ["Xhat = (X - mu) * s"]],
    ["ln.output", ["return gamma * Xhat + beta, (Xhat, s)"]],
    ["ln.dxhat", ["dXhat = dY * gamma"]],
    [
      "ln.dx",
      [
        "dX = s * (dXhat - dXhat.mean(axis=-1, keepdims=True)",
        "- Xhat * (dXhat * Xhat).mean(axis=-1, keepdims=True))",
      ],
    ],
    ["ln.dgamma", ["dgamma = (dY * Xhat).sum(axis=0)"]],
    ["ln.dbeta", ["dbeta = dY.sum(axis=0)"]],
  ])("%s links to its exact code lines", (id, expected) => {
    const lines = linksForEquation(id)
      .filter(({ role }) => role === "implements")
      .map(({ resolved }) => targetLines(resolved).trim());
    expect(lines).toEqual(expected);
  });

  test("the forward is biased: the mean of squares divides by d", () => {
    const { X } = forwardCase("layer_norm.test_file").inputs;
    const row = X[0]!;
    const mu = row.reduce((a, b) => a + b) / row.length;
    const biased = row.reduce((total, x) => total + (x - mu) ** 2, 0) / row.length;
    expect(
      Math.abs(forwardCase("layer_norm.test_file").outputs.var![0]![0]! - biased),
    ).toBeLessThan(1e-12);
  });

  test("the numbers of test_layer_norm_worked_example", () => {
    const { X, gamma, beta } = forwardCase("layer_norm.test_file").inputs;
    const { Y, cache } = layerNormForward(X, gamma, beta);
    closeTo(Y[0]!, [-1.069, -0.535, 0, 4.207], 1e-3);
    const { dY } = backwardCase("layer_norm.test_file").inputs as { dY: Matrix };
    const { dX } = layerNormBackward(dY, cache, gamma);
    closeTo(dX[0]!, [0.4391, -0.2482, -0.4009, 0.21], 1e-3);
  });

  test("a row with no spread gives s from epsilon alone, and the output is beta", () => {
    const { Y, s, var: variance } = forwardCase("layer_norm.constant_row").outputs;
    expect(variance![0]![0]).toBe(0);
    expect(s![0]![0]).toBeCloseTo(1 / Math.sqrt(1e-5), 9);
    expect(Y![0]).toEqual(forwardCase("layer_norm.constant_row").inputs.beta);
  });

  test.each(CASES)("the derived matrix reproduces the coded dX: %s", (id) => {
    const { X, gamma, beta } = forwardCase(id).inputs;
    const { cache } = layerNormForward(X, gamma, beta);
    const { dY } = backwardCase(id).inputs as { dY: Matrix };
    const stored = (backwardCase(id).outputs as { dX: Matrix }).dX;
    const J = layerNormJacobian(cache.Xhat[0]!, cache.s[0]![0]!);
    const dXhat = dY[0]!.map((value, i) => value * gamma[i]!);
    const derived = X[0]!.map((_, j) =>
      dXhat.reduce((total, value, i) => total + value * J[i]![j]!, 0),
    );
    closeTo(derived, stored[0]!, 1e-12);
  });

  test("the derived matrix equals central differences of the real forward", () => {
    const { X, gamma, beta } = forwardCase("layer_norm.test_file").inputs;
    const { cache } = layerNormForward(X, gamma, beta);
    const { J } = getGoldenCase("ops", "gradcheck.layer_norm_jacobian").outputs as { J: Matrix };
    closeTo(layerNormJacobian(cache.Xhat[0]!, cache.s[0]![0]!).flat(), J.flat(), 1e-8);
  });

  test("central differences of the real forward equal the real backward", () => {
    const { outputs } = getGoldenCase("ops", "gradcheck.layer_norm");
    const real = backwardCase("layer_norm.test_file").outputs as {
      dX: Matrix;
      dgamma: Vector;
      dbeta: Vector;
    };
    closeTo((outputs.dX as Matrix).flat(), real.dX.flat(), 1e-6);
    closeTo(outputs.dgamma as Vector, real.dgamma, 1e-6);
    closeTo(outputs.dbeta as Vector, real.dbeta, 1e-6);
  });

  // The last step of the derivation, read as code. Each mean is written as 1/d and a sum on the page.
  test("the last step of the derivation, read as code, is the expression in ops.py", () => {
    const { latex } = getEquation("ln.derive.result").equation;
    const result = latex.split("=").at(-1)!;
    const asCode = result
      .replaceAll(
        String.raw`\frac{1}{d}\sum_{i} d\hat{X}_{ti}\hat{X}_{ti}`,
        "(dXhat * Xhat).mean()",
      )
      .replaceAll(String.raw`\frac{1}{d}\sum_{i} d\hat{X}_{ti}`, "dXhat.mean()")
      .replaceAll(String.raw`d\hat{X}_{tj}`, "dXhat")
      .replaceAll(String.raw`\hat{X}_{tj}\,`, "Xhat *")
      .replaceAll("s_t", "s *")
      .replaceAll(String.raw`\left(`, "(")
      .replaceAll(String.raw`\right)`, ")")
      .replace(/\s+/g, "");
    const source = linksForEquation("ln.dx")
      .filter(({ role }) => role === "implements")
      .map(({ resolved }) => targetLines(resolved))
      .join("")
      .replace("dX = ", "")
      .replaceAll("axis=-1, keepdims=True", "")
      .replace(/\s+/g, "");
    expect(asCode).toBe(source);
  });

  test("the derivation is labelled, and it uses no step the code cannot back", () => {
    const concept = loadConcepts().find(({ id }) => id === "ops.layer_norm_derivation")!;
    expect(concept.status).toBe("derived");
    expect(concept.pdfRef).toBeNull();
    for (const { id } of concept.equations) {
      expect(
        linksForEquation(id).some(({ role }) => role === "implements"),
        id,
      ).toBe(true);
    }
    const order = concept.equations.map(({ id }) => body.indexOf(`<MathBlock id="${id}"`));
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });

  test("the general explanation is labelled as general", () => {
    expect(section(LN_PAGE, "LayerNorm forward")).toMatch(
      /<Callout kind="note">\s+In general, normalising[^]*This is a general explanation\./,
    );
  });

  test("the backward section says how the gradient reaches an update", () => {
    expect(section(LN_PAGE, "LayerNorm backward")).toContain(
      "How this reaches a parameter update.",
    );
  });

  test("the pre-LN positions are the three call sites in the call graph", () => {
    const callers = graph.usedBy["ops.layer_norm_forward"]!.production;
    expect([...callers].sort()).toEqual(["transformer.block_forward", "transformer.forward"]);
    const lines = callers.flatMap(
      (caller) =>
        graph.production[caller]!.calls.find(({ target }) => target === "ops.layer_norm_forward")!
          .lines,
    );
    const cited = ["ln.block-first", "ln.block-second", "ln.final"].map(
      (id) => requireTarget(CODE_REFS[id]!).startLine,
    );
    expect(cited.sort((a, b) => a - b)).toEqual([...lines].sort((a, b) => a - b));
    for (const id of ["ln.block-first", "ln.block-second", "ln.final"]) {
      expect(body).toContain(`<CodeStep id="${id}"`);
      expect(targetLines(requireTarget(CODE_REFS[id]!))).toContain("cfg.ln_eps");
    }
  });

  test("attention reads the first copy, and the feed-forward layer reads the second", () => {
    const { source } = requireTarget(CODE_REFS["ln.block-first"]!).symbol;
    expect(source.indexOf("layer_norm_forward(X,")).toBeLessThan(
      source.indexOf("attention_forward(Xn"),
    );
    expect(source.indexOf("attention_forward(Xn")).toBeLessThan(
      source.indexOf("layer_norm_forward(X1"),
    );
    expect(source.indexOf("layer_norm_forward(X1")).toBeLessThan(source.indexOf("ffn_forward(Xn2"));
  });

  test("config.ln_eps is never read, and train.main does not pass it", () => {
    const entry = loadConfig().constants.find(({ name }) => name === "ln_eps")!;
    expect(entry.readBy).toEqual([]);
    expect(targetLines(requireTarget(CODE_REFS["train.model"]!))).not.toContain("ln_eps");
    expect(targetLines(requireTarget(CODE_REFS["ln.eps-field"]!))).toContain("ln_eps");
  });

  test("every LayerNorm parameter starts at ones or zeros", () => {
    const names = /names=\{(\[.*?\])\}/.exec(section(LN_PAGE, "Parameters"))![1]!;
    const listed = JSON.parse(names) as string[];
    const { parameters } = loadInitParams();
    expect(listed.sort()).toEqual(
      parameters
        .filter(({ name }) => /ln[12f]_[gb]$/.test(name))
        .map(({ name }) => name)
        .sort(),
    );
    for (const name of listed) {
      const expected = name.endsWith("_g") ? "ones" : "zeros";
      expect(parameters.find((parameter) => parameter.name === name)!.init).toBe(expected);
    }
  });

  test("every example is labelled illustrative", () => {
    expect(body).not.toContain('kind="actual"');
  });
});

describe("the derived label on one equation", () => {
  test("a step the PDF does not state shows the label, and the others do not", () => {
    const derived = renderToStaticMarkup(createElement(MathBlock, { id: "linear.partials" }));
    expect(derived).toContain("Derived from implementation");
    const stated = renderToStaticMarkup(createElement(MathBlock, { id: "linear.dW" }));
    expect(stated).not.toContain("Derived from implementation");
  });
});

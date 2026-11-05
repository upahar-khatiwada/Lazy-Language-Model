import { describe, expect, test } from "vitest";
import { BLOCK_CACHES } from "@/data/caches";
import { CODE_REFS } from "@/data/code-refs";
import { BLOCK_FLOW, MODEL_FLOW } from "@/data/gradient-flow";
import { getSymbol, requireTarget, targetLines } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { resolveDimension, type DimensionName } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { linksForEquation } from "@/lib/links";
import { blockForward, type FlatParams } from "@/lib/math/block";
import type { ModelSizes } from "@/lib/math/model";
import type { Matrix } from "@/lib/math/tensor";
import { loadConfig } from "@/lib/repo";
import { STREAM_STEPS, streamVector, type StreamRun } from "@/lib/residual-stream";

const body = readPageBody("mathematics/transformer-block");
const CONCEPTS = [
  "transformer.ffn_forward",
  "transformer.ffn_backward",
  "transformer.block_forward",
  "transformer.block_backward",
];
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

// Code statements of a function, without blank lines, comments or its def line.
function statements(symbolId: string): string[] {
  return getSymbol(symbolId)
    .source.split(/\r?\n/)
    .slice(1)
    .map((line) => line.replace(/\s+#.*$/, "").trim())
    .filter(Boolean);
}

const implementing = (id: string, symbol: string) =>
  linksForEquation(id)
    .filter((link) => link.symbol === symbol)
    .map((link) =>
      targetLines(link.resolved)
        .replace(/\s+#.*$/, "")
        .trim(),
    );

describe("Part A: the block equations match block_forward line by line", () => {
  const ORDER: [string, string][] = [
    [
      "block.xn",
      'Xn, c_ln1 = layer_norm_forward(X, params[p + "ln1_g"], params[p + "ln1_b"], cfg.ln_eps)',
    ],
    ["block.att", "att, c_att = attention_forward(Xn, params, p, cfg)"],
    ["block.x1", "X1 = X + att"],
    [
      "block.xn2",
      'Xn2, c_ln2 = layer_norm_forward(X1, params[p + "ln2_g"], params[p + "ln2_b"], cfg.ln_eps)',
    ],
    ["block.ff", "ff, c_ff = ffn_forward(Xn2, params, p)"],
    ["block.y", "return X1 + ff, dict(ln1=c_ln1, att=c_att, ln2=c_ln2, ff=c_ff)"],
  ];

  test("every statement after the prefix has one equation, in the same order", () => {
    const code = statements("transformer.block_forward");
    expect(code[0]).toBe('p = f"b{l}."');
    expect(code.slice(1)).toEqual(ORDER.map(([, line]) => line));
  });

  test.each(ORDER)("%s links to its statement", (id, line) => {
    expect(implementing(id, "transformer.block_forward")).toEqual([line]);
  });

  test("the page shows the equations in the code's order", () => {
    const positions = ORDER.map(([id]) => body.indexOf(`<MathBlock id="${id}" />`));
    expect(positions.every((position) => position > 0)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  });

  test("the equations say what the code says", () => {
    const concept = getConcept("transformer.block_forward");
    const latex = (id: string) => concept.equations.find((equation) => equation.id === id)!.latex;
    expect(latex("block.x1")).toBe("X_1 = X + \\mathrm{Att}");
    expect(latex("block.y")).toBe("Y = X_1 + \\mathrm{Ff}");
    expect(latex("block.compact")).toMatch(
      /X_1 = X \+ \\operatorname\{Attention\}\(\\operatorname\{LN\}_1\(X\)\)/,
    );
    expect(latex("block.compact")).toMatch(
      /Y = X_1 \+\s+\\operatorname\{FFN\}\(\\operatorname\{LN\}_2\(X_1\)\)/,
    );
  });

  test("the page names it pre-LN for where the layer norm sits, and labels the reason as general", () => {
    expect(body).toContain(
      "This form is called pre-LN because of where the LayerNorm sits in the code",
    );
    expect(body).toMatch(
      /In general, placing normalisation before a sub-layer[^]*This is a general explanation/,
    );
    expect(body).not.toMatch(/post-LN|Post-LN/);
    const lines = (id: string) => requireTarget(CODE_REFS[id]!).startLine;
    expect(lines("ln.block-first")).toBeLessThan(lines("attention.call"));
    expect(lines("ln.block-second")).toBeLessThan(lines("block.ffn-call"));
  });
});

describe("Part A: the gradient flow matches block_backward's order", () => {
  const ORDER: [string, string][] = [
    ["block.back.dxn2", 'dXn2 = ffn_backward(dY, c["ff"], params, p, grads)'],
    [
      "block.back.branch2",
      'dX1_branch, dg, db = layer_norm_backward(dXn2, c["ln2"], params[p + "ln2_g"])',
    ],
    ["block.back.dx1", "dX1 = dY + dX1_branch"],
    ["block.back.dxn", 'dXn = attention_backward(dX1, c["att"], params, p, cfg, grads)'],
    [
      "block.back.branch1",
      'dX_branch, dg, db = layer_norm_backward(dXn, c["ln1"], params[p + "ln1_g"])',
    ],
    ["block.back.dx", "return dX1 + dX_branch"],
  ];

  test("the function's statements, apart from grads, are these six in this order", () => {
    const code = statements("transformer.block_backward").filter(
      (line) => !line.startsWith("grads[") && !line.startsWith("p = "),
    );
    expect(code).toEqual(ORDER.map(([, line]) => line));
  });

  test.each(ORDER)("%s links to its statement", (id, line) => {
    expect(implementing(id, "transformer.block_backward")).toEqual([line]);
  });

  test("the diagram has one step per statement, in order", () => {
    const shown = BLOCK_FLOW.map((step) => targetLines(requireTarget(step.target)).trim());
    expect(shown).toEqual(ORDER.map(([, line]) => line));
  });

  test("the diagram's lines increase, and each call is what the code calls", () => {
    const starts = BLOCK_FLOW.map((step) => requireTarget(step.target).startLine);
    expect([...starts].sort((a, b) => a - b)).toEqual(starts);
    expect(BLOCK_FLOW.map((step) => step.call ?? step.inline)).toEqual([
      "transformer.ffn_backward",
      "ops.layer_norm_backward",
      "dY + dX1_branch",
      "transformer.attention_backward",
      "ops.layer_norm_backward",
      "dX1 + dX_branch",
    ]);
  });

  test("a traced run calls the same functions in the same order", () => {
    const { calls } = getGoldenCase("model", "loss_and_grads.order").outputs as {
      calls: { name: string; depth: number; l?: number }[];
    };
    const start = calls.findIndex((call) => call.name === "block_backward" && call.l === 1);
    const end = calls.findIndex((call) => call.name === "block_backward" && call.l === 0);
    const inside = calls.slice(start + 1, end).filter((call) => call.depth === 2);
    expect(inside.map((call) => call.name)).toEqual(
      BLOCK_FLOW.flatMap((step) => (step.call ? [step.call.split(".").at(-1)!] : [])),
    );
  });

  test("every entry of a block's grads comes from the step that says so", () => {
    const { fills } = getGoldenCase("model", "loss_and_grads.order").outputs as {
      fills: { key: string; function: string; line: number }[];
    };
    const fromBlock = fills.filter((fill) => fill.function === "block_backward");
    expect(fromBlock.length).toBeGreaterThan(0);
    for (const { key, line } of fromBlock) {
      const name = key.replace(/^b\d+\./, "b{l}.");
      const step = BLOCK_FLOW.find((candidate) => candidate.writes?.includes(name));
      expect(step, key).toBeDefined();
      const owned = [step!.target, ...(step!.writeTargets ?? [])]
        .map((target) => requireTarget(target))
        .filter(({ symbol }) => symbol.id === "transformer.block_backward")
        .map(({ startLine }) => startLine);
      expect(owned, key).toContain(line);
    }
  });

  test("the feed-forward equations follow ffn_backward's order", () => {
    const ids = ["ffn.back.da", "ffn.back.dz", "ffn.back.dxn"];
    const first = ids.map((id) => implementing(id, "transformer.ffn_backward")[0]);
    expect(first).toEqual([
      'dA, dW2, db2 = linear_backward(c["A"], params[p + "ff2_w"], dOut)',
      'dZ = relu_backward(c["Z"], dA)',
      'dXn, dW1, db1 = linear_backward(c["Xn"], params[p + "ff1_w"], dZ)',
    ]);
    const code = statements("transformer.ffn_backward");
    const at = (line: string) => code.indexOf(line);
    expect(at(first[0]!)).toBeLessThan(at(first[1]!));
    expect(at(first[1]!)).toBeLessThan(at(first[2]!));
  });

  test("the residual gradient split holds in the stored run, including with both branches off", () => {
    const close = (a: Matrix, b: Matrix) =>
      a.forEach((row, t) =>
        row.forEach((value, c) => expect(Math.abs(value - b[t]![c]!)).toBeLessThan(1e-12)),
      );
    const sum = (a: Matrix, b: Matrix) => a.map((row, t) => row.map((v, c) => v + b[t]![c]!));
    const on = getGoldenCase("block", "block_backward.tiny");
    const { dX1, dX1_branch, dX, dX_branch } = on.outputs as Record<string, Matrix>;
    const dY = on.inputs.dY as Matrix;
    close(dX1!, sum(dY, dX1_branch!));
    close(dX!, sum(dX1!, dX_branch!));
    const off = getGoldenCase("block", "block_backward.branches_off").outputs as Record<
      string,
      Matrix
    >;
    expect(off.dX1_branch!.flat().every((value) => value === 0)).toBe(true);
    expect(off.dX_branch!.flat().every((value) => value === 0)).toBe(true);
    close(off.dX!, dY);
  });
});

describe("Part A: shapes use real config values", () => {
  const sizes = getGoldenCase("block", "block.config_shapes");
  const { forward, backward } = sizes.outputs as Record<string, Record<string, number[]>>;
  const constant = (name: string) =>
    loadConfig().constants.find((item) => item.name === name)!.value as number;

  test("the run used the values in config.py", () => {
    expect(sizes.inputs.seq_len).toBe(constant("seq_len"));
    expect(sizes.inputs.emb_dim).toBe(constant("emb_dim"));
    expect(sizes.inputs.ff_dim).toBe(constant("ff_dim"));
    expect(sizes.inputs.num_heads).toBe(constant("num_heads"));
  });

  test("each shape stage resolves to what the real functions returned", () => {
    const known = { ...forward, ...backward, out: forward!.ff!, dOut: backward!.dY! };
    let compared = 0;
    for (const id of CONCEPTS) {
      for (const shape of getConcept(id).shapes) {
        for (const stage of shape.stages) {
          const real = (known as Record<string, number[]>)[stage.name];
          expect(real, `${shape.id} ${stage.name}`).toBeDefined();
          const resolved = stage.dims.map((name) => resolveDimension(name as DimensionName).value);
          expect(resolved, `${shape.id} ${stage.name}`).toEqual(real);
          compared++;
        }
      }
    }
    expect(compared).toBeGreaterThan(20);
  });

  test("no number for a size is typed on the page", () => {
    expect(body).not.toMatch(/\b(128|32|16)\b/);
    expect(body).toContain('<ConfigValue name="ff_dim" />');
    expect(body).toContain('<ConfigValue name="emb_dim" />');
    expect(body).toContain('<ConfigValue name="num_layers" />');
  });
});

describe("the stepper", () => {
  const golden = getGoldenCase("block", "block_forward.tiny");
  const { X, params, l, cfg } = golden.inputs as {
    X: Matrix;
    params: FlatParams;
    l: number;
    cfg: ModelSizes;
  };
  const run = { X, ...blockForward(X, params, l, cfg.num_heads, cfg.ln_eps) } as StreamRun;
  const stored = { X, ...golden.outputs } as unknown as StreamRun;

  test("the steps follow the statements of block_forward", () => {
    expect(STREAM_STEPS.map((step) => step.ref)).toEqual([
      "block.def",
      "ln.block-first",
      "attention.call",
      "attention.residual",
      "ln.block-second",
      "block.ffn-call",
      "block.return",
    ]);
    const starts = STREAM_STEPS.map((step) => requireTarget(CODE_REFS[step.ref]!).startLine);
    expect([...starts].sort((a, b) => a - b)).toEqual(starts);
  });

  test.each(STREAM_STEPS.map((step) => [step.id, step] as const))(
    "%s shows the values of the real function",
    (_id, step) => {
      X.forEach((_, row) => {
        for (const field of step.rows) {
          streamVector(run, field, row).forEach((value, c) =>
            expect(Math.abs(value - streamVector(stored, field, row)[c]!)).toBeLessThan(1e-9),
          );
        }
      });
    },
  );

  test("each add step's sum is the last vector", () => {
    for (const step of STREAM_STEPS.filter(({ kind }) => kind === "add")) {
      const [a, b, total] = step.rows;
      X.forEach((_, row) => {
        streamVector(run, total!, row).forEach((value, c) =>
          expect(
            Math.abs(value - streamVector(run, a!, row)[c]! - streamVector(run, b!, row)[c]!),
          ).toBeLessThan(1e-12),
        );
      });
    }
  });

  test("the example makes ReLU visible: one row of the feed-forward output is all zero", () => {
    const { ff } = golden.outputs as { ff: Matrix };
    expect(ff.some((row) => row.every((value) => value === 0))).toBe(true);
    expect(ff.some((row) => row.some((value) => value !== 0))).toBe(true);
  });
});

describe("caches", () => {
  const keys = getGoldenCase("block", "block_forward.tiny").outputs.cache_keys as Record<
    string,
    string[]
  >;
  const row = (name: string) => BLOCK_CACHES.find((cache) => cache.name === name)!;

  test("the fields are the keys of the real caches", () => {
    expect(row('c["att"]').holds.map(({ field }) => field)).toEqual(keys.att);
    expect(row('c["ff"]').holds.map(({ field }) => field)).toEqual(keys.ff);
    expect(keys.block).toEqual(["ln1", "att", "ln2", "ff"]);
    expect(BLOCK_CACHES.map(({ name }) => name).sort()).toEqual(
      keys.block!.map((key) => `c["${key}"]`).sort(),
    );
  });

  test("the rows are in the order block_backward reads them", () => {
    const lines = BLOCK_CACHES.map(({ read }) => requireTarget(read).startLine);
    expect([...lines].sort((a, b) => a - b)).toEqual(lines);
  });

  test("each field is read by the function named", () => {
    for (const cache of BLOCK_CACHES.filter(({ reader }) => reader.startsWith("transformer."))) {
      const source = getSymbol(cache.reader).source;
      for (const { field } of cache.holds) expect(source, field).toContain(`c["${field}"]`);
    }
    const source = getSymbol("ops.layer_norm_backward").source;
    expect(source).toContain("Xhat, s = cache");
  });

  test("the page warns that two caches have a key named A", () => {
    expect(body).toContain(
      "both the feed-forward cache and the attention cache have a key named `A`",
    );
    expect(row('c["att"]').holds.find(({ field }) => field === "A")!.meaning).toContain(
      "different",
    );
  });
});

describe("the page", () => {
  const sections = body.split(/^## /m).slice(1);

  test("sections are in the required order", () => {
    expect(sections.map((text) => text.split("\n")[0]!.trim())).toEqual([
      "Feed-forward network",
      "Feed-forward backward",
      "The pre-LN block",
      "Residual connections in the backward pass",
      "Caches",
      "Parameters of one block",
    ]);
  });

  test.each(CONCEPTS)("%s follows the nine protocol steps, in order", (id) => {
    const section = sections.find((text) => text.includes(`<ConceptCard concept="${id}"`))!;
    const found = [...section.matchAll(/<ConceptStep>([^<]+)<\/ConceptStep>/g)].map((m) => m[1]);
    expect(found).toEqual(STEPS);
    for (const part of ["ConceptWhy", "Symbols", "TensorShape", "EquationCode", "Trace"]) {
      expect(section, part).toContain(`<${part} concept="${id}"`);
    }
    expect(section).toMatch(new RegExp(`<\\w+Example concept="${id}"`));
    expect(section).toContain(`<CodeReference symbol="${id}"`);
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

  test("backward sections say how the gradient reaches an update", () => {
    for (const title of ["Feed-forward backward", "Residual connections in the backward pass"]) {
      const section = sections.find((text) => text.startsWith(title))!;
      expect(section).toContain("**How this reaches a parameter update.**");
      expect(section).toContain('<CodeLine id="train.adamw" />');
    }
  });

  test("general knowledge is labelled, and no claim is made about the trained model", () => {
    const calls = [...body.matchAll(/In general, /g)];
    expect(calls).toHaveLength(3);
    expect(body.match(/This is a general explanation/g)).toHaveLength(3);
    expect(body).toContain("Illustrative Example");
    expect(body).not.toMatch(/focuses on|attends to|dropout|gelu/i);
  });

  test("ReLU's A is the only A in the symbol tables", () => {
    const symbols = getConcept("transformer.ffn_forward").symbols.find((s) => s.symbol === "A")!;
    expect(symbols.meaning).toContain("nothing else");
  });

  test("the model flow reaches every block once through this page's diagram", () => {
    expect(MODEL_FLOW.filter((step) => step.perBlock).map((step) => step.id)).toEqual(["dX-block"]);
  });
});

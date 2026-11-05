import { describe, expect, test } from "vitest";
import { MODEL_CACHES } from "@/data/caches";
import { BLOCK_FLOW, MODEL_FLOW } from "@/data/gradient-flow";
import { getSymbol, loadSymbolIndex, requireTarget, targetLines } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { resolveDimension, type DimensionName } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { linksForEquation } from "@/lib/links";
import { lossAndGrads, type ModelSizes } from "@/lib/math/model";
import type { FlatParams } from "@/lib/math/block";
import { expandBlockName, loadInitParams } from "@/lib/parameters";
import { loadCallGraph, loadCodeIndex, loadConfig } from "@/lib/repo";
import { verifySiteData } from "@/lib/site-verify";
import { navSlugs } from "@/lib/pages";

const body = readPageBody("mathematics/full-model");
const CONCEPTS = ["transformer.forward", "transformer.loss_and_grads"];
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

const order = getGoldenCase("model", "loss_and_grads.order").outputs as {
  calls: { name: string; depth: number; l?: number }[];
  fills: { key: string; function: string; line: number }[];
};
const sizes = getGoldenCase("model", "model.config_shapes");
const shapes = sizes.outputs as {
  params: Record<string, number[]>;
  grads: Record<string, number[]>;
  logits: number[];
  dZ: number[];
  dXf: number[];
  dX: number[][];
};

describe("Part B: the backward order equals loss_and_grads exactly", () => {
  const REQUIRED = [
    "forward",
    "cross_entropy_forward",
    "cross_entropy_backward",
    "linear_backward",
    "layer_norm_backward",
    "block_backward(1)",
    "block_backward(0)",
    "embed_backward",
  ];

  // MODEL_FLOW's calls, with the block step repeated for each block, last block first.
  const flowCalls = (L: number) =>
    MODEL_FLOW.flatMap((step) => {
      if (!step.call) return [];
      const name = step.call.split(".").at(-1)!;
      return step.perBlock ? Array.from({ length: L }, (_, i) => `${name}(${L - 1 - i})`) : [name];
    });

  test("a traced run of the real function makes these calls", () => {
    const seen = order.calls
      .filter((call) => call.depth === 1)
      .map((call) => (call.l === undefined ? call.name : `${call.name}(${call.l})`));
    expect(seen).toEqual(REQUIRED);
  });

  test("the diagram's arrows are those calls, in that order", () => {
    expect(flowCalls(2)).toEqual(REQUIRED);
  });

  test("the engine makes those calls in that order", () => {
    const { params, ids, targets, cfg } = getGoldenCase("model", "loss_and_grads.tiny").inputs as {
      params: FlatParams;
      ids: number[];
      targets: number[];
      cfg: ModelSizes;
    };
    expect(lossAndGrads(params, ids, targets, cfg).calls).toEqual(REQUIRED);
  });

  test("each statement in the diagram is a statement of the function, in line order", () => {
    const lines = getSymbol("transformer.loss_and_grads").source.split(/\r?\n/);
    let position = 0;
    for (const step of MODEL_FLOW) {
      const statement = targetLines(requireTarget(step.target)).trim();
      const found = lines.findIndex((line, i) => i >= position && line.trim() === statement);
      expect(found, statement).toBeGreaterThanOrEqual(position);
      position = found + 1;
    }
  });

  test("the call graph has no call in this function the diagram leaves out", () => {
    const calls = loadCallGraph().production["transformer.loss_and_grads"]!.calls.map(
      ({ target }) => target,
    );
    expect([...calls].sort()).toEqual(
      MODEL_FLOW.flatMap((step) => (step.call ? [step.call] : [])).sort(),
    );
  });

  test("the page lists the order in words, in that order", () => {
    const sentence = body.match(/The order is ([^\n]+)/)![1]!;
    const named = [...sentence.matchAll(/`(\w+)`/g)]
      .map((match) => match[1]!)
      .filter((name) => getCalledNames().has(name));
    expect(named).toEqual([
      "forward",
      "cross_entropy_forward",
      "cross_entropy_backward",
      "linear_backward",
      "layer_norm_backward",
      "block_backward",
      "embed_backward",
    ]);
  });

  test.each(MODEL_FLOW.filter((step) => step.writes).map((step) => [step.id, step] as const))(
    "%s fills the entries it says, at the lines it says",
    (_id, step) => {
      const owned = [step.target, ...(step.writeTargets ?? [])]
        .map((target) => requireTarget(target))
        .filter(({ symbol }) => symbol.id === "transformer.loss_and_grads")
        .map(({ startLine }) => startLine);
      for (const key of step.writes!) {
        const fill = order.fills.find((candidate) => candidate.key === key)!;
        expect(fill, key).toBeDefined();
        expect(fill.function).toBe("loss_and_grads");
        expect(owned, key).toContain(fill.line);
      }
    },
  );

  test("the blocks fill their entries last block first", () => {
    const first = (prefix: string) => order.fills.findIndex(({ key }) => key.startsWith(prefix));
    expect(first("head_")).toBeLessThan(first("lnf_"));
    expect(first("lnf_")).toBeLessThan(first("b1."));
    expect(first("b1.")).toBeLessThan(first("b0."));
    expect(first("b0.")).toBeLessThan(first("tok_emb"));
  });
});

function getCalledNames() {
  return new Set(MODEL_FLOW.flatMap((step) => (step.call ? [step.call.split(".").at(-1)!] : [])));
}

describe("Part B: every parameter appears in the table with the right shape", () => {
  const { parameters } = loadInitParams();
  const L = resolveDimension("L").value!;
  const size = {
    d: resolveDimension("d").value!,
    F: resolveDimension("F").value!,
    Tmax: resolveDimension("Tmax").value!,
    V: sizes.inputs.V as number,
  };

  test("the run used the sizes in config.py", () => {
    expect(sizes.inputs.emb_dim).toBe(size.d);
    expect(sizes.inputs.ff_dim).toBe(size.F);
    expect(sizes.inputs.context_len).toBe(size.Tmax);
    expect(sizes.inputs.num_layers).toBe(L);
  });

  test("expanded keys equal the keys of the real init_params, in insertion order", () => {
    const expanded = expandBlockName(parameters, L);
    expect(expanded.map(({ key }) => key)).toEqual(Object.keys(shapes.params));
  });

  test("each key's shape equals the shape from the real init_params", () => {
    for (const { key, dims } of expandBlockName(parameters, L)) {
      expect(
        dims.map((dim) => size[dim]),
        key,
      ).toEqual(shapes.params[key]);
    }
  });

  test("grads has the same keys and shapes as the parameters, in the real run", () => {
    expect(Object.keys(shapes.grads)).toEqual(Object.keys(shapes.params));
    for (const [key, shape] of Object.entries(shapes.params)) {
      expect(shapes.grads[key], key).toEqual(shape);
    }
  });

  test("the page shows the whole table, not a subset", () => {
    expect(body).toContain("<ParameterTable />");
    expect(body).toContain("<ParameterKeys />");
  });

  test("the two flows together fill every parameter", () => {
    const named = [...MODEL_FLOW, ...BLOCK_FLOW].flatMap((step) => step.writes ?? []).sort();
    expect(named).toEqual(parameters.map(({ name }) => name).sort());
    expect(new Set(named).size).toBe(named.length);
  });

  test("a traced run fills every key except, possibly, the key biases", () => {
    const filled = new Set(order.fills.map(({ key }) => key));
    const expected = getGoldenCase("model", "loss_and_grads.tiny").inputs.params as FlatParams;
    for (const key of Object.keys(expected)) {
      if (!key.endsWith(".bk")) expect(filled.has(key), key).toBe(true);
    }
  });
});

describe("the flow data is checked against the code", () => {
  const base = () => ({
    symbols: loadSymbolIndex(),
    callGraph: loadCallGraph(),
    config: loadConfig(),
    navSlugs: navSlugs(),
    parameterNames: loadInitParams().parameters.map((parameter) => parameter.name),
    repoFiles: loadCodeIndex().files.map((file) => file.path),
  });
  const withStep = (id: string, change: object) =>
    MODEL_FLOW.map((step) => (step.id === id ? { ...step, ...change } : step));

  test("the real data has no problems", () => {
    expect(verifySiteData(base())).toEqual([]);
  });

  test.each([
    ["a wrong call", { call: "ops.relu_backward" }, "does not call ops.relu_backward"],
    ["a line that is gone", { target: { ...MODEL_FLOW[3]!.target, match: "nope" } }, "not found"],
    ["both call and inline", { inline: "dZ" }, "exactly one of call and inline"],
  ])("rejects %s", (_name, change, message) => {
    const problems = verifySiteData({ ...base(), modelFlow: withStep("dZ", change) });
    expect(problems.join("\n")).toContain(message);
  });

  test("rejects steps out of order", () => {
    const [a, b, ...rest] = MODEL_FLOW;
    const problems = verifySiteData({ ...base(), modelFlow: [b!, a!, ...rest] });
    expect(problems.join("\n")).toContain("comes before the previous step");
  });

  test("rejects a flow that leaves out a parameter", () => {
    const modelFlow = withStep("embed", { writes: ["tok_emb"] });
    expect(verifySiteData({ ...base(), modelFlow }).join("\n")).toContain(
      "nothing declared for pos_emb",
    );
  });

  test("rejects a cache field that the code does not build", () => {
    const caches = [{ ...MODEL_CACHES[0]!, holds: [{ field: "nope", meaning: "x" }] }];
    expect(verifySiteData({ ...base(), caches }).join("\n")).toContain("nope is not in the line");
  });
});

describe("shapes use the real config", () => {
  test("each shape stage resolves to what the real functions returned", () => {
    const known: Record<string, number[]> = {
      logits: shapes.logits,
      dZ: shapes.dZ,
      dXf: shapes.dXf,
      dX: shapes.dX[0]!,
      X: shapes.dX[0]!,
      Xf: shapes.dXf,
      ids: [sizes.inputs.seq_len as number],
      head_w: shapes.grads.head_w!,
      head_b: shapes.grads.head_b!,
      tok_emb: shapes.grads.tok_emb!,
      pos_emb: shapes.grads.pos_emb!,
    };
    let compared = 0;
    for (const id of CONCEPTS) {
      for (const shape of getConcept(id).shapes) {
        for (const stage of shape.stages) {
          const real = known[stage.name];
          expect(real, stage.name).toBeDefined();
          const resolved = stage.dims.map(
            (name) => resolveDimension(name as DimensionName).value ?? (sizes.inputs.V as number),
          );
          expect(resolved, `${shape.id} ${stage.name}`).toEqual(real);
          compared++;
        }
      }
    }
    expect(compared).toBeGreaterThan(8);
  });

  test("no size is typed on the page, and V stays a symbol", () => {
    expect(body).not.toMatch(/\b(128|32|16|575)\b/);
    expect(body).toContain('<ConfigValue name="num_layers" />');
  });
});

describe("the forward equations", () => {
  const statement = (id: string) =>
    linksForEquation(id)
      .filter((link) => link.symbol === "transformer.forward")
      .map((link) => targetLines(link.resolved).trim());

  test.each([
    ["model.blocks", "X, c = block_forward(X, params, l, cfg)"],
    [
      "model.final",
      'Xf, c_lnf = layer_norm_forward(X, params["lnf_g"], params["lnf_b"], cfg.ln_eps)',
    ],
    ["model.logits", 'logits = Xf @ params["head_w"] + params["head_b"]'],
  ])("%s", (id, line) => {
    expect(statement(id)).toEqual([line]);
  });

  test("the function returns logits and a cache, and calls no softmax itself", () => {
    const { source } = getSymbol("transformer.forward");
    expect(source).toContain("return logits, dict(blocks=block_caches, lnf=c_lnf, Xf=Xf)");
    expect(source).not.toContain("softmax");
    const calls = loadCallGraph().production["transformer.forward"]!.calls.map((c) => c.target);
    expect(calls).not.toContain("ops.softmax_forward");
  });

  test("the page describes the loss and the sampler as they are", () => {
    expect(getSymbol("ops.cross_entropy_forward").source).not.toContain("softmax_forward");
    expect(body).toContain("It does not call `softmax_forward`");
    expect(getSymbol("generate.sample_next").source).toContain(
      "p = softmax_forward(logits / temperature)",
    );
    expect(body).toContain("`sample_next` calls `softmax_forward`");
  });
});

describe("the cache table", () => {
  test("the fields are the keys of the real cache", () => {
    const { cache_keys, block_caches } = getGoldenCase("model", "forward.tiny").outputs as {
      cache_keys: string[];
      block_caches: number;
    };
    expect(cache_keys).toEqual(["blocks", "lnf", "Xf"]);
    expect(block_caches).toBe(2);
    expect(MODEL_CACHES.map(({ name }) => name)).toEqual([
      'cache["Xf"]',
      'cache["lnf"]',
      'cache["blocks"][l]',
    ]);
  });
});

describe("the page", () => {
  const sections = body.split(/^## /m).slice(1);

  test("sections are in the required order", () => {
    expect(sections.map((text) => text.split("\n")[0]!.trim())).toEqual([
      "Forward pass",
      "Backward pass",
      "Gradient flow",
      "The gradient dictionary",
      "Parameter keys",
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
    const { shapes: list, equations } = getConcept(id);
    for (const shape of list) expect(body).toContain(`shape="${shape.id}"`);
    for (const equation of equations) expect(body).toContain(`<MathBlock id="${equation.id}"`);
  });

  test("L means only the number of blocks: the equation that uses L for the loss stays off the page", () => {
    expect(body).not.toContain('<MathBlock id="ce.loss"');
    expect(
      getConcept("transformer.forward").symbols.find((s) => s.symbol === "L")!.meaning,
    ).toContain("the loss is written as the word loss");
  });

  test("the backward section reuses earlier equations and states the rest as order only", () => {
    for (const id of ["ce.backward", "linear.dW", "linear.db", "linear.dX"]) {
      expect(body).toContain(`<MathBlock id="${id}" />`);
    }
    for (const { label } of getConcept("transformer.loss_and_grads").equations) {
      expect(label).toMatch(/^Order of calls/);
    }
    expect(body).toContain("They add no new derivative");
  });

  test("the two rows the page points out behave as it says", () => {
    const { params, ids } = getGoldenCase("model", "loss_and_grads.tiny").inputs as {
      params: FlatParams;
      ids: number[];
    };
    const { grads } = getGoldenCase("model", "loss_and_grads.tiny").outputs as {
      grads: { pos_emb: number[][] } & Record<string, number[]>;
    };
    expect((params.pos_emb as number[][]).length).toBeGreaterThan(ids.length);
    expect(grads.pos_emb!.at(-1)!.every((value) => value === 0)).toBe(true);
    for (const key of ["b0.bk", "b1.bk"]) {
      expect(Math.max(...grads[key]!.map(Math.abs))).toBeLessThan(1e-12);
    }
    expect(body).toContain("its last row has a gradient of exactly 0");
  });

  test("it says how the gradient reaches an update", () => {
    expect(sections[1]).toContain("**How this reaches a parameter update.**");
    expect(sections[1]).toContain('<CodeLine id="train.adamw" />');
  });

  test("its claims about the dictionary hold in the code", () => {
    const counted = [...MODEL_FLOW, ...BLOCK_FLOW].flatMap((step) => step.writes ?? []);
    expect(new Set(counted).size).toBe(counted.length);
    expect(getSymbol("transformer.embed_backward").source).toContain("np.add.at");
    expect(getSymbol("optim.Adam.step").source).toContain("grads[");
  });

  test("the numbers are labelled, and no claim is made about the trained model", () => {
    expect(body).toContain("Illustrative Example");
    expect(body).toContain("say nothing about the trained model");
    expect(body).not.toMatch(/focuses on|attends to|dropout|weight tying|is tied/i);
  });
});

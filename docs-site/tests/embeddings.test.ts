import { describe, expect, test } from "vitest";
import { getSymbol, targetLines } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { resolveDimension } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { linksForEquation } from "@/lib/links";
import { embedBackward, embedForward, overwritingBackward } from "@/lib/math/embeddings";
import type { Matrix } from "@/lib/math/tensor";
import { loadInitParams } from "@/lib/parameters";
import { loadCallGraph, loadConfig } from "@/lib/repo";

const SLUG = "mathematics/embeddings";
const body = readPageBody(SLUG);
const graph = loadCallGraph();
const CONCEPTS = [
  "embedding.token",
  "embedding.position",
  "transformer.embed_forward",
  "transformer.embed_backward",
];

const implementing = (id: string) =>
  linksForEquation(id)
    .filter(({ role }) => role === "implements")
    .map(({ resolved }) => targetLines(resolved).trim());

const section = (title: string) =>
  body.split(/^## /m).find((text) => text.startsWith(`${title}\n`))!;

const configValue = (name: string) =>
  loadConfig().constants.find((constant) => constant.name === name)!.value as number;

describe("learned positions", () => {
  test("the code creates pos_emb as a parameter, from xavier_uniform", () => {
    const { parameters } = loadInitParams();
    const position = parameters.find(({ name }) => name === "pos_emb")!;
    expect(position.init).toBe("xavier_uniform");
    expect(position.dims).toEqual(["Tmax", "d"]);
    expect(getSymbol("transformer.init_params").source).toContain(
      '"pos_emb": xavier_uniform(cfg.context_len, d, rng)',
    );
  });

  test("the page and the concept say so", () => {
    expect(getConcept("embedding.position").summary).toMatch(/second learned table/);
    expect(getConcept("embedding.position").summary).toMatch(/not computed from a fixed\s+formula/);
    expect(section("Positional embedding")).toContain("The positions are learned");
  });

  test("the page names no other kind of position encoding", () => {
    const text = [body, ...CONCEPTS.map((id) => JSON.stringify(getConcept(id)))].join("\n");
    expect(text).not.toMatch(/sin\b|cos\b|sinusoid|rotary|rope\b/i);
  });

  test("the general explanation is labelled as general", () => {
    expect(section("Positional embedding")).toMatch(
      /<Callout kind="note">\s+In general, attention[^]*This is a general explanation\./,
    );
  });

  test("a backward pass and an optimizer step change the rows in the window", () => {
    const { gradient_nonzero, row_changed } = getGoldenCase(
      "embeddings",
      "position_update.no_decay",
    ).outputs as { gradient_nonzero: boolean[]; row_changed: boolean[] };
    expect(gradient_nonzero).toEqual([true, true, true, true, true, false, false, false]);
    expect(row_changed).toEqual(gradient_nonzero);
  });

  test("weight decay also changes the rows no sequence reaches", () => {
    const { row_changed } = getGoldenCase("embeddings", "position_update.with_decay").outputs as {
      row_changed: boolean[];
    };
    expect(row_changed.every(Boolean)).toBe(true);
  });
});

describe("shapes use the real config values", () => {
  const sizes = getGoldenCase("embeddings", "embed_forward.config_sizes");

  test("the golden case ran with the values in config.py", () => {
    expect(sizes.inputs.seq_len).toBe(configValue("seq_len"));
    expect(sizes.inputs.context_len).toBe(configValue("context_len"));
    expect(sizes.inputs.emb_dim).toBe(configValue("emb_dim"));
  });

  test("the concept shapes resolve to what the real functions returned", () => {
    const T = resolveDimension("T").value;
    const d = resolveDimension("d").value;
    const Tmax = resolveDimension("Tmax").value;
    expect(sizes.outputs.X0_shape).toEqual([T, d]);
    expect(sizes.outputs.pos_emb_shape).toEqual([Tmax, d]);
    expect(sizes.outputs.at_limit_shape).toEqual([Tmax, d]);
    const stages = getConcept("transformer.embed_forward").shapes[0]!.stages;
    expect(stages.at(-1)).toEqual({ name: "X0", dims: ["T", "d"] });
    const slice = getConcept("embedding.position").shapes[0]!.stages;
    expect(slice[0]).toEqual({ name: "pos_emb", dims: ["Tmax", "d"] });
    expect(slice[1]).toEqual({ name: "pos_emb[:T]", dims: ["T", "d"] });
  });

  test("V stays symbolic, because it is not in config.py", () => {
    expect(resolveDimension("V").value).toBeNull();
    const tokens = getConcept("transformer.embed_backward").shapes[0]!.stages;
    expect(tokens[1]).toEqual({ name: 'grads["tok_emb"]', dims: ["V", "d"] });
  });

  test("a sequence over the limit fails with the message in forward", () => {
    expect(sizes.outputs.too_long_message).toBe("sequence longer than the position table");
    expect(getSymbol("transformer.forward").source).toContain(
      '"sequence longer than the position table"',
    );
  });

  test("every shape on the page is a shape of the concept", () => {
    for (const id of CONCEPTS) {
      for (const shape of getConcept(id).shapes) {
        expect(body).toContain(`<TensorShape concept="${id}" shape="${shape.id}" />`);
      }
    }
  });

  test("the page types neither table size in its prose", () => {
    const prose = section("Why the position table is longer than a window");
    expect(prose).not.toMatch(/\b(16|31|32)\b/);
    expect(prose).toContain('<ConfigValue name="context_len" />');
    expect(prose).toContain('<ConfigValue name="seq_len" />');
    expect(prose).toContain("<UnusedPositionRows />");
  });

  test("seq_len is not a field of ModelConfig, and context_len is", () => {
    const source = getSymbol("transformer.ModelConfig").source;
    expect(source).toContain("context_len: int");
    expect(source).not.toContain("seq_len");
  });
});

describe("accumulation equals np.add.at", () => {
  const CASES = [
    "embed_backward.test_file",
    "embed_backward.all_same",
    "embed_backward.no_repeat",
    "embed_backward.short",
  ];

  test("the line that does it is np.add.at", () => {
    expect(implementing("emb.dtok")).toEqual([
      'np.add.at(grads["tok_emb"], ids, dX0) # grads["tok_emb"][ids[t]] += dX0[t]',
    ]);
    expect(implementing("emb.dpos")).toEqual(['grads["pos_emb"][: len(ids)] += dX0']);
  });

  test.each(CASES)("the engine equals the real embed_backward: %s", (id) => {
    const { inputs, outputs } = getGoldenCase("embeddings", id);
    const { dX0, ids, V, Tmax } = inputs as { dX0: Matrix; ids: number[]; V: number; Tmax: number };
    const { tok, pos } = embedBackward(dX0, ids, V, Tmax);
    expect(tok).toEqual(outputs.tok);
    expect(pos).toEqual(outputs.pos);
  });

  test("the example has the values the repository's test asserts", () => {
    const { tok, pos } = getGoldenCase("embeddings", "embed_backward.test_file").outputs as {
      tok: Matrix;
      pos: Matrix;
    };
    const { dX0 } = getGoldenCase("embeddings", "embed_backward.test_file").inputs as {
      dX0: Matrix;
    };
    expect(tok[2]).toEqual([3, 2]);
    expect(tok[0]).toEqual([0, 1]);
    expect(tok[1]).toEqual([0, 0]);
    expect(pos).toEqual(dX0);
    const test = getSymbol("tests.test_transformer.test_embedding_backward_sums_repeated_ids");
    expect(test.source).toContain('grads["tok_emb"][2], [3, 2]');
    expect(test.source).toContain('grads["tok_emb"][0], [0, 1]');
  });

  test("the plain indexed add really loses a row, in NumPy and in the engine", () => {
    const { inputs, outputs } = getGoldenCase("embeddings", "embed_backward.test_file");
    const { dX0, ids, V } = inputs as { dX0: Matrix; ids: number[]; V: number };
    const overwritten = outputs.overwritten as Matrix;
    expect(overwritten[2]).toEqual([2, 2]);
    expect(overwritten[2]).not.toEqual((outputs.tok as Matrix)[2]);
    expect(overwritingBackward(dX0, ids, V)).toEqual(overwritten);
  });

  test("without a repeat, the two agree", () => {
    const { tok, overwritten } = getGoldenCase("embeddings", "embed_backward.no_repeat").outputs;
    expect(tok).toEqual(overwritten);
  });

  test("central differences of the real forward agree with the backward", () => {
    const { tok, pos } = getGoldenCase("embeddings", "gradcheck.embedding").outputs as {
      tok: Matrix;
      pos: Matrix;
    };
    const real = getGoldenCase("embeddings", "embed_backward.test_file").outputs as {
      tok: Matrix;
      pos: Matrix;
    };
    tok.flat().forEach((value, i) => {
      expect(Math.abs(value - real.tok.flat()[i]!)).toBeLessThan(1e-6);
    });
    pos.flat().forEach((value, i) => {
      expect(Math.abs(value - real.pos.flat()[i]!)).toBeLessThan(1e-6);
    });
  });

  test("a row past T gets nothing", () => {
    const { pos } = getGoldenCase("embeddings", "embed_backward.short").outputs as { pos: Matrix };
    expect(pos[2]).toEqual([0, 0]);
  });
});

describe("the forward example", () => {
  test("it has the values the repository's test asserts", () => {
    const { inputs, outputs } = getGoldenCase("embeddings", "embed_forward.test_file");
    const { tok_emb, pos_emb, ids } = inputs as {
      tok_emb: Matrix;
      pos_emb: Matrix;
      ids: number[];
    };
    const expected = [
      [0.5, 0.5],
      [0.2, 0.2],
      [0.7, 0.5],
    ];
    (outputs.X0 as Matrix).flat().forEach((value, i) => {
      expect(Math.abs(value - expected.flat()[i]!)).toBeLessThan(1e-12);
    });
    const { X0, tokens } = embedForward(tok_emb, pos_emb, ids);
    expect(tokens[0]).toEqual(tokens[2]);
    expect(X0[0]).not.toEqual(X0[2]);
  });

  test("the engine refuses a sequence longer than the table, as forward does", () => {
    expect(() => embedForward([[1]], [[1]], [0, 0])).toThrow("position table");
  });
});

describe("the page follows the protocol", () => {
  test.each(CONCEPTS)("%s has every step", (id) => {
    expect(body).toContain(`<ConceptCard concept="${id}" split />`);
    expect(body).toContain(`<ConceptWhy concept="${id}" />`);
    expect(body).toContain(`<Symbols concept="${id}" />`);
    expect(body).toContain(`<EquationCode concept="${id}" />`);
    expect(body).toContain(`<Trace concept="${id}" />`);
    expect(body).toMatch(new RegExp(`<\\w*Example concept="${id}" />`));
    for (const { id: equation } of getConcept(id).equations) {
      expect(body).toContain(`<MathBlock id="${equation}" />`);
    }
  });

  test("the code is real, and the callers come from the call graph", () => {
    expect(body).toContain('<CodeReference symbol="transformer.embed_forward"');
    expect(body).toContain('<CodeReference symbol="transformer.embed_backward"');
    expect(graph.usedBy["transformer.embed_forward"]!.production).toEqual(["transformer.forward"]);
    expect(graph.usedBy["transformer.embed_backward"]!.production).toEqual([
      "transformer.loss_and_grads",
    ]);
  });

  test("the backward section says how the gradient reaches an update", () => {
    expect(section("Embedding backward")).toContain("How this reaches a parameter update.");
  });

  test("the inaccurate comment is not used as an explanation", () => {
    const text = section("Positional embedding");
    expect(text).toContain('says "weighted average of values". That does not describe the code');
  });

  test("the reason for the two sizes is not invented", () => {
    expect(section("Why the position table is longer than a window")).toContain(
      "not available from the implementation",
    );
  });

  test("no example is labelled as model output", () => {
    expect(body).not.toContain('kind="actual"');
  });
});

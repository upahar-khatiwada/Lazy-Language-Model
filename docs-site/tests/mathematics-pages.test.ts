import { describe, expect, test } from "vitest";
import { SYMBOL_COVERAGE } from "@/data/coverage";
import { NOTATION } from "@/data/notation";
import { CODE_REFS } from "@/data/code-refs";
import { requireTarget, targetLines } from "@/lib/code-targets";
import { getEquation, loadConcepts } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { resolveDimension } from "@/lib/dimensions";
import { loadGolden, type GoldenCase } from "@/lib/golden";
import { bpeSteps, type Merge } from "@/lib/math/bpe";
import { makeWindows, windowCount } from "@/lib/math/data";
import { blockParameterCount, parameterCount, type ParameterDims } from "@/lib/math/params";
import { countInV, evaluateCount, loadInitParams } from "@/lib/parameters";
import { loadCallGraph, loadCodeIndex, loadConfig } from "@/lib/repo";

const PROTOCOL = [
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

const PAGES = [
  "mathematics/notation-and-shapes",
  "mathematics/tokenization",
  "mathematics/data-pipeline",
  "mathematics/linear-relu-initialisation",
  "mathematics/softmax-cross-entropy",
  "mathematics/layernorm",
];

// The h2 sections of a page that show a concept, with the steps in the order they are written.
function conceptSections(slug: string) {
  return readPageBody(slug)
    .split(/^## /m)
    .slice(1)
    .filter((section) => section.includes("<ConceptCard "))
    .map((section) => ({
      title: section.split("\n")[0]!,
      concept: /<ConceptCard concept="([^"]+)"/.exec(section)![1]!,
      body: section,
      steps: [...section.matchAll(/<ConceptStep>(.+?)<\/ConceptStep>/g)].map((m) => m[1]!),
    }));
}

describe.each(PAGES)("%s", (slug) => {
  test("has concepts", () => {
    expect(conceptSections(slug).length).toBeGreaterThan(0);
  });

  test.each(conceptSections(slug).map((section) => [section.title, section] as const))(
    "%s follows the authoring protocol",
    (_title, { steps, concept, body }) => {
      expect(steps).toEqual(PROTOCOL);
      expect(body).toContain(`<Symbols concept="${concept}"`);
      expect(body).toContain(`<EquationCode concept="${concept}"`);
      expect(body).toContain(`<Trace concept="${concept}"`);
      // A MathBlock for every equation of the concept, once.
      const found = loadConcepts().find((item) => item.id === concept)!;
      for (const { id } of found.equations) {
        expect(body.split(`<MathBlock id="${id}"`).length - 1, id).toBe(1);
      }
    },
  );

  test("types no equation and no example number in the page", () => {
    const body = readPageBody(slug);
    expect(body).not.toMatch(/\$\$|\\frac|\\sum/);
  });
});

describe("Part A: symbols", () => {
  test("every symbol maps to a real code name", () => {
    const required = ["T", "Tmax", "d", "H", "dh", "F", "L", "V"];
    expect(NOTATION.map((row) => row.dimension).sort()).toEqual([...required].sort());
    for (const row of NOTATION) {
      expect(targetLines(requireTarget(row.target)), row.dimension).toContain(row.code);
      if (row.config) {
        expect(loadConfig().constants.map((constant) => constant.name)).toContain(row.config);
      }
    }
  });

  test("the mapping the task states", () => {
    const configOf = Object.fromEntries(NOTATION.map((row) => [row.dimension, row.config]));
    expect(configOf).toEqual({
      T: "seq_len",
      Tmax: "context_len",
      d: "emb_dim",
      H: "num_heads",
      dh: "head_dim",
      F: "ff_dim",
      L: "num_layers",
      V: null,
    });
  });

  test("values come from config.json, and V has none", () => {
    for (const { dimension, config } of NOTATION) {
      const expected = loadConfig().constants.find((constant) => constant.name === config)?.value;
      expect(resolveDimension(dimension).value).toBe(expected ?? null);
    }
  });
});

// Evaluates the LaTeX of a concept's formula, so that the equation on the page is what gets tested,
// not a second copy of it.
function evaluateTex(id: string, dims: ParameterDims): number {
  const { latex } = getEquation(id).equation;
  const expression = latex
    .split("=")[1]!
    .replace(/\\left|\\right/g, "")
    .replace(/T_\{\\max\}/g, `(${dims.Tmax})`)
    .replace(/\bL\b/g, `(${dims.L})`)
    .replace(/V/g, `(${dims.V})`)
    .replace(/F/g, `(${dims.F})`)
    .replace(/d/g, `(${dims.d})`)
    .replace(/\^2/g, "**2")
    .replace(/\)\(/g, ")*(")
    .replace(/(\d)\(/g, "$1*(");
  return new Function(`return ${expression};`)() as number;
}

describe("Part A: the parameter formula", () => {
  const cases: GoldenCase[] = loadGolden("params");

  test("the golden file has a tiny configuration and the config shape", () => {
    expect(cases.map((item) => item.id)).toContain("count_params.tiny");
    expect(cases.map((item) => item.id)).toContain("count_params.config_shape_with_test_vocab");
  });

  test.each(cases.map((item) => [item.id, item] as const))(
    "the equation on the page equals count_params: %s",
    (_id, item) => {
      const dims = item.inputs as ParameterDims;
      expect(evaluateTex("params.total", dims)).toBe(item.outputs.count);
      expect(evaluateTex("params.block", dims)).toBe(item.outputs.block);
      expect(parameterCount(dims)).toBe(item.outputs.count);
      expect(blockParameterCount(dims)).toBe(item.outputs.block);
    },
  );

  test("the formula equals the sum of the shapes init_params created", () => {
    const { parameters, probe, countParams } = loadInitParams();
    expect(evaluateCount(parameters, probe)).toBe(countParams);
    expect(parameterCount(probe)).toBe(countParams);
    for (const dims of [
      { V: 2, Tmax: 3, d: 5, F: 7, L: 11 },
      { V: 575, Tmax: 32, d: 32, F: 128, L: 2 },
    ]) {
      expect(parameterCount(dims)).toBe(evaluateCount(parameters, dims));
    }
  });

  test("with config values the formula gives the a V + b the page shows", () => {
    const value = (name: "Tmax" | "d" | "F" | "L") => resolveDimension(name).value!;
    const values = { Tmax: value("Tmax"), d: value("d"), F: value("F"), L: value("L") };
    const { perVocabulary, constant } = countInV(loadInitParams().parameters, values);
    expect(parameterCount({ ...values, V: 0 })).toBe(constant);
    expect(parameterCount({ ...values, V: 1 }) - constant).toBe(perVocabulary);
  });
});

describe("Part A: the page", () => {
  const body = readPageBody("mathematics/notation-and-shapes");

  test("labels the test-only section, and every interactive example is illustrative", () => {
    expect(body).toContain("test-only mathematics");
    expect(body).not.toContain('kind="actual"');
  });
});

const bpeCase = (id: string) => loadGolden("bpe").find((item) => item.id === id)!;

describe.each([
  ["tokenizer.py", "mathematics/tokenization"],
  ["data.py", "mathematics/data-pipeline"],
])("every symbol of %s", (path, slug) => {
  const file = loadCodeIndex().files.find((candidate) => candidate.path === path)!;
  const everySymbol = [
    ...file.constants.map(({ name }) => `${file.module}.${name}`),
    ...file.functions.map(({ id }) => id),
    ...file.classes.flatMap(({ id, methods }) => [id, ...methods.map((method) => method.id)]),
  ];
  const rows = SYMBOL_COVERAGE[path]!;
  const sections = readPageBody(slug).split(/^## /m);

  test("is listed exactly once", () => {
    expect(rows.map((row) => row.symbol).sort()).toEqual([...everySymbol].sort());
  });

  test.each(rows.map((row) => [row.symbol, row] as const))(
    "%s is shown in its section, or skipped with a reason",
    (symbol, row) => {
      if ("skip" in row) {
        expect(row.skip.length).toBeGreaterThan(20);
        return;
      }
      const section = sections.find((text) => text.startsWith(`${row.section}\n`));
      expect(section, row.section).toBeDefined();
      expect(section).toContain(`<CodeReference symbol="${symbol}"`);
    },
  );
});

describe("Part B: the stepper against the real train_bpe", () => {
  test.each(["train_bpe_on_text.stepper", "train_bpe_on_text.test_file"])(
    "the merges are equal: %s",
    (id) => {
      const { inputs, outputs } = bpeCase(id);
      const { steps, stopped } = bpeSteps(inputs.text as string);
      const python = outputs.merges as Merge[];
      expect(steps.length).toBeGreaterThan(0);
      expect(steps.map(({ best, merged }) => [best.pair, merged, best.count])).toEqual(
        python.map(({ pair, merged, frequency }) => [pair, merged, frequency]),
      );
      expect(stopped).toBe("no-pairs");
    },
  );

  test("the first text shows a tie, and the first maximum found wins", () => {
    const [first] = bpeSteps(bpeCase("train_bpe_on_text.stepper").inputs.text as string).steps;
    expect(first!.best.pair).toEqual(["l", "o"]);
    expect(first!.tied.map(({ pair }) => pair)).toContainEqual(["o", "w"]);
    expect(first!.best.count).toBe(first!.tied[0]!.count);
  });

  test("merges never cross a pre-token, and never hold whitespace or punctuation", () => {
    for (const item of loadGolden("bpe").filter(({ mirrors }) => mirrors.endsWith("_on_text"))) {
      for (const { merged } of item.outputs.merges as Merge[]) expect(merged).toMatch(/^\w+$/);
    }
    expect(bpeCase("train_bpe_on_text.punctuation_never_merges").outputs.merges).toEqual([]);
  });

  test("a merge limit stops early, as max_merges does", () => {
    const { steps, stopped } = bpeSteps(
      bpeCase("train_bpe_on_text.limit").inputs.text as string,
      3,
    );
    expect(steps).toHaveLength(3);
    expect(stopped).toBe("limit");
  });

  test("the stepper is labelled, and the page does not present its merges as the model's", () => {
    const body = readPageBody("mathematics/tokenization");
    expect(body).toContain('<BpeExample concept="tokenizer.train_bpe"');
    expect(body).not.toContain('kind="actual"');
  });
});

describe("Part B: the unknown token, as the page describes it", () => {
  const output = (id: string) =>
    bpeCase(id).outputs as { tokens: string[]; ids: number[]; decoded: string };

  test("an unseen character gets id 0, and decode gives the literal string", () => {
    expect(output("encode.unseen_punctuation").ids.at(-1)).toBe(0);
    expect(output("encode.unseen_punctuation").decoded).toBe("low<unk>");
  });

  test("each unseen character gets its own id 0", () => {
    expect(output("encode.each_unseen_character").ids).toEqual([0, 0, 0]);
  });

  test("a word keeps its known part", () => {
    const { tokens, ids } = output("encode.unseen_inside_word");
    expect(tokens).toEqual(["low", "z"]);
    expect(ids[0]).not.toBe(0);
    expect(ids[1]).toBe(0);
  });

  test("the text <unk> is five characters, and is only id 0 if they were unseen", () => {
    expect(output("encode.unk_text_unseen").ids).toEqual([0, 0, 0, 0, 0]);
    expect(output("encode.unk_text_seen").ids).not.toContain(0);
    expect(output("encode.unk_text_seen").decoded).toBe("<unk>");
  });

  test("the text a tokenizer was built from has no id 0", () => {
    expect(output("encode.training_text_has_no_unknown").ids).not.toContain(0);
  });

  test("id 0 is the unknown token", () => {
    expect((bpeCase("build_tokenizer.stepper").outputs.itos as string[])[0]).toBe("<unk>");
  });
});

describe("Part C: the data pipeline", () => {
  const dataCase = (id: string) => loadGolden("data").find((item) => item.id === id)!;
  const windowCases = loadGolden("data").filter(
    ({ mirrors, outputs }) => mirrors === "data.make_windows" && outputs.count !== undefined,
  );

  test.each(windowCases.map((item) => [item.id, item] as const))(
    "the window-count formula matches make_windows: %s",
    (_id, item) => {
      const { ids, seq_len, stride } = item.inputs as {
        ids: number[];
        seq_len: number;
        stride: number;
      };
      expect(windowCount(ids.length, seq_len, stride)).toBe(item.outputs.count);
      expect((item.outputs.inputs as number[][]).length).toBe(item.outputs.count);
    },
  );

  test("the equivalent form in the PDF gives the same count", () => {
    for (const item of windowCases) {
      const {
        ids,
        seq_len: T,
        stride: k,
      } = item.inputs as {
        ids: number[];
        seq_len: number;
        stride: number;
      };
      expect(Math.floor((ids.length - T - 1) / k) + 1).toBe(item.outputs.count);
    }
  });

  test.each(windowCases.map((item) => [item.id, item] as const))(
    "targets equal inputs shifted by one: %s",
    (_id, item) => {
      const inputs = item.outputs.inputs as number[][];
      const targets = item.outputs.targets as number[][];
      inputs.forEach((row, w) => expect(targets[w]!.slice(0, -1)).toEqual(row.slice(1)));
      // The engine behind the interactive table.
      const { ids, seq_len, stride } = item.inputs as {
        ids: number[];
        seq_len: number;
        stride: number;
      };
      const engine = makeWindows(ids, seq_len, stride);
      engine.inputs.forEach((row, w) =>
        expect(engine.targets[w]!.slice(0, -1)).toEqual(row.slice(1)),
      );
    },
  );

  test("make_windows stores DataFrames, and iterate_batches hands out NumPy arrays", () => {
    expect(dataCase("make_windows.test_file").outputs.stored_as).toEqual([
      "DataFrame",
      "DataFrame",
    ]);
    expect(dataCase("iterate_batches.shuffled").outputs.stored_as).toBe("ndarray");
    expect(readPageBody("mathematics/data-pipeline")).toContain("two pandas DataFrames");
  });

  test("the split comes before the windows in train.main", () => {
    const main = loadCallGraph().production["train.main"]!.calls;
    const line = (target: string) => main.find((call) => call.target === target)!.lines;
    expect(Math.max(...line("data.split_ids"))).toBeLessThan(
      Math.min(...line("data.make_windows")),
    );
    expect(Math.max(...line("data.encode_corpus"))).toBeLessThan(
      Math.min(...line("data.split_ids")),
    );
    expect(Math.max(...line("tokenizer.build_tokenizer"))).toBeLessThan(
      Math.min(...line("data.split_ids")),
    );
    expect(requireTarget(CODE_REFS["train.split"]!).endLine).toBeLessThan(
      requireTarget(CODE_REFS["train.windows"]!).startLine,
    );
  });

  test("the page quotes those lines in that order", () => {
    const body = readPageBody("mathematics/data-pipeline");
    const order = ["train.corpus", "train.tokenizer", "train.ids", "train.split", "train.windows"];
    const places = order.map((id) => body.indexOf(`<CodeStep id="${id}"`));
    expect(places.every((place) => place >= 0)).toBe(true);
    expect(places).toEqual([...places].sort((a, b) => a - b));
  });

  test("splitting first removes exactly the T windows that cross the cut", () => {
    for (const item of loadGolden("data").filter(({ mirrors }) => mirrors === "data.split_ids")) {
      const { seq_len } = item.inputs as { seq_len: number };
      const out = item.outputs as Record<string, number>;
      expect(out.crossing).toBe(seq_len);
      expect(out.train_windows! + out.val_windows!).toBe(out.whole_windows! - seq_len);
      expect(out.val_first).toBe(out.train_last! + 1);
    }
  });

  test("iterate_batches keeps a short last batch, and the same seed repeats", () => {
    expect(dataCase("iterate_batches.test_file").outputs.sizes).toEqual([16, 16, 14]);
    const { outputs } = dataCase("iterate_batches.shuffled");
    expect(outputs.pairs_aligned).toBe(true);
    expect(outputs.same_seed_repeats).toBe(true);
    const [first, second] = outputs.epoch_starts as number[][][];
    expect(first!.flat()).not.toEqual(second!.flat());
  });

  test("max_rows applies before missing rows are dropped", () => {
    expect(dataCase("load_corpus.limit_before_dropping").outputs.text).toBe("a\n");
  });

  test("the factorisation stays in words: it has no equation record", () => {
    const body = readPageBody("mathematics/data-pipeline");
    expect(body).toContain("In general,");
    expect(loadConcepts().flatMap(({ equations }) => equations.map(({ id }) => id))).not.toContain(
      "data.factorisation",
    );
  });
});

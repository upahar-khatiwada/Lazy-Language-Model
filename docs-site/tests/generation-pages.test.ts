import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { getSymbol } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { getGoldenCase, loadGolden } from "@/lib/golden";
import { linksForEquation } from "@/lib/links";
import type { Vector } from "@/lib/math/tensor";
import { labSentences, labView } from "@/lib/sampling-lab";
import {
  greedyRows,
  LAB_PRESETS,
  labPresets,
  samplingDefaults,
  stepperCase,
  stepperInputs,
  temperatureRows,
  topKRows,
  topPRows,
} from "@/components/GenerationExamples";
import { stepperSteps } from "@/components/GenerationStepper";

const TOLERANCE = 1e-9;

function expectVector(actual: Vector, expected: Vector, label: string) {
  expect(actual.length, label).toBe(expected.length);
  actual.forEach((value, i) => {
    expect(Math.abs(value - expected[i]!), `${label}[${i}]`).toBeLessThanOrEqual(TOLERANCE);
  });
}

const outputsOf = (id: string) =>
  getGoldenCase("sampling", id).outputs as {
    kind: string;
    index?: number;
    p?: Vector;
    after_temperature?: Vector;
    after_top_k?: Vector;
    cutoff?: number;
    order?: number[];
    cum?: Vector;
    keep_count?: number;
    tie_split?: boolean;
  };

describe("the probability lab equals the real sample_next", () => {
  test.each(LAB_PRESETS.map(({ id }) => id))("%s", (id) => {
    const preset = labPresets().find((candidate) => candidate.id === id)!;
    const outputs = outputsOf(id);
    // Python's own sort order, so that every token must match.
    const view = labView(preset.logits, preset.settings, outputs.order);

    if (outputs.kind === "greedy") {
      expect(view).toEqual({ kind: "greedy", index: outputs.index });
      return;
    }
    if (view.kind !== "distribution") throw new Error(`${id}: expected a distribution`);
    expectVector(
      view.rows.map((row) => row.final),
      outputs.p!,
      "final",
    );
    expectVector(
      view.rows.map((row) => row.afterTemperature),
      outputs.after_temperature!,
      "afterTemperature",
    );
    expectVector(
      view.rows.map((row) => row.afterTopK),
      outputs.after_top_k!,
      "afterTopK",
    );
    expect(view.rows.map((row) => row.kept)).toEqual(outputs.p!.map((value) => value > 0));
    expect(view.topKRan).toBe(outputs.cutoff !== undefined);
    if (outputs.cutoff !== undefined) {
      expect(Math.abs(view.cutoff! - outputs.cutoff)).toBeLessThanOrEqual(TOLERANCE);
    }
    expect(view.topPRan).toBe(outputs.order !== undefined);
    expect(view.tieDependent).toBe(outputs.tie_split ?? false);
    if (outputs.order) {
      view.rows.forEach((row) => {
        expect(outputs.order![row.rank!]).toBe(row.index);
        expect(Math.abs(row.cum! - outputs.cum![row.rank!]!)).toBeLessThanOrEqual(TOLERANCE);
      });
    }
  });

  test("the lab's own tie order keeps the same number of tokens and the same values", () => {
    const preset = labPresets().find((candidate) => candidate.id === "lab.top_p_tie")!;
    const view = labView(preset.logits, preset.settings);
    if (view.kind !== "distribution") throw new Error("expected a distribution");
    const sorted = (values: Vector) => [...values].sort((a, b) => a - b);
    expectVector(
      sorted(view.rows.map((row) => row.final)),
      sorted(outputsOf("lab.top_p_tie").p!),
      "sorted",
    );
    expect(view.tieDependent).toBe(true);
  });

  test("the first preset uses the settings in config.json", () => {
    const value = (name: string) =>
      (
        JSON.parse(readFileSync(join(process.cwd(), "data/generated/config.json"), "utf-8")) as {
          constants: { name: string; value: number }[];
        }
      ).constants.find((constant) => constant.name === name)!.value;
    const { settings } = labPresets()[0]!;
    expect(settings).toEqual({
      temperature: value("temperature"),
      topK: value("top_k"),
      topP: value("top_p"),
    });
  });

  test("the sentences state what the branches did", () => {
    const say = (id: string) => {
      const preset = labPresets().find((candidate) => candidate.id === id)!;
      return labSentences(labView(preset.logits, preset.settings), preset.settings).join(" ");
    };
    expect(say("lab.greedy")).toContain("top_k, top_p and rng are not used");
    expect(say("lab.config_defaults")).toContain("top-k is skipped");
    expect(say("lab.config_defaults")).toContain("the token that crosses it is kept");
    expect(say("lab.no_truncation")).toContain("top-p is skipped");
    expect(say("lab.top_k_tie")).toContain("more than k remain");
    expect(say("lab.top_k")).not.toContain("more than k remain");
    expect(say("lab.top_p_tie")).toContain("the sort order decides");
  });

  test("inputs that overflow are reported, not shown", () => {
    const view = labView([1e308, -1e308], { temperature: 1e-3, topK: 0, topP: 1 });
    expect(view.kind).toBe("not-finite");
  });
});

describe("the tie handling and the prefix rule are as coded", () => {
  test("top-k keeps every token at the cut-off, so more than k can stay", () => {
    const { k, after, cutoff, p } = topKRows("top_k.ties_are_all_kept");
    expect(k).toBe(2);
    expect(after.filter((value) => value > 0)).toHaveLength(3);
    expect(cutoff).toBe(p[0]);
    expectVector(after, outputsOf("top_k.ties_are_all_kept").p!, "after");
  });

  test("top-k with no tie keeps exactly k", () => {
    const { k, after } = topKRows("top_k.two");
    expect(after.filter((value) => value > 0)).toHaveLength(k);
  });

  test("top-p keeps the crossing token", () => {
    const { rows, topP, keepCount } = topPRows("top_p.keeps_crossing_token");
    expect(keepCount).toBe(outputsOf("top_p.keeps_crossing_token").keep_count);
    expect(rows[keepCount - 2]!.cum).toBeLessThan(topP);
    expect(rows[keepCount - 1]!.cum).toBeGreaterThanOrEqual(topP);
    expect(rows[keepCount - 1]!.kept).toBe(true);
    expect(rows[keepCount]!.kept).toBe(false);
  });

  test("top-p stops at the token that lands exactly on the threshold", () => {
    const { rows, topP, keepCount } = topPRows("top_p.exact_boundary");
    expect(keepCount).toBe(2);
    expect(rows[1]!.cum).toBe(topP);
  });

  test("the page describes both as the code does", () => {
    const body = readPageBody("generation/sampling");
    expect(body).toContain("`p >= cutoff`");
    expect(body).toContain("more than `top_k` tokens can remain");
    expect(body).toContain("`np.searchsorted(cum, top_p)`");
    expect(body).toContain("the token at that position is kept");
  });

  test("the code still reads the way the page says", () => {
    const { source } = getSymbol("generate.sample_next");
    expect(source).toContain("cutoff = np.sort(p)[-top_k]");
    expect(source).toContain("p = np.where(p >= cutoff, p, 0.0)");
    expect(source).toContain("keep = order[: np.searchsorted(cum, top_p) + 1]");
    expect(source).toContain("if temperature < 1e-6:");
  });
});

describe("the greedy case", () => {
  test("the engine's rows equal the real function, tie to the first index included", () => {
    for (const row of greedyRows()) {
      expect(row.result, row.id).toBe(outputsOf(row.id).index);
    }
    expect(greedyRows().at(-1)!.result).toBe(1);
  });

  test("exactly 1e-6 is not greedy", () => {
    expect(outputsOf("greedy.at_guard_is_not_greedy").kind).toBe("distribution");
    expect(outputsOf("greedy.below_guard").kind).toBe("greedy");
  });

  test("the page documents it", () => {
    const body = readPageBody("generation/sampling");
    expect(body).toContain("## Greedy guard");
    expect(body).toContain("a greedy step draws nothing from the generator");
  });
});

describe("temperature", () => {
  test("the rows equal the real function", () => {
    for (const row of temperatureRows()) {
      expectVector(row.p, outputsOf(row.id).after_temperature!, row.id);
    }
  });
});

describe("the stepper follows the real generate", () => {
  test.each(["stepper.config_defaults", "stepper.greedy"])("%s", (id) => {
    const { inputs, outputs } = stepperCase(id);
    const steps = stepperSteps(stepperInputs(id));
    expect(steps.map((step) => step.window)).toEqual(outputs.windows);
    expect([...steps.at(-1)!.ids, steps.at(-1)!.next]).toEqual(outputs.ids);
    expect(outputs.ids.slice(0, inputs.seed_ids.length)).toEqual(inputs.seed_ids);
    expect(outputs.ids).toHaveLength(inputs.seed_ids.length + inputs.max_new_tokens);
    // The window fills, then slides.
    expect(steps[0]!.window).toHaveLength(inputs.seed_ids.length);
    expect(steps.at(-1)!.window).toHaveLength(inputs.window);
  });

  test("only the sampled run uses the generator", () => {
    expect(stepperCase("stepper.greedy").outputs.draws).toEqual([]);
    expect(stepperCase("stepper.config_defaults").outputs.draws).toHaveLength(
      stepperCase("stepper.config_defaults").inputs.max_new_tokens,
    );
  });

  test("the sampled run's settings and seed are those in config.py", () => {
    const { inputs } = stepperCase("stepper.config_defaults");
    const row = (name: string) => samplingDefaults().find((item) => item.name === name)!.text;
    expect(String(inputs.temperature)).toBe(row("temperature"));
    expect(String(inputs.top_k)).toBe(row("top_k"));
    expect(String(inputs.top_p)).toBe(row("top_p"));
    expect(String(inputs.seed)).toBe(row("seed"));
  });

  test("every case is labelled Illustrative Example", () => {
    for (const item of loadGolden("sampling")) expect(item.kind).toBe("illustrative");
  });
});

describe("defaults and callers", () => {
  const state = (name: string) => samplingDefaults().find((row) => row.name === name)!.state;

  test("top_k is off and top_p is on", () => {
    expect(state("top_k")).toBe("Off");
    expect(state("top_p")).toBe("On");
    expect(state("temperature")).toBe("On");
  });

  test("both callers pass config.seq_len as the window", () => {
    const train = getSymbol("train.main").source;
    const talk = getSymbol("talk.main").source;
    expect(train).toContain("generate(params, cfg, seed_ids, 30, config.seq_len,");
    expect(train).toContain("seed_ids = [int(i) for i in va_x.iloc[0][:3]]");
    expect(talk).toContain("generate(params, cfg, ids, config.max_new_tokens, config.seq_len,");
  });
});

describe("the concepts and pages", () => {
  const CONCEPTS: [string, string[]][] = [
    ["sampling.greedy", ["sampling.greedy"]],
    ["sampling.temperature", ["sampling.temperature"]],
    ["sampling.top_k", ["top_k.cutoff", "top_k.keep", "top_k.renormalise"]],
    ["sampling.top_p", ["top_p.sort", "top_p.cum", "top_p.prefix", "top_p.renormalise"]],
    ["sampling.choice", ["sampling.choice"]],
    ["generate.generate", ["generate.window", "generate.logits", "generate.append"]],
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

  test.each(CONCEPTS)("%s has its equations, and each one a link that computes it", (id, eqs) => {
    expect(getConcept(id).equations.map((equation) => equation.id)).toEqual(eqs);
    for (const equation of eqs) {
      expect(linksForEquation(equation).some((link) => link.role === "implements")).toBe(true);
    }
  });

  test("the inventory rows M54 to M59 are covered, with M56 and M57 derived", () => {
    const rows = CONCEPTS.flatMap(([id]) => getConcept(id).inventory);
    expect(rows.sort()).toEqual(["M54", "M55", "M56", "M57", "M58", "M59"]);
    expect(getConcept("sampling.top_k").status).toBe("derived");
    expect(getConcept("sampling.top_p").status).toBe("derived");
    expect(getConcept("sampling.temperature").pdfRef?.equations).toEqual(["60"]);
  });

  test.each([
    ["generation/sampling", 5],
    ["generation/autoregressive-generation", 1],
  ])("%s follows the nine steps for each concept", (slug, count) => {
    const found = [...readPageBody(slug).matchAll(/<ConceptStep>(.*?)<\/ConceptStep>/g)].map(
      (match) => match[1],
    );
    expect(found).toEqual(Array.from({ length: count }, () => STEPS).flat());
  });

  test("the pages name no method the code lacks", () => {
    const text = (
      readPageBody("generation/sampling") + readPageBody("generation/autoregressive-generation")
    ).toLowerCase();
    for (const word of ["beam", "repetition penalty", "nucleus", "min-p", "contrastive"]) {
      expect(text).not.toContain(word);
    }
  });

  test("the page states the model's limits", () => {
    const body = readPageBody("generation/autoregressive-generation");
    expect(body).toContain("The model is a text continuation model.");
    expect(body).toContain("It is not an assistant.");
    expect(body).toContain("**This page has no backward pass.**");
  });
});

import { describe, expect, test } from "vitest";
import { TEST_LIMIT } from "@/components/GradcheckExamples";
import { getSymbol, targetLines } from "@/lib/code-targets";
import { getConcept } from "@/lib/concepts";
import { readPageBody } from "@/lib/content";
import { getGoldenCase } from "@/lib/golden";
import { linksForEquation } from "@/lib/links";
import { loadInitParams } from "@/lib/parameters";
import { loadCallGraph } from "@/lib/repo";

const SLUG = "testing/gradient-checking";
const body = readPageBody(SLUG);
const graph = loadCallGraph();
const source = (id: string) => getSymbol(id).source;

const implementing = (id: string) =>
  linksForEquation(id)
    .filter(({ role }) => role === "implements")
    .map(({ resolved }) => targetLines(resolved).trim());

const worst = (id: string) =>
  getGoldenCase("gradcheck", id).outputs.worst as Record<string, number>;
const above = (values: Record<string, number>) =>
  Object.entries(values)
    .filter(([name, value]) => !name.endsWith(".bk") && value >= TEST_LIMIT)
    .map(([name]) => name);

describe("the defaults and the formula match gradcheck.py", () => {
  test("the default h on the page is the default in both functions", () => {
    expect(source("gradcheck.numerical_grad")).toContain("h = 1e-5");
    expect(source("gradcheck.check_params")).toContain("h = 1e-5");
    expect(body).toContain("`h = 1e-5`");
    for (const id of ["gradcheck.numerical_grad", "gradcheck.check_params"]) {
      const row = getConcept(id).symbols.find(({ codeName }) => codeName === "h")!;
      expect(row.meaning).toContain("1e-5");
    }
  });

  test("eps and samples_per_param match the code", () => {
    expect(source("gradcheck.relative_error")).toContain("eps = 1e-12");
    const eps = getConcept("gradcheck.relative_error").symbols.find(
      ({ codeName }) => codeName === "eps",
    )!;
    expect(eps.meaning).toContain("1e-12");
    expect(body).toContain("The guard only helps below `1e-12`.");
    expect(body).toContain("`0 / 1e-12`");
    expect(source("gradcheck.check_params")).toContain("samples_per_param = 5");
    const samples = getConcept("gradcheck.check_params").symbols.find(
      ({ codeName }) => codeName === "samples_per_param",
    )!;
    expect(samples.meaning).toContain("5");
  });

  test("each equation links to the line that computes it", () => {
    expect(implementing("gradcheck.central")).toEqual(["g[idx] = (fp - fm) / (2 * h)"]);
    expect(implementing("gradcheck.relative")).toEqual([
      "return abs(a - b) / max(abs(a), abs(b), eps)",
    ]);
    expect(implementing("gradcheck.worst")).toEqual([
      "w = max(w, relative_error(grads[name][idx], (fp - fm) / (2 * h)))",
      "worst[name] = w",
    ]);
  });

  test("the relative error equation has all three terms in its denominator", () => {
    const { latex } = getConcept("gradcheck.relative_error").equations[0]!;
    expect(latex).toMatch(/\\max\\left\(\|a\|,\\, \|b\|,\\, \\varepsilon\\right\)/);
  });
});

describe("the tolerances come from the tests", () => {
  const FILE_TESTS = [
    "tests.test_gradcheck.test_numerical_grad_on_quadratic",
    "tests.test_gradcheck.test_every_tensor_passes",
    "tests.test_gradcheck.test_key_bias_gradient_is_zero",
  ];

  test("the page quotes exactly the limits the test file has", () => {
    const inTests = new Set(FILE_TESTS.flatMap((id) => source(id).match(/1e-\d+/g) ?? []));
    expect([...inTests].sort()).toEqual(["1e-12", "1e-4", "1e-8"]);
    const table = body.slice(body.indexOf("## What the tests assert"), body.indexOf("**Limits"));
    for (const limit of inTests) expect(table).toContain(limit);
    expect(source(FILE_TESTS[1]!)).toContain("assert err < 1e-4, (name, err)");
    expect(table).toMatch(/test_every_tensor_passes[^\n]*below `1e-4`/);
    expect(source(FILE_TESTS[2]!)).toContain(".max() < 1e-12");
    expect(table).toMatch(/test_key_bias_gradient_is_zero[^\n]*below `1e-12`/);
    expect(source(FILE_TESTS[0]!)).toContain("atol=1e-8");
    expect(table).toMatch(/test_numerical_grad_on_quadratic[^\n]*`atol=1e-8`/);
  });

  test("the limit used to mark the table is the test's", () => {
    expect(source(FILE_TESTS[1]!)).toContain(`err < ${TEST_LIMIT.toExponential()}`);
  });

  test("the PDF's bands do not appear as project rules", () => {
    expect(body).not.toMatch(/1e-6|1e-3/);
  });

  test("the test skips .bk and checks it separately", () => {
    expect(source(FILE_TESTS[1]!)).toContain('name.endswith(".bk")');
    expect(body).toContain("**Why `.bk` is skipped.**");
    expect(body).toContain('<CodeStep id="gradcheck.test-bk" />');
  });
});

describe("the page is labelled and stays out of the pipeline", () => {
  test("no pipeline function reaches gradcheck.py", () => {
    expect(graph.usedBy["gradcheck.numerical_grad"]!.production).toEqual([]);
    expect(graph.usedBy["gradcheck.check_params"]!.production).toEqual([]);
    expect(graph.usedBy["gradcheck.relative_error"]!.production).toEqual([
      "gradcheck.check_params",
    ]);
    expect(body).toContain("This whole page is test-only mathematics.");
  });

  test("every function of gradcheck.py has its code and its concept on the page", () => {
    for (const id of [
      "gradcheck.numerical_grad",
      "gradcheck.relative_error",
      "gradcheck.check_params",
    ]) {
      expect(body).toContain(`<CodeReference symbol="${id}"`);
      expect(body).toContain(`<ConceptCard concept="${id}"`);
      expect(body).toContain(`<Trace concept="${id}"`);
      expect(getConcept(id).category).toBe("gradient-checking");
      expect(getConcept(id).status).toBe("implemented");
    }
  });

  test("the general explanation is labelled", () => {
    expect(body).toMatch(/In general, a central difference[^]*This is a general explanation\./);
  });

  test("no example is labelled as model output", () => {
    expect(body).not.toContain('kind="actual"');
  });
});

describe("reading a failure", () => {
  const table = body.slice(body.indexOf("## Reading a failure"));

  test("the name table covers every parameter name init_params creates", () => {
    const { parameters } = loadInitParams();
    expect(parameters.length).toBeGreaterThan(0);
    for (const { name } of parameters) expect(table, name).toContain(`\`${name}\``);
  });

  test("with correct gradients, every checked name is below the limit", () => {
    expect(above(worst("check_params.correct"))).toEqual([]);
  });

  test("scaling one gradient flags exactly that name", () => {
    expect(above(worst("check_params.one_tensor_wrong"))).toEqual(["b1.wq"]);
  });

  test("a wrong block output flags only what is computed after it", () => {
    const flagged = above(worst("check_params.block_output_wrong"));
    expect(flagged.slice(0, 2)).toEqual(["tok_emb", "pos_emb"]);
    expect(flagged.slice(2).every((name) => name.startsWith("b0."))).toBe(true);
    expect(flagged.some((name) => name.startsWith("b1."))).toBe(false);
    expect(flagged).not.toContain("head_w");
    expect(flagged).not.toContain("lnf_g");
  });

  test("the first failure in the loop's order is the name the page gives", () => {
    const { names } = getGoldenCase("gradcheck", "check_params.block_output_wrong").outputs as {
      names: string[];
    };
    const flagged = above(worst("check_params.block_output_wrong"));
    expect(names.find((name) => flagged.includes(name))).toBe("tok_emb");
    expect(table).toContain("names only `tok_emb`");
  });

  test("the guard cases say what the page says", () => {
    const error = (id: string) => getGoldenCase("gradcheck", id).outputs.error as number;
    expect(error("relative_error.both_zero")).toBe(0);
    expect(error("relative_error.guard_used")).toBeCloseTo(0.1, 12);
    expect(error("relative_error.tiny_against_small")).toBeGreaterThan(0.99);
    expect(error("relative_error.opposite_signs")).toBe(2);
  });

  test("h changes the cube's error in both directions, and the square's only by round-off", () => {
    const error = (id: string) => getGoldenCase("gradcheck", id).outputs.error as number;
    const middle = error("numerical_grad.cube_default_h");
    expect(error("numerical_grad.cube_large_h")).toBeGreaterThan(middle);
    expect(error("numerical_grad.cube_tiny_h")).toBeGreaterThan(middle);
    expect(error("numerical_grad.square_large_h")).toBeLessThan(1e-12);
  });
});

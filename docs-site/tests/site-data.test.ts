import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { CODE_REFS } from "@/data/code-refs";
import { GLANCE } from "@/data/home";
import { PIPELINE } from "@/data/pipeline";
import { loadSymbolIndex } from "@/lib/code-targets";
import { buildGlance } from "@/lib/model-glance";
import { navSlugs, writtenHref } from "@/lib/pages";
import {
  countFormulaTex,
  countInV,
  evaluateCount,
  loadInitParams,
  realCount,
  realShape,
  symbolicShape,
} from "@/lib/parameters";
import {
  loadCallGraph,
  loadCodeIndex,
  loadConfig,
  loadRepoMeta,
  type RepoConfig,
} from "@/lib/repo";
import { verifySiteData } from "@/lib/site-verify";

const base = () => ({
  symbols: loadSymbolIndex(),
  callGraph: loadCallGraph(),
  config: loadConfig(),
  navSlugs: navSlugs(),
  parameterNames: loadInitParams().parameters.map((parameter) => parameter.name),
  repoFiles: loadCodeIndex().files.map((file) => file.path),
});

test("hand-written site data matches the code at the pinned commit", () => {
  expect(verifySiteData(base())).toEqual([]);
});

describe("the model at a glance", () => {
  const values = (config: RepoConfig) =>
    new Map(
      buildGlance(GLANCE, config, loadRepoMeta())
        .flatMap((group) => group.rows)
        .map((row) => [row.name, row.value]),
    );

  test("shows the values in config.json, including the derived head width", () => {
    const shown = values(loadConfig());
    const config = loadConfig().constants;
    expect(shown.get("emb_dim")).toBe(config.find((c) => c.name === "emb_dim")?.valueText);
    expect(shown.get("head_dim")).toBe(String(config.find((c) => c.name === "head_dim")?.value));
    expect(shown.get("lr_max")).toBe(config.find((c) => c.name === "lr_max")?.valueText);
  });

  test("follows a changed value in a copy of config.json", () => {
    const copy = structuredClone(loadConfig());
    copy.constants.find((c) => c.name === "num_layers")!.valueText = "7";
    expect(values(copy).get("num_layers")).toBe("7");
  });

  test("a row naming a constant that does not exist fails", () => {
    const bad = [{ title: "x", rows: [{ label: "x", config: "nope" }] }];
    expect(() => buildGlance(bad, loadConfig(), loadRepoMeta())).toThrow("no constant named nope");
  });

  test("links each value to its line in config.py", () => {
    const repo = loadRepoMeta();
    const row = buildGlance(GLANCE, loadConfig(), repo)
      .flatMap((group) => group.rows)
      .find((candidate) => candidate.name === "emb_dim")!;
    const { line } = loadConfig().constants.find((c) => c.name === "emb_dim")!;
    expect(row.permalink).toBe(`${repo.githubBaseUrl}/blob/${repo.sha}/config.py#L${line}`);
  });

  test("flags constants that no module reads", () => {
    const rows = buildGlance(GLANCE, loadConfig(), loadRepoMeta()).flatMap((group) => group.rows);
    expect(rows.filter((row) => row.unread).map((row) => row.name)).toEqual(["head_dim"]);
  });
});

describe("the pipeline", () => {
  const nodes = PIPELINE.flatMap((phase) => phase.nodes);

  test("has the steps of the real code, in order", () => {
    expect(nodes.map((node) => node.id)).toEqual([
      "text",
      "tokenizer",
      "ids",
      "windows",
      "embeddings",
      "ln1",
      "attention",
      "residual1",
      "ln2",
      "ffn",
      "residual2",
      "final-ln",
      "head",
      "logits",
      "loss",
      "backward",
      "clipping",
      "adamw",
      "generation",
    ]);
  });

  const withNode = (change: (node: (typeof nodes)[number]) => object) =>
    PIPELINE.map((phase) => ({
      ...phase,
      nodes: phase.nodes.map((node) => (node.id === "ln1" ? { ...node, ...change(node) } : node)),
    }));

  test.each([
    ["a line that is gone", { target: { ...nodes[5]!.target, match: "nope" } }, "not found"],
    ["a wrong call", { call: "ops.relu_forward" }, "does not call ops.relu_forward"],
    ["an unknown page", { page: "nope/nope" }, "not in the navigation"],
    [
      "an inline text not in the line",
      { call: undefined, inline: "nope" },
      "is not in the resolved",
    ],
    ["both call and inline", { inline: "X" }, "exactly one of call and inline"],
  ])("rejects %s", (_name, change, message) => {
    const problems = verifySiteData({ ...base(), pipeline: withNode(() => change) });
    expect(problems.join("\n")).toContain(message);
  });

  test("rejects steps in the wrong order", () => {
    const [text, tokenizer, ...rest] = PIPELINE[0]!.nodes;
    const swapped = [
      { ...PIPELINE[0]!, nodes: [tokenizer!, text!, ...rest] },
      ...PIPELINE.slice(1),
    ];
    const problems = verifySiteData({ ...base(), pipeline: swapped });
    expect(problems.join("\n")).toContain("comes before the previous node");
  });
});

test("writtenHref is null until the page exists, and unknown pages fail", () => {
  expect(writtenHref("mathematics/tokenization")).toBe("/mathematics/tokenization/");
  expect(writtenHref("code/code-explorer")).toBeNull();
  expect(() => writtenHref("nope/nope")).toThrow("not in the navigation");
});

describe("the walkthrough and the forward diagram", () => {
  test("How This LLM Works shows every train and talk step, in code order", () => {
    const mdx = readFileSync(
      join(process.cwd(), "content", "getting-started", "how-this-llm-works.mdx"),
      "utf-8",
    );
    const used = [...mdx.matchAll(/<CodeStep id="([^"]+)"/g)].map((match) => match[1]);
    expect(used).toEqual(Object.keys(CODE_REFS).filter((id) => /^(train|talk)\./.test(id)));
  });

  test("rejects a walkthrough out of line order", () => {
    const codeRefs = {
      "train.b": { file: "train.py", symbol: "train.main", match: "ids = encode_corpus(" },
      "train.a": { file: "train.py", symbol: "train.main", match: "tok = build_tokenizer(" },
    };
    expect(verifySiteData({ ...base(), codeRefs }).join("\n")).toContain(
      "comes before the previous train step",
    );
  });

  test("every box in the diagram is a call or an expression in forward or block_forward", () => {
    const forward = PIPELINE.filter((phase) => phase.forward);
    expect(forward.map((phase) => phase.id)).toEqual(["embeddings", "blocks", "output"]);
    for (const phase of forward) {
      for (const node of phase.nodes) {
        expect(["transformer.forward", "transformer.block_forward"]).toContain(node.target.symbol);
      }
    }
    // And the other way round: every function those two call has a box.
    const graph = loadCallGraph().production;
    const boxes = forward.flatMap((phase) => [
      ...phase.nodes.flatMap((node) => (node.call ? [node.call] : [])),
      ...(phase.repeat ? [phase.repeat.call] : []),
    ]);
    for (const caller of ["transformer.forward", "transformer.block_forward"]) {
      for (const { target } of graph[caller]!.calls) expect(boxes).toContain(target);
    }
  });
});

describe("the parameter table", () => {
  const { parameters, probe, countParams } = loadInitParams();

  test("the count worked out from shapes equals count_params", () => {
    expect(evaluateCount(parameters, probe)).toBe(countParams);
  });

  test("the formula is collected from the shapes", () => {
    expect(countFormulaTex(parameters)).toBe(
      "2V d + T_{\\max} d + 2d + V + L\\left(4d^{2} + 2d F + 9d + F\\right)",
    );
  });

  test("follows a changed value, and V stays symbolic", () => {
    const values = { d: 8, F: 16, Tmax: 4 };
    expect(realShape(["V", "d"], values)).toBe("(V, 8)");
    expect(realShape(["Tmax", "d"], values)).toBe("(4, 8)");
    expect(symbolicShape(["Tmax", "d"])).toBe("(T_max, d)");
    expect(realCount(["V"], values)).toBe("V");
    expect(realCount(["d", "V"], values)).toBe("8V");
    const { perVocabulary } = countInV(parameters, { ...values, L: 1 });
    expect(perVocabulary).toBe(2 * 8 + 1);
  });

  test("a missing or invented name is reported", () => {
    const names = parameters.map((parameter) => parameter.name);
    const missing = verifySiteData({ ...base(), parameterNames: [...names, "extra"] });
    expect(missing.join("\n")).toContain("nothing declared for extra");
    const invented = verifySiteData({ ...base(), parameterNames: names.slice(1) });
    expect(invented.join("\n")).toContain("tok_emb does not exist");
  });

  test("the repository map covers exactly the real files", () => {
    const files = loadCodeIndex().files.map((file) => file.path);
    const extra = verifySiteData({ ...base(), repoFiles: [...files, "nope.py"] });
    expect(extra.join("\n")).toContain("nothing declared for nope.py");
  });
});

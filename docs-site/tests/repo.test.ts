import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  loadCallGraph,
  loadCodeIndex,
  loadConfig,
  loadRepoMeta,
  loadTestsIndex,
  type FunctionSymbol,
} from "@/lib/repo";

// These tests compare generated data with the real repository files, without any Python.
const repoRoot = join(process.cwd(), "..");
const sourceLines = (path: string) =>
  readFileSync(join(repoRoot, path), "utf-8").replace(/\r\n/g, "\n").split("\n");

const files = loadCodeIndex().files;
const fileNamed = (path: string) => {
  const file = files.find((candidate) => candidate.path === path);
  if (!file) throw new Error(`${path} is not in code-index.json`);
  return file;
};

describe("code-index.json", () => {
  test("every symbol's source and lines match the real file", () => {
    let checked = 0;
    for (const file of files) {
      const lines = sourceLines(file.path);
      const symbols: FunctionSymbol[] = [...file.functions];
      for (const cls of file.classes) {
        const slice = lines.slice(cls.startLine - 1, cls.endLine).join("\n");
        expect(cls.source, cls.id).toBe(slice);
        expect(lines[cls.defLine - 1]?.trimStart(), cls.id).toMatch(/^class /);
        symbols.push(...cls.methods);
        checked += 1;
      }
      for (const symbol of symbols) {
        const slice = lines.slice(symbol.startLine - 1, symbol.endLine).join("\n");
        expect(symbol.source, symbol.id).toBe(slice);
        expect(lines[symbol.startLine - 1]?.trimStart(), symbol.id).toMatch(/^(def |async def |@)/);
        expect(lines[symbol.defLine - 1]?.trimStart(), symbol.id).toMatch(/^(def |async def )/);
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(0);
  });

  test("every def and class in every file is indexed", () => {
    for (const file of files) {
      const found = sourceLines(file.path).filter((text) =>
        /^\s*(async def|def|class) /.test(text),
      );
      const methods = file.classes.flatMap((cls) => cls.methods);
      expect(file.functions.length + file.classes.length + methods.length, file.path).toBe(
        found.length,
      );
    }
  });

  test("permalinks use the pinned sha and the symbol's line range", () => {
    const meta = loadRepoMeta();
    const symbol = fileNamed("ops.py").functions.find((fn) => fn.name === "linear_forward");
    expect(symbol?.permalink).toBe(
      `${meta.githubBaseUrl}/blob/${meta.sha}/ops.py#L${symbol?.startLine}-L${symbol?.endLine}`,
    );
  });

  test("ModelConfig is captured as a dataclass with its fields", () => {
    const modelConfig = fileNamed("transformer.py").classes.find((c) => c.name === "ModelConfig");
    expect(modelConfig?.isDataclass).toBe(true);
    expect(modelConfig?.decorators).toEqual(["@dataclass(frozen=True)"]);
    expect(modelConfig?.fields.map((f) => f.name)).toEqual([
      "vocab_size",
      "context_len",
      "emb_dim",
      "num_heads",
      "ff_dim",
      "num_layers",
      "ln_eps",
    ]);
    expect(modelConfig?.fields.at(-1)?.default).toBe("1e-5");
  });

  test("Adam is captured with its methods", () => {
    const adam = fileNamed("optim.py").classes.find((c) => c.name === "Adam");
    expect(adam?.methods.map((m) => m.name)).toEqual(["__init__", "step"]);
  });

  test("the extractor does not index docs-site", () => {
    expect(files.some((file) => file.path.startsWith("docs-site/"))).toBe(false);
  });
});

describe("config.json", () => {
  const constants = loadConfig().constants;

  test("contains every assignment in config.py, with its value and comment", () => {
    const lines = sourceLines("config.py");
    const assigned = lines.flatMap((text, index) => {
      const match = /^(\w+)\s*=\s*(.*?)\s*#\s*(.*)$/.exec(text);
      return match
        ? [{ name: match[1], valueText: match[2], line: index + 1, comment: match[3] }]
        : [];
    });
    expect(assigned.length).toBeGreaterThan(0);
    expect(
      constants.map(({ name, valueText, line, comment }) => ({ name, valueText, line, comment })),
    ).toEqual(assigned);
  });

  test("head_dim is derived from emb_dim and num_heads", () => {
    const value = (name: string) => constants.find((c) => c.name === name)?.value;
    const headDim = constants.find((c) => c.name === "head_dim");
    expect(headDim?.derived).toBe(true);
    expect(headDim?.value).toBe(Math.floor(Number(value("emb_dim")) / Number(value("num_heads"))));
  });

  test("values with no effect are read by nothing", () => {
    for (const name of ["ln_eps", "head_dim"]) {
      expect(constants.find((c) => c.name === name)?.readBy).toEqual([]);
    }
  });
});

describe("call-graph.json", () => {
  const graph = loadCallGraph();

  test("ops.linear_forward is used only by tests", () => {
    expect(graph.usedBy["ops.linear_forward"]).toEqual({
      production: [],
      tests: ["tests.test_ops.test_linear_worked_example"],
    });
  });

  test("methods on inferred types resolve", () => {
    expect(graph.usedBy["optim.Adam.step"]?.production).toEqual(["train.main"]);
    expect(graph.usedBy["tokenizer.Tokenizer.encode"]?.production).toContain("talk.main");
  });

  test("callers are split by test and non-test module", () => {
    expect(Object.keys(graph.production).some((id) => id.startsWith("tests."))).toBe(false);
    expect(Object.keys(graph.tests).every((id) => id.startsWith("tests."))).toBe(true);
  });

  test("reachability matches REPO_INVENTORY.md section 5", () => {
    const seen = new Set<string>();
    const pending = ["train.main", "talk.main"];
    while (pending.length > 0) {
      const id = pending.pop();
      if (id === undefined || seen.has(id)) continue;
      seen.add(id);
      pending.push(...(graph.production[id]?.calls.map((call) => call.target) ?? []));
    }
    const rootSymbols = Object.keys(graph.usedBy).filter((id) => !id.startsWith("tests."));
    expect(rootSymbols.filter((id) => !seen.has(id)).sort()).toEqual([
      "gradcheck.check_params",
      "gradcheck.numerical_grad",
      "gradcheck.relative_error",
      "ops.linear_forward",
      "transformer.count_params",
    ]);
  });
});

describe("tests-index.json", () => {
  const testFiles = loadTestsIndex().files;
  const tests = testFiles.flatMap((file) => file.tests);

  test("totals match REPO_INVENTORY.md section 3", () => {
    expect(testFiles).toHaveLength(8);
    expect(tests).toHaveLength(64);
    expect(tests.reduce((total, t) => total + (t.cases ?? 0), 0)).toBe(67);
  });

  test("fixtures and helpers are followed", () => {
    const byName = (name: string) => tests.find((t) => t.name === name);
    expect(byName("test_unk_is_id_zero")?.fixtures).toEqual(["tests.test_tokenizer.tok"]);
    expect(byName("test_unk_is_id_zero")?.reaches).toEqual(["tokenizer.build_tokenizer"]);
    expect(byName("test_every_tensor_passes")?.helpers).toEqual([
      "tests.test_gradcheck.tiny_setup",
    ]);
    expect(byName("test_every_tensor_passes")?.reaches).toContain("transformer.init_params");
  });
});

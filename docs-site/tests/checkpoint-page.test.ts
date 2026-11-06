import { readFileSync } from "node:fs";
import { join } from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import { checkpointCase, TestCoverage, UntestedSymbols } from "@/components/CheckpointExamples";
import { CHECKPOINT_FILES } from "@/data/checkpoint";
import { CODE_REFS } from "@/data/code-refs";
import { testCoverageRows, untestedSymbols } from "@/lib/code-explorer";
import { getSymbol, requireTarget, targetLines } from "@/lib/code-targets";
import { readPageBody } from "@/lib/content";
import { loadCodeIndex, loadTestsIndex } from "@/lib/repo";

const body = readPageBody("code/checkpoints-tests");

const extensionsIn = (id: string) =>
  [...getSymbol(id).source.matchAll(/prefix \+ "([^"]+)"/g)].map((match) => match[1]);

describe("every checkpoint file the code writes is described", () => {
  const described = CHECKPOINT_FILES.map((file) => file.extension);

  test("the list equals what save_checkpoint writes and load_checkpoint reads", () => {
    expect(extensionsIn("checkpoint.save_checkpoint")).toEqual(described);
    expect(extensionsIn("checkpoint.load_checkpoint")).toEqual(described);
  });

  test("the real function wrote those files, in that order", () => {
    const { inputs, outputs } = checkpointCase();
    expect(outputs.writes).toEqual(described.map((extension) => inputs.prefix + extension));
    expect(outputs.folders).toEqual(["checkpoints"]);
  });

  test("each row's write line names its extension or the file it was opened for", () => {
    const write = requireTarget(CODE_REFS["checkpoint.save-weights"]!);
    expect(targetLines(write)).toContain('prefix + ".npz"');
    for (const { write: ref, read } of CHECKPOINT_FILES) {
      expect(CODE_REFS[ref], ref).toBeDefined();
      expect(CODE_REFS[read], read).toBeDefined();
    }
  });

  test("the .json file holds exactly the fields of ModelConfig", () => {
    const fields = loadCodeIndex()
      .files.flatMap((file) => file.classes)
      .find((cls) => cls.id === "transformer.ModelConfig")!
      .fields.map((field) => field.name);
    expect(Object.keys(JSON.parse(checkpointCase().outputs.config_json))).toEqual(fields);
  });

  test("the .tok.json file holds merges and itos, and merges have the three keys", () => {
    const tokenizer = JSON.parse(checkpointCase().outputs.tokenizer_json);
    expect(Object.keys(tokenizer)).toEqual(["merges", "itos"]);
    expect(Object.keys(tokenizer.merges[0])).toEqual(["pair", "merged", "frequency"]);
    expect(tokenizer.itos[0]).toBe("<unk>");
  });

  test("the .npz keys are the parameter names, kept as they are", () => {
    const keys = checkpointCase().outputs.arrays.map((array) => array.key);
    expect(keys).toContain("tok_emb");
    expect(keys).toContain("b0.wq");
    expect(keys).toContain("head_b");
  });

  test("the page names every file and says what is not saved, as in the code", () => {
    for (const extension of [".npz", ".json", ".tok.json"]) {
      expect(body).toContain(`checkpoints/model${extension}`);
    }
    const save = getSymbol("checkpoint.save_checkpoint").source;
    expect(save).toMatch(/def save_checkpoint\(prefix, params, cfg: ModelConfig, tok: Tokenizer\)/);
    expect(save).not.toMatch(/opt|rng|step/);
  });

  test(".gitignore excludes what the page says it excludes", () => {
    const ignored = readFileSync(join(process.cwd(), "..", ".gitignore"), "utf-8");
    expect(ignored).toContain("checkpoints/");
    expect(ignored).toContain("/data/");
    expect(body).toContain("`.gitignore` lists `checkpoints/` and `/data/`");
  });
});

describe("the inference walkthrough follows talk.main", () => {
  test("its steps come in the order of the code", () => {
    const section = body.slice(body.indexOf("## Interactive inference"));
    const refs = [...section.matchAll(/<CodeLine id="([^"]+)"/g)].map((match) => match[1]!);
    const steps = refs.filter((id) => /^(talk|inference)\.|talk-quit/.test(id));
    const lines = steps.map((id) => requireTarget(CODE_REFS[id]!).startLine);
    expect(lines).toEqual([...lines].sort((a, b) => a - b));
    expect(new Set(lines).size).toBe(lines.length);
  });

  test("only the new tokens are decoded", () => {
    expect(targetLines(requireTarget(CODE_REFS["talk.print"]!))).toContain("out[len(ids):]");
  });
});

describe("the reproduction note states what the code does", () => {
  test("train.py passes no arguments, so the prefix is the default", () => {
    expect(getSymbol("train.main").source).toContain('def main(prefix="checkpoints/model")');
    const train = loadCodeIndex().files.find((file) => file.path === "train.py")!;
    expect(train.source).toMatch(/if __name__ == "__main__":\n\s+main\(\)/);
  });

  test("talk.py takes the prefix from the command line", () => {
    const talk = loadCodeIndex().files.find((file) => file.path === "talk.py")!;
    expect(talk.source).toContain("main(*sys.argv[1:])");
  });
});

describe("every test file appears with accurate coverage notes", () => {
  const index = loadTestsIndex().files;
  const rows = testCoverageRows();

  test("one row per file in tests-index.json", () => {
    expect(rows.map((row) => row.path)).toEqual(index.map((file) => file.path));
    expect(rows.length).toBeGreaterThan(0);
  });

  test.each(index.map((file) => [file.path, file] as const))("%s", (path, file) => {
    const row = rows.find((candidate) => candidate.path === path)!;
    expect(row.testCount).toBe(file.tests.length);
    expect(row.exercised).toEqual([...new Set(file.tests.flatMap((t) => t.reaches))].sort());
    expect(row.exercised.length).toBeGreaterThan(0);
    for (const test of row.tests) expect(test.purpose, test.id).not.toBeNull();
  });

  test("no test has parameters that the index could not count", () => {
    for (const file of index) for (const t of file.tests) expect(t.cases, t.id).not.toBeNull();
  });

  test("the rendered table shows each file's count and functions", () => {
    const html = renderToStaticMarkup(TestCoverage());
    for (const row of rows) {
      const start = html.indexOf(`data-test-file="${row.path}"`);
      expect(start, row.path).toBeGreaterThan(-1);
      const end = html.indexOf("</tr>", start);
      const part = html.slice(start, end);
      expect(part).toContain(`data-test-count="true">${row.testCount}`);
      for (const id of row.exercised) expect(part, id).toContain(`>${id}</code>`);
      for (const test of row.tests) expect(part, test.id).toContain(`#${test.name}`);
    }
  });

  test("the page states no pass count", () => {
    expect(body).not.toMatch(/\d+ passed/);
  });

  test("the untested list is the contract's", () => {
    expect(untestedSymbols().sort()).toEqual(
      [
        "data.encode_corpus",
        "talk.main",
        "train.batch_loss_and_grads",
        "train.main",
        "train.validation_loss",
      ].sort(),
    );
    const html = renderToStaticMarkup(UntestedSymbols());
    for (const id of untestedSymbols()) expect(html).toContain(`>${id}</code>`);
  });
});

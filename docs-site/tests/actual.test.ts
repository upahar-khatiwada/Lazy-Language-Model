import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, test, vi } from "vitest";
import { ActualAttention } from "@/components/ActualAttention";
import {
  ActualEmbeddingsPanel,
  ActualGenerationPanel,
  ActualTokenizerPanel,
} from "@/components/ActualPanels";
import { ActualModelOutputSlot, AttentionLabExample } from "@/components/AttentionExamples";
import { AttentionLab } from "@/components/AttentionLab";
import * as actualData from "@/lib/actual";
import {
  loadActual,
  provenanceLine,
  readActual,
  type ActualData,
  type NoActualData,
} from "@/lib/actual";
import {
  attentionProblems,
  causalMask,
  tokenLabels,
  visibleToken,
  type ActualAttentionData,
} from "@/lib/actual-view";
import { LAB_TOKENS, type LabInputs } from "@/lib/attention-lab";
import { getGoldenCase } from "@/lib/golden";

const render = (element: ReactElement) => renderToStaticMarkup(element);

afterEach(() => {
  vi.restoreAllMocks();
});

const NO_DATA: NoActualData = {
  available: false,
  reason: "No checkpoint at somewhere.",
  howToCreate: ["python docs-site/scripts/export_actual.py --train"],
};

// A small correct matrix, so these tests do not depend on a checkpoint being on this machine.
const fixture: ActualAttentionData = {
  window: {
    split: "validation",
    index: 0,
    ids: [1, 2],
    tokens: ["zz", "\n"],
    targets: [2, 3],
    text: "zz\n",
  },
  T: 2,
  H: 2,
  L: 1,
  dh: 16,
  largestRowSumGap: 0,
  layers: [
    {
      layer: 0,
      heads: [
        {
          head: 0,
          A: [
            [1, 0],
            [0.25, 0.75],
          ],
        },
        {
          head: 1,
          A: [
            [1, 0],
            [0.5, 0.5],
          ],
        },
      ],
    },
  ],
};

const breaking = (change: (data: ActualAttentionData) => void) => {
  const copy = structuredClone(fixture);
  change(copy);
  return attentionProblems(copy).join("\n");
};

describe("what is checked about exported attention weights", () => {
  test("a correct matrix has no problems", () => {
    expect(attentionProblems(fixture)).toEqual([]);
  });

  test.each<[string, (data: ActualAttentionData) => void, string]>([
    ["a row off 1", (d) => (d.layers[0]!.heads[0]!.A[1]![1] = 0.8), "row 1: sums to"],
    ["a masked cell not 0", (d) => (d.layers[0]!.heads[1]!.A[0] = [0.5, 0.5]), "masked cell 0,1"],
    ["a missing head", (d) => d.layers[0]!.heads.pop(), "1 heads, H is 2"],
    ["a missing layer", (d) => d.layers.pop(), "0 layers, L is 1"],
    ["a missing row", (d) => d.layers[0]!.heads[0]!.A.pop(), "1 rows, T is 2"],
    ["a short row", (d) => (d.layers[0]!.heads[0]!.A[0] = [1]), "row 0: 1 columns"],
    ["a token count off T", (d) => d.window.tokens.pop(), "1 tokens, T is 2"],
  ])("%s is caught", (_name, change, expected) => {
    expect(breaking(change)).toContain(expected);
  });

  test("the mask covers exactly the cells after the row", () => {
    expect(causalMask(3)).toEqual([
      [false, true, true],
      [false, false, true],
      [false, false, false],
    ]);
  });

  test("labels stay unique when tokens repeat, and blank tokens are visible", () => {
    const labels = tokenLabels([" ", "the", " ", "the", "\n"]);
    expect(new Set(labels).size).toBe(5);
    expect(labels[0]).toBe("0 ␠");
    expect(visibleToken("\n")).toBe("\\n");
  });
});

describe("reading the export", () => {
  const folderWith = (meta: unknown) => {
    const folder = mkdtempSync(join(tmpdir(), "actual-"));
    writeFileSync(join(folder, "meta.json"), JSON.stringify(meta));
    return folder;
  };

  test("no folder means no data, not a failed build", () => {
    expect(readActual(join(tmpdir(), "no-such-actual-folder")).available).toBe(false);
  });

  test("actualAvailable false gives the reason and how to create a checkpoint", () => {
    const folder = folderWith({
      schema: 1,
      actualAvailable: false,
      reason: "None here.",
      howToCreate: ["run it"],
    });
    expect(readActual(folder)).toEqual({
      available: false,
      reason: "None here.",
      howToCreate: ["run it"],
    });
    rmSync(folder, { recursive: true });
  });

  test("a malformed file fails loudly", () => {
    const folder = folderWith({ schema: 1, actualAvailable: true });
    expect(() => readActual(folder)).toThrow("Invalid data");
    rmSync(folder, { recursive: true });
  });
});

describe("with no data", () => {
  const inputs = getGoldenCase("attention", "attention_forward.cat_sat").inputs as LabInputs;
  const code = {
    lines: [],
    source: "",
    filePath: "transformer.py",
    startLine: 1,
    permalink: "https://example.test",
  };

  test("the lab keeps the switch disabled, with the existing explanation", () => {
    const html = render(
      createElement(AttentionLab, {
        inputs,
        code,
        stepLines: {} as never,
        actual: null,
      }),
    );
    expect(html).toMatch(/<input[^>]*type="radio"[^>]*disabled=""/);
    expect(html).toContain("Actual Model Output is not available yet.");
    expect(html).not.toContain('data-testid="actual-attention"');
  });

  test("the slot and every panel explain instead of showing numbers", () => {
    vi.spyOn(actualData, "loadActual").mockReturnValue(NO_DATA);
    const slot = render(createElement(ActualModelOutputSlot));
    expect(slot).toContain("No data yet");
    expect(slot).toContain("disabled");
    for (const panel of [ActualTokenizerPanel, ActualEmbeddingsPanel, ActualGenerationPanel]) {
      const html = render(createElement(panel));
      expect(html).toContain('data-testid="actual-unavailable"');
      expect(html).toContain("No checkpoint at somewhere.");
      expect(html).toContain("export_actual.py");
    }
  });

  test("the server wrapper passes nothing to the lab", async () => {
    vi.spyOn(actualData, "loadActual").mockReturnValue(NO_DATA);
    const element = await AttentionLabExample();
    expect((element.props as { actual: unknown }).actual).toBeNull();
  });
});

describe("the two sources never mix", () => {
  const inputs = getGoldenCase("attention", "attention_forward.cat_sat").inputs as LabInputs;
  const code = {
    lines: [],
    source: "",
    filePath: "transformer.py",
    startLine: 1,
    permalink: "https://example.test",
  };

  test("with data the lab opens on Illustrative, with no exported value on the page", () => {
    const html = render(
      createElement(AttentionLab, {
        inputs,
        code,
        stepLines: { scores: [], scale: [], mask: [], softmax: [], output: [] } as never,
        actual: { data: fixture, provenance: "P" },
      }),
    );
    expect(html).not.toMatch(/<input[^>]*data-testid="lab-source-actual"[^>]*disabled/);
    expect(html).not.toContain('data-testid="actual-attention"');
    expect(html).not.toContain("zz");
  });

  test("the Actual side shows no illustrative token and no Illustrative label", () => {
    const html = render(createElement(ActualAttention, { data: fixture, provenance: "P-LINE" }));
    expect(html).toContain("Actual Model Output");
    expect(html).not.toContain("Illustrative Example");
    expect(html).toContain("P-LINE");
    for (const token of LAB_TOKENS) expect(html).not.toContain(`>${token}<`);
  });

  test("the Actual component takes no illustrative input, by its source", () => {
    const source = readFileSync("components/ActualAttention.tsx", "utf-8");
    expect(source).not.toContain("attention-lab");
    expect(source).not.toContain('kind="illustrative"');
  });

  test("it makes no claim about what the model focuses on", () => {
    const source = [
      readFileSync("components/ActualAttention.tsx", "utf-8"),
      readFileSync("components/ActualPanels.tsx", "utf-8"),
    ].join("\n");
    expect(source).not.toMatch(/focus(es)? on|attends to|pays attention|looks at/i);
  });
});

// These run on what is in the repository, and hold for whichever export that is.
describe("the export in this repository", () => {
  const actual = loadActual();

  test("is either available with correct matrices, or explains itself", () => {
    if (!actual.available) {
      expect(actual.reason.length).toBeGreaterThan(0);
      expect(actual.howToCreate.length).toBeGreaterThan(0);
      return;
    }
    expect(attentionProblems(actual.attention)).toEqual([]);
    const { T, H, L } = actual.attention;
    expect(actual.attention.layers).toHaveLength(L);
    for (const { heads } of actual.attention.layers) {
      expect(heads).toHaveLength(H);
      for (const { A } of heads) {
        expect(A).toHaveLength(T);
        A.forEach((row, i) => {
          expect(row).toHaveLength(T);
          expect(row.reduce((total, value) => total + value, 0)).toBeCloseTo(1, 9);
          row.forEach((value, j) => {
            if (j > i) expect(value).toBe(0);
            else expect(value).toBeGreaterThanOrEqual(0);
          });
        });
      }
    }
  });

  // Skipped when no checkpoint backs the site, as a fresh clone may have.
  const withData = actual.available ? test : test.skip;
  const data = actual as ActualData;

  withData("the sizes agree with each other and with the checkpoint's model shape", () => {
    const { modelConfig } = data.meta.provenance;
    expect(data.meta.vocabSize).toBe(modelConfig.vocab_size);
    expect(data.attention.T).toBe(data.embeddings.T);
    expect(data.attention.H).toBe(modelConfig.num_heads);
    expect(data.attention.L).toBe(modelConfig.num_layers);
    expect(data.embeddings.tokEmb).toEqual([modelConfig.vocab_size, modelConfig.emb_dim]);
    expect(data.embeddings.posEmb).toEqual([modelConfig.context_len, modelConfig.emb_dim]);
    expect(data.embeddings.X0).toEqual([data.attention.T, modelConfig.emb_dim]);
  });

  withData("the parameter count matches the formula on the Architecture page", () => {
    const {
      vocab_size: V,
      context_len: Tmax,
      emb_dim: d,
      ff_dim: F,
      num_layers: L,
    } = data.meta.provenance.modelConfig as Record<string, number>;
    const formula =
      (2 * d! + 1) * V! + (Tmax! + 2) * d! + L! * (4 * d! * d! + 2 * d! * F! + F! + 9 * d!);
    expect(data.meta.paramCount).toBe(formula);
  });

  withData("tokens, ids and probabilities hold together", () => {
    for (const { tokens, ids, text, roundTrip } of data.tokenizer.sentences) {
      expect(tokens).toHaveLength(ids.length);
      expect(tokens.join("")).toBe(text);
      expect(roundTrip).toBe(true);
    }
    const { plain, sampling } = data.probabilities;
    expect(plain.probabilities).toHaveLength(data.meta.vocabSize);
    expect(plain.sum).toBeCloseTo(1, 9);
    expect(sampling.sum).toBeCloseTo(1, 9);
    for (const { top } of [plain, sampling]) {
      for (let i = 1; i < top.length; i++) {
        expect(top[i]!.probability).toBeLessThanOrEqual(top[i - 1]!.probability);
      }
    }
  });

  withData("the provenance line states commit, checkpoint hash, seed and D1's disclosure", () => {
    const source = data.meta.provenance;
    const line = provenanceLine(source);
    const weights = source.checkpointFiles.find(({ name }) => name.endsWith(".npz"))!;
    expect(line).toContain(source.gitShortSha);
    expect(line).toContain(weights.sha256.slice(0, 12));
    expect(line).toContain(`Seed ${source.seed}`);
    expect(line).toContain(source.disclosure);
    expect(source.disclosure).toMatch(/not at the pinned commit|not in the repository/);
  });

  withData("every Actual panel shows the provenance line and a label", () => {
    const line = provenanceLine(data.meta.provenance);
    const escaped = render(createElement("span", null, line));
    const inner = escaped.slice("<span>".length, -"</span>".length);
    for (const panel of [
      ActualTokenizerPanel,
      ActualEmbeddingsPanel,
      ActualGenerationPanel,
      ActualModelOutputSlot,
    ]) {
      const html = render(createElement(panel));
      expect(html, panel.name).toContain(inner);
      expect(html, panel.name).toContain("Actual Model Output");
      expect(html, panel.name).not.toContain("Illustrative Example");
    }
  });

  withData("the Tokenizer panel shows the real merges", () => {
    const html = render(createElement(ActualTokenizerPanel));
    for (const { rank, merged } of data.tokenizer.merges) {
      expect(html).toContain(`data-merge="${rank}"`);
      expect(html).toContain(`<code>${visibleToken(merged).replaceAll("'", "&#x27;")}</code>`);
    }
  });

  withData("the Embeddings panel shows the real shapes", () => {
    const html = render(createElement(ActualEmbeddingsPanel));
    expect(html).toContain(`(${data.embeddings.tokEmb.join(", ")})`);
    expect(html).toContain(`(${data.embeddings.X0.join(", ")})`);
  });

  withData("the Generation panel shows the real probabilities", () => {
    const html = render(createElement(ActualGenerationPanel));
    for (const { probability } of data.probabilities.plain.top) {
      expect(html).toContain(probability.toFixed(4));
    }
  });
});

describe("the export script", () => {
  const source = readFileSync("scripts/export_actual.py", "utf-8");

  test("defaults to the repository's prefix and trains only inside docs-site/.cache", () => {
    expect(source).toContain('DEFAULT_PREFIX = "checkpoints/model"');
    expect(source).toContain('CACHE_DIR = "docs-site/.cache"');
    expect(source).toMatch(/is_relative_to\(\(REPO \/ CACHE_DIR\)/);
  });

  test("names the real prefix default in train.py", () => {
    const signature = execFileSync("git", ["grep", "-n", "def main", "HEAD", "--", ":/train.py"], {
      encoding: "utf-8",
    });
    expect(signature).toContain('def main(prefix="checkpoints/model")');
  });

  test("imports training only for --train", () => {
    expect(source).not.toMatch(/^import train|^from train/m);
    expect(source).toContain("import train as repository_train");
  });
});

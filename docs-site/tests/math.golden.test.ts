import { describe, expect, test } from "vitest";
import { GOLDEN_FILES, loadGolden, type GoldenCase } from "@/lib/golden";
import {
  attentionForward,
  attentionScores,
  mergeHeads,
  scaleScores,
  splitHeads,
  type AttentionParams,
} from "@/lib/math/attention";
import {
  applyMerges,
  bpeSteps,
  buildTokenizer,
  countWords,
  decode,
  encode,
  mergeTokens,
  trainBpe,
  type Merge,
  type Pair,
  type WordFrequency,
} from "@/lib/math/bpe";
import {
  crossEntropy,
  crossEntropyBackward,
  layerNormBackward,
  layerNormForward,
  layerNormStats,
  linearBackward,
  linearForward,
  logSoftmax,
  perplexity,
  relu,
  reluBackward,
  softmax,
  softmaxBackward,
  softmaxBackwardRows,
  softmaxRows,
  xavierLimit,
} from "@/lib/math/ops";
import {
  numericalGrad,
  numericalGradVector as row,
  outputDot as dotted,
} from "@/lib/math/gradcheck";
import {
  adamStep,
  clipGradNorm,
  createAdamState,
  lrSchedule,
  type AdamSettings,
  type TensorMap,
} from "@/lib/math/optim";
import {
  batchesPerEpoch,
  batchSizes,
  joinRows,
  makeWindows,
  range,
  splitIds,
  windowCount,
  windowStarts,
} from "@/lib/math/data";
import { blockParameterCount, parameterCount, type ParameterDims } from "@/lib/math/params";
import { sampleNextDistribution, type SampleResult } from "@/lib/math/sampling";
import type { Matrix, Tensor3, Vector } from "@/lib/math/tensor";
import { shapeOf } from "@/lib/viz/shape";

const TOLERANCE = 1e-9;

// Same structure, and each number within TOLERANCE. NaN fails. -Infinity must match exactly.
function expectClose(actual: unknown, expected: unknown, path: string): void {
  if (typeof expected === "number") {
    expect(typeof actual, path).toBe("number");
    const value = actual as number;
    if (!Number.isFinite(expected)) {
      expect(value, path).toBe(expected);
      return;
    }
    expect(Math.abs(value - expected) <= TOLERANCE, `${path}: got ${value}, want ${expected}`).toBe(
      true,
    );
    return;
  }
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual), path).toBe(true);
    const list = actual as unknown[];
    expect(list.length, `${path} length`).toBe(expected.length);
    expected.forEach((item, i) => expectClose(list[i], item, `${path}[${i}]`));
    return;
  }
  if (typeof expected === "object" && expected !== null) {
    const record = actual as Record<string, unknown>;
    for (const [key, item] of Object.entries(expected)) {
      expectClose(record[key], item, `${path}.${key}`);
    }
    return;
  }
  expect(actual, path).toEqual(expected);
}

type Runner = (item: GoldenCase) => void;

const check = (item: GoldenCase, actual: Record<string, unknown>) => {
  expect(Object.keys(actual).sort(), item.id).toEqual(Object.keys(item.outputs).sort());
  expectClose(actual, item.outputs, item.id);
};

function checkSampling(item: GoldenCase): void {
  const inputs = item.inputs as {
    logits: Vector;
    temperature: number;
    top_k: number;
    top_p: number;
  };
  const outputs = item.outputs as {
    kind: string;
    index?: number;
    p?: Vector;
    after_temperature?: Vector;
    after_top_k?: Vector;
    order?: number[];
    cum?: Vector;
    keep_count?: number;
    tie_split?: boolean;
  };
  const settings = {
    temperature: inputs.temperature,
    topK: inputs.top_k,
    topP: inputs.top_p,
  };
  const narrow = (result: SampleResult) => {
    if (result.kind !== "distribution") throw new Error(`${item.id}: expected a distribution`);
    return result;
  };

  if (outputs.kind === "greedy") {
    const result = sampleNextDistribution(inputs.logits, settings);
    expect(result).toEqual({ kind: "greedy", index: outputs.index });
    return;
  }

  // Python's own sort order: every token must match.
  const exact = narrow(sampleNextDistribution(inputs.logits, settings, outputs.order));
  expectClose(exact.p, outputs.p, `${item.id}.p`);
  expectClose(exact.afterTemperature, outputs.after_temperature, `${item.id}.after_temperature`);
  expectClose(exact.afterTopK, outputs.after_top_k, `${item.id}.after_top_k`);

  // The engine's own stable order.
  const own = narrow(sampleNextDistribution(inputs.logits, settings));
  if (outputs.order === undefined) {
    expect(own.topP.applied, item.id).toBe(false);
    expectClose(own.p, outputs.p, `${item.id}.own.p`);
    return;
  }
  expect(own.topP.applied, item.id).toBe(true);
  expect(exact.topP.keepCount, item.id).toBe(outputs.keep_count);
  expect(own.topP.keepCount, item.id).toBe(outputs.keep_count);
  expectClose(exact.topP.cum, outputs.cum, `${item.id}.cum`);
  expectClose(own.topP.cum, outputs.cum, `${item.id}.own.cum`);
  expect(own.topP.tieDependent, item.id).toBe(outputs.tie_split);
  expect(exact.topP.tieDependent, item.id).toBe(outputs.tie_split);
  if (outputs.tie_split) {
    const sorted = (values: Vector) => [...values].sort((a, b) => a - b);
    expectClose(sorted(own.p), sorted(outputs.p!), `${item.id}.own.sorted`);
  } else {
    expectClose(own.p, outputs.p, `${item.id}.own.p`);
  }
}

const RUNNERS: Record<string, Runner> = {
  "ops.linear_forward": (item) => {
    const { X, W, b } = item.inputs as { X: Matrix; W: Matrix; b: Vector };
    check(item, { Y: linearForward(X, W, b) });
  },
  "ops.linear_backward": (item) => {
    const { X, W, dY } = item.inputs as { X: Matrix; W: Matrix; dY: Matrix };
    check(item, linearBackward(X, W, dY));
  },
  "ops.relu_forward": (item) => check(item, { Y: relu(item.inputs.X as Matrix) }),
  "ops.relu_backward": (item) => {
    check(item, { dZ: reluBackward(item.inputs.Z as Matrix, item.inputs.dA as Matrix) });
  },
  "ops.softmax_forward": (item) => {
    const { x } = item.inputs;
    const y = Array.isArray((x as unknown[])[0]) ? softmaxRows(x as Matrix) : softmax(x as Vector);
    check(item, { y });
  },
  "ops.softmax_backward": (item) => {
    const { y, dY } = item.inputs;
    const dx = Array.isArray((y as unknown[])[0])
      ? softmaxBackwardRows(y as Matrix, dY as Matrix)
      : softmaxBackward(y as Vector, dY as Vector);
    check(item, { dx });
  },
  "ops.cross_entropy_forward": (item) => {
    const logits = item.inputs.logits as Matrix;
    const steps = logits.map(logSoftmax);
    check(item, {
      ...crossEntropy(logits, item.inputs.targets as number[]),
      z: steps.map(({ z }) => z),
      logp: steps.map(({ logp }) => logp),
    });
  },
  "ops.cross_entropy_backward": (item) => {
    check(item, {
      dZ: crossEntropyBackward(item.inputs.probs as Matrix, item.inputs.targets as number[]),
    });
  },
  "ops.layer_norm_forward": (item) => {
    const { X, gamma, beta, epsilon } = item.inputs as {
      X: Matrix;
      gamma: Vector;
      beta: Vector;
      epsilon?: number;
    };
    const { Y, cache } = layerNormForward(X, gamma, beta, epsilon);
    const { mu, var: variance } = layerNormStats(X, epsilon);
    check(item, { Y, Xhat: cache.Xhat, s: cache.s, mu, var: variance });
  },
  "ops.layer_norm_backward": (item) => {
    const { dY, Xhat, s, gamma } = item.inputs as {
      dY: Matrix;
      Xhat: Matrix;
      s: Matrix;
      gamma: Vector;
    };
    check(item, layerNormBackward(dY, { Xhat, s }, gamma));
  },
  "ops.perplexity": (item) => {
    check(item, { perplexity: perplexity(item.inputs.loss as number) });
  },
  // The draws come from NumPy's generator and have no mirror. The limit and the bound are checked.
  "ops.xavier_uniform": (item) => {
    const { fan_in, fan_out } = item.inputs as { fan_in: number; fan_out: number };
    const { a, low, high, shape, within } = item.outputs as {
      a: number;
      low: number;
      high: number;
      shape: number[];
      within: boolean;
    };
    expectClose(xavierLimit(fan_in, fan_out), a, `${item.id}.a`);
    expect(shape).toEqual([fan_in, fan_out]);
    expect(within).toBe(true);
    expect(Math.max(Math.abs(low), Math.abs(high))).toBeLessThanOrEqual(a);
  },
  "transformer.init_params": (item) => {
    const rows = item.outputs.rows as {
      name: string;
      shape: number[];
      low: number;
      high: number;
    }[];
    expect(rows.length).toBeGreaterThan(0);
    for (const { name, shape, low, high } of rows) {
      if (shape.length === 2) {
        const limit = xavierLimit(shape[0]!, shape[1]!);
        expect(Math.max(Math.abs(low), Math.abs(high)), name).toBeLessThanOrEqual(limit);
        expect(high, name).toBeGreaterThan(low);
      } else {
        expect(low, name).toBe(high);
        expect(low, name).toBe(/_g$/.test(name) ? 1 : 0);
      }
    }
  },
  "gradcheck.numerical_grad": (item) => {
    const inputs = item.inputs as Record<string, unknown>;
    if (inputs.problem === "linear") {
      const { X, W, b, dY } = inputs as { X: Matrix; W: Matrix; b: Vector; dY: Matrix };
      check(item, {
        dX: numericalGrad((v) => dotted(linearForward(v, W, b), dY), X),
        dW: numericalGrad((v) => dotted(linearForward(X, v, b), dY), W),
        db: row((v) => dotted(linearForward(X, W, v), dY), b),
      });
    } else if (inputs.problem === "cross_entropy") {
      const { logits, targets } = inputs as { logits: Matrix; targets: number[] };
      check(item, { dZ: numericalGrad((v) => crossEntropy(v, targets).loss, logits) });
    } else if (inputs.problem === "layer_norm") {
      const { X, gamma, beta, dY } = inputs as {
        X: Matrix;
        gamma: Vector;
        beta: Vector;
        dY: Matrix;
      };
      check(item, {
        dX: numericalGrad((v) => dotted(layerNormForward(v, gamma, beta).Y, dY), X),
        dgamma: row((v) => dotted(layerNormForward(X, v, beta).Y, dY), gamma),
        dbeta: row((v) => dotted(layerNormForward(X, gamma, v).Y, dY), beta),
      });
    } else {
      const { x } = inputs as { x: Vector };
      const ones = x.map(() => 1);
      const zeros = x.map(() => 0);
      check(item, {
        J: x.map((_, i) => row((v) => layerNormForward([v], ones, zeros).cache.Xhat[0]![i]!, x)),
      });
    }
  },
  "transformer.split_heads": (item) => {
    check(item, { Mh: splitHeads(item.inputs.M as Matrix, item.inputs.H as number) });
  },
  "transformer.merge_heads": (item) => check(item, { M: mergeHeads(item.inputs.Mh as Tensor3) }),
  "transformer.attention_forward": (item) => {
    const { Xn, H, ...params } = item.inputs as { Xn: Matrix; H: number } & AttentionParams;
    const { raw, ...result } = attentionForward(Xn, params, H);
    check(item, result);
    // The separate steps compose to what Python computes on one line.
    const dh = (Xn[0]!.length / H) | 0;
    expectClose(
      scaleScores(attentionScores(result.Qh, result.Kh), dh),
      item.outputs.scaled,
      `${item.id}.steps`,
    );
    expect(raw.length).toBe(H);
  },
  "optim.lr_schedule": (item) => {
    const { steps, warmup, total, lr_max, lr_min } = item.inputs as {
      steps: number[];
      warmup: number;
      total: number;
      lr_max: number;
      lr_min: number;
    };
    check(item, { lrs: steps.map((step) => lrSchedule(step, warmup, total, lr_max, lr_min)) });
  },
  "optim.clip_grad_norm": (item) => {
    const before = JSON.stringify(item.inputs.grads);
    check(item, clipGradNorm(item.inputs.grads as TensorMap, item.inputs.max_norm as number));
    expect(JSON.stringify(item.inputs.grads), "inputs stay unchanged").toBe(before);
  },
  "optim.Adam.step": (item) => {
    const { params, grads, lrs, settings } = item.inputs as {
      params: TensorMap;
      grads: TensorMap[];
      lrs: number[];
      settings: { beta1?: number; beta2?: number; eps?: number; weight_decay?: number };
    };
    const adamSettings: AdamSettings = {
      ...(settings.beta1 === undefined ? {} : { beta1: settings.beta1 }),
      ...(settings.beta2 === undefined ? {} : { beta2: settings.beta2 }),
      ...(settings.eps === undefined ? {} : { eps: settings.eps }),
      ...(settings.weight_decay === undefined ? {} : { weightDecay: settings.weight_decay }),
    };
    let state = createAdamState(params, adamSettings);
    let current = params;
    const steps = grads.map((stepGrads, i) => {
      const next = adamStep(state, current, stepGrads, lrs[i]!);
      state = next.state;
      current = next.params;
      return { params: current, m: state.m, v: state.v, t: state.t };
    });
    check(item, { steps });
  },
  "generate.sample_next": checkSampling,
  "data.load_corpus": (item) => {
    const { rows, max_rows } = item.inputs as { rows: (string | null)[]; max_rows: number };
    check(item, { text: joinRows(rows, max_rows) });
  },
  "data.make_windows": (item) => {
    const { ids, seq_len, stride } = item.inputs as {
      ids: number[];
      seq_len: number;
      stride: number;
    };
    if (item.outputs.raises) {
      expect(() => makeWindows(ids, seq_len, stride)).toThrow(item.outputs.message as string);
      return;
    }
    const { starts, inputs, targets } = makeWindows(ids, seq_len, stride);
    check(item, {
      starts,
      inputs,
      targets,
      count: windowCount(ids.length, seq_len, stride),
      stored_as: item.outputs.stored_as,
    });
    expect(windowStarts(ids.length, seq_len, stride).length, item.id).toBe(item.outputs.count);
  },
  "data.split_ids": (item) => {
    const { n, val_fraction, seq_len } = item.inputs as {
      n: number;
      val_fraction: number;
      seq_len: number;
    };
    const ids = range(0, n);
    const { cut, train, val } = splitIds(ids, val_fraction);
    const whole = makeWindows(ids, seq_len);
    check(item, {
      cut,
      train_length: train.length,
      val_length: val.length,
      train_last: train.at(-1),
      val_first: val[0],
      train_windows: windowCount(train.length, seq_len),
      val_windows: windowCount(val.length, seq_len),
      whole_windows: whole.starts.length,
      crossing: whole.inputs.filter((row, w) => row[0]! < cut && whole.targets[w]!.at(-1)! >= cut)
        .length,
    });
  },
  // The order comes from pandas and NumPy's generator and has no mirror. The rest is checked.
  "data.iterate_batches": (item) => {
    const { n, seq_len, batch_size, shuffle } = item.inputs as {
      n: number;
      seq_len: number;
      batch_size: number;
      shuffle: boolean;
    };
    const count = windowCount(n, seq_len);
    const outputs = item.outputs as { epoch_starts: number[][][]; sizes: number[] };
    expect(outputs.sizes).toEqual(batchSizes(count, batch_size));
    expect(outputs.sizes).toHaveLength(batchesPerEpoch(count, batch_size));
    for (const epoch of outputs.epoch_starts) {
      expect(epoch.flat().sort((a, b) => a - b)).toEqual(range(0, count));
      if (!shuffle) expect(epoch.flat()).toEqual(range(0, count));
    }
  },
  "transformer.count_params": (item) => {
    const dims = item.inputs as ParameterDims;
    check(item, { count: parameterCount(dims), block: blockParameterCount(dims) });
  },
  "tokenizer.merge_tokens": (item) => {
    const { tokens, pair, merged } = item.inputs as {
      tokens: string[];
      pair: Pair;
      merged: string;
    };
    check(item, { tokens: mergeTokens(tokens, pair, merged) });
  },
  "tokenizer.train_bpe": (item) => {
    const { word_freqs, max_merges } = item.inputs as {
      word_freqs: WordFrequency[];
      max_merges?: number;
    };
    const { merges, pairCounts } = trainBpe(word_freqs, max_merges);
    check(item, { merges, pair_counts: pairCounts });
  },
  "tokenizer.count_words": (item) => {
    check(item, { word_freqs: countWords(item.inputs.text as string) });
  },
  // The stepper's own steps, against the real train_bpe on the same text.
  "tokenizer.train_bpe_on_text": (item) => {
    const { text, max_merges } = item.inputs as { text: string; max_merges?: number };
    const { wordFreqs, initial, steps, stopped } = bpeSteps(text, max_merges);
    check(item, {
      word_freqs: wordFreqs,
      merges: steps.map(({ best, merged }) => ({
        pair: best.pair,
        merged,
        frequency: best.count,
      })),
      // Python computes pair_freq once more on the pass that finds no pair, and stops.
      pair_counts: [...steps.map(({ counts }) => counts), ...(stopped === "no-pairs" ? [[]] : [])],
      splits: [initial, ...steps.map(({ after }) => after)],
    });
  },
  "tokenizer.apply_merges": (item) => {
    const { text, merges } = item.inputs as { text: string; merges: Merge[] };
    check(item, { tokens: applyMerges(text, merges) });
  },
  "tokenizer.build_tokenizer": (item) => {
    const { text, max_merges } = item.inputs as { text: string; max_merges?: number };
    const tok = buildTokenizer(text, max_merges);
    check(item, {
      merges: tok.merges,
      itos: tok.itos,
      stoi: [...tok.stoi.entries()],
      vocab_size: tok.vocabSize,
    });
  },
  "tokenizer.Tokenizer.encode": (item) => {
    const { train_text, text } = item.inputs as { train_text: string; text: string };
    const tok = buildTokenizer(train_text);
    const ids = encode(tok, text);
    check(item, { tokens: applyMerges(text, tok.merges), ids, decoded: decode(tok, ids) });
  },
  "tokenizer.Tokenizer.decode": (item) => {
    const { train_text, ids } = item.inputs as { train_text: string; ids: number[] };
    check(item, { text: decode(buildTokenizer(train_text), ids) });
  },
};

// The functions this task requires. Each must have golden cases, so that removing a fixture from
// make_golden.py cannot silently remove a check.
const REQUIRED = [
  "ops.linear_forward",
  "ops.linear_backward",
  "ops.relu_forward",
  "ops.relu_backward",
  "ops.softmax_forward",
  "ops.softmax_backward",
  "ops.cross_entropy_forward",
  "ops.cross_entropy_backward",
  "ops.layer_norm_forward",
  "ops.layer_norm_backward",
  "ops.perplexity",
  "ops.xavier_uniform",
  "transformer.init_params",
  "gradcheck.numerical_grad",
  "transformer.split_heads",
  "transformer.merge_heads",
  "transformer.attention_forward",
  "optim.lr_schedule",
  "optim.clip_grad_norm",
  "optim.Adam.step",
  "generate.sample_next",
  "data.load_corpus",
  "data.split_ids",
  "data.make_windows",
  "data.iterate_batches",
  "tokenizer.merge_tokens",
  "tokenizer.train_bpe",
  "tokenizer.count_words",
  "tokenizer.train_bpe_on_text",
  "tokenizer.apply_merges",
  "tokenizer.build_tokenizer",
  "tokenizer.Tokenizer.encode",
  "tokenizer.Tokenizer.decode",
  "transformer.count_params",
];

const allCases = GOLDEN_FILES.flatMap((file) => loadGolden(file).map((item) => ({ file, item })));

function compareShapes(shape: unknown, value: unknown, path: string): void {
  if (Array.isArray(shape) && (shape.length === 0 || typeof shape[0] === "number")) {
    expect(shapeOf(value), path).toEqual(shape);
    return;
  }
  if (Array.isArray(shape)) {
    shape.forEach((item, i) => {
      if (item !== null) compareShapes(item, (value as unknown[])[i], `${path}[${i}]`);
    });
    return;
  }
  for (const [key, item] of Object.entries(shape as Record<string, unknown>)) {
    compareShapes(item, (value as Record<string, unknown>)[key], `${path}.${key}`);
  }
}

describe("every required function has golden cases", () => {
  test.each(REQUIRED)("%s", (symbol) => {
    expect(allCases.filter(({ item }) => item.mirrors === symbol).length).toBeGreaterThan(0);
  });
});

describe.each(allCases.map(({ file, item }) => [`${file}/${item.id}`, item] as const))(
  "%s",
  (_name, item) => {
    test("matches the Python result to within 1e-9", () => {
      const runner = RUNNERS[item.mirrors];
      if (!runner) throw new Error(`No runner for ${item.mirrors}`);
      runner(item);
    });

    test("is labelled and its stored shapes are right", () => {
      expect(item.kind).toBe("illustrative");
      compareShapes(item.shapes.inputs, item.inputs, "inputs");
      compareShapes(item.shapes.outputs, item.outputs, "outputs");
    });
  },
);

describe("fixtures reach the behaviour they exist for", () => {
  const sampling = (id: string) => loadGolden("sampling").find((item) => item.id === id)!.outputs;

  test("a tie can be split by the cutoff, and NumPy's order is not stable there", () => {
    const split = sampling("top_p.tie_split_by_cutoff");
    expect(split.tie_split).toBe(true);
    expect(split.keep_count).toBe(3);
    expect(sampling("top_p.tie_fully_kept").tie_split).toBe(false);
    expect(sampling("top_p.tie_fully_dropped").tie_split).toBe(false);
  });

  test("the guard is strict: temperature 1e-6 is not greedy", () => {
    expect(sampling("greedy.at_guard_is_not_greedy").kind).toBe("distribution");
    expect(sampling("greedy.below_guard").kind).toBe("greedy");
  });

  test("an exact cumulative boundary stops at that token", () => {
    expect(sampling("top_p.exact_boundary").keep_count).toBe(2);
    expect(sampling("top_p.just_past_boundary").keep_count).toBe(3);
  });

  test("-inf survives the trip through JSON", () => {
    const masked = loadGolden("attention").find((item) => item.id === "attention_forward.t3_d4_h2")!
      .outputs.masked as Tensor3;
    expect(masked[0]![0]![1]).toBe(-Infinity);
  });
});

describe("errors where Python raises", () => {
  test("splitHeads when d is not divisible by H", () => {
    expect(() => splitHeads([[1, 2, 3]], 2)).toThrow("not divisible");
  });

  test("lrSchedule on a division by zero", () => {
    expect(() => lrSchedule(0, 0, 10, 1, 0)).toThrow("warmup");
    expect(() => lrSchedule(5, 2, 2, 1, 0)).toThrow("total");
  });
});

import { describe, expect, test } from "vitest";
import { GOLDEN_FILES, loadGolden, type GoldenCase } from "@/lib/golden";
import {
  attentionBackward,
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
  relativeError,
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
  batchLossAndGrads,
  LOOP_STEPS,
  loopStep,
  stepCounts,
  validationLoss,
} from "@/lib/math/train";
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
import {
  blockBackward,
  blockForward,
  ffnBackward,
  ffnForward,
  type FlatParams,
} from "@/lib/math/block";
import { embedBackward, embedForward, overwritingBackward } from "@/lib/math/embeddings";
import { forward, lossAndGrads, type ModelSizes } from "@/lib/math/model";
import { blockParameterCount, parameterCount, type ParameterDims } from "@/lib/math/params";
import {
  generationSteps,
  sampleNextDistribution,
  topKCutoff,
  type SampleResult,
} from "@/lib/math/sampling";
import { matmul, type Matrix, type Tensor3, type Vector } from "@/lib/math/tensor";
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

const matrixZeros = (rows: number, cols: number): Matrix =>
  Array.from({ length: rows }, () => new Array<number>(cols).fill(0));

// One block's parameters, all zeros, at real sizes. Only shapes are compared.
function blockZeros(d: number, F: number): FlatParams {
  const vector = (size: number) => new Array<number>(size).fill(0);
  const params: FlatParams = {
    "b0.ln1_g": vector(d),
    "b0.ln1_b": vector(d),
    "b0.ln2_g": vector(d),
    "b0.ln2_b": vector(d),
    "b0.ff1_w": matrixZeros(d, F),
    "b0.ff1_b": vector(F),
    "b0.ff2_w": matrixZeros(F, d),
    "b0.ff2_b": vector(d),
  };
  for (const n of "qkvo") {
    params[`b0.w${n}`] = matrixZeros(d, d);
    params[`b0.b${n}`] = vector(d);
  }
  return params;
}

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
    cutoff?: number;
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
  if (outputs.cutoff !== undefined) {
    expectClose(
      topKCutoff(exact.afterTemperature, inputs.top_k),
      outputs.cutoff,
      `${item.id}.cutoff`,
    );
  }

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

// The real generate.generate with forward replaced by a table. The picks of rng.choice have no mirror, so
// the stored draws stand in for them. Windows, logits rows, distributions and greedy picks are checked.
function checkGenerate(item: GoldenCase): void {
  const inputs = item.inputs as {
    table: Matrix;
    seed_ids: number[];
    max_new_tokens: number;
    window: number;
    temperature: number;
    top_k: number;
    top_p: number;
  };
  const outputs = item.outputs as {
    ids: number[];
    windows: number[][];
    logits_shapes: number[][];
    distributions: Matrix;
    draws: number[];
  };
  const steps = generationSteps({
    seedIds: inputs.seed_ids,
    maxNewTokens: inputs.max_new_tokens,
    window: inputs.window,
    settings: { temperature: inputs.temperature, topK: inputs.top_k, topP: inputs.top_p },
    logitsRows: (ids) => ids.map((id) => inputs.table[id]!),
    pick: (step) => outputs.draws[step]!,
  });
  expect(steps.map((step) => step.window)).toEqual(outputs.windows);
  expect(steps.map((step) => [step.window.length, inputs.table[0]!.length])).toEqual(
    outputs.logits_shapes,
  );
  expect([...steps.at(-1)!.ids, steps.at(-1)!.next]).toEqual(outputs.ids);
  const sampled = steps.flatMap((step) => (step.result.kind === "distribution" ? [step] : []));
  expect(sampled.length, item.id).toBe(outputs.distributions.length);
  sampled.forEach((step, i) => {
    if (step.result.kind !== "distribution") return;
    expectClose(step.result.p, outputs.distributions[i], `${item.id}.p[${i}]`);
    expect(step.result.p[step.next]!, `${item.id}: a removed token was drawn`).toBeGreaterThan(0);
  });
}

// Draws from the real generator. They have no mirror, but the engine's distribution must explain them.
function checkDraws(item: GoldenCase): void {
  const inputs = item.inputs as {
    logits: Vector;
    temperature: number;
    top_k: number;
    top_p: number;
  };
  const outputs = item.outputs as {
    first_draws: number[];
    counts: number[];
    total: number;
    same_seed_same_draws: boolean;
  };
  const result = sampleNextDistribution(inputs.logits, {
    temperature: inputs.temperature,
    topK: inputs.top_k,
    topP: inputs.top_p,
  });
  if (result.kind !== "distribution") throw new Error(`${item.id}: expected a distribution`);
  expect(outputs.same_seed_same_draws, item.id).toBe(true);
  expect(outputs.counts.reduce((sum, count) => sum + count, 0)).toBe(outputs.total);
  result.p.forEach((value, token) => {
    if (value === 0) expect(outputs.counts[token], `${item.id} token ${token}`).toBe(0);
    // Five standard deviations of a binomial count.
    const spread = 5 * Math.sqrt(outputs.total * value * (1 - value));
    expect(Math.abs(outputs.counts[token]! - outputs.total * value)).toBeLessThanOrEqual(spread);
  });
  outputs.first_draws.forEach((token) => expect(result.p[token]!).toBeGreaterThan(0));
}

// Central differences for named entries of a parameter dictionary.
function numericParams(params: FlatParams, names: string[], scalar: (next: FlatParams) => number) {
  return Object.fromEntries(
    names.map((name) => {
      const value = params[name]!;
      return [
        name,
        Array.isArray(value[0])
          ? numericalGrad((v) => scalar({ ...params, [name]: v }), value as Matrix)
          : row((v) => scalar({ ...params, [name]: v }), value as Vector),
      ];
    }),
  );
}

const RUNNERS: Record<string, Runner> = {
  "transformer.ffn_forward": (item) => {
    const { Xn, params } = item.inputs as { Xn: Matrix; params: FlatParams };
    const { out, cache } = ffnForward(Xn, params, "b0.");
    check(item, { out, Z: cache.Z, A: cache.A, cache_keys: Object.keys(cache) });
  },
  "transformer.ffn_backward": (item) => {
    const { Xn, params, dOut } = item.inputs as { Xn: Matrix; params: FlatParams; dOut: Matrix };
    check(item, ffnBackward(dOut, ffnForward(Xn, params, "b0.").cache, params, "b0."));
  },
  "transformer.block_forward": (item) => {
    if (item.inputs.problem === "config_shapes") {
      // The engine, run at the sizes in config.py, has the shapes of the real functions.
      const { seq_len, emb_dim, num_heads, ff_dim } = item.inputs as Record<string, number>;
      const T = seq_len!;
      const d = emb_dim!;
      const F = ff_dim!;
      const params = blockZeros(d, F);
      const zeros = matrixZeros(T, d);
      const block = blockForward(zeros, params, 0, num_heads!, 1e-5);
      const back = blockBackward(zeros, block.cache, params, 0, num_heads!);
      const ffn = ffnBackward(zeros, block.cache.ff, params, "b0.");
      const sized = (values: Record<string, unknown>) =>
        Object.fromEntries(Object.entries(values).map(([key, value]) => [key, shapeOf(value)]));
      const { Xn, att, X1, Xn2, ff, Y } = block;
      const { dXn2, dX1_branch, dX1, dXn, dX_branch, dX } = back;
      check(item, {
        forward: sized({
          X: zeros,
          Xn,
          att,
          X1,
          Xn2,
          ff,
          Y,
          Z: block.cache.ff.Z,
          A: block.cache.ff.A,
        }),
        backward: sized({
          dY: zeros,
          dXn2,
          dX1_branch,
          dX1,
          dXn,
          dX_branch,
          dX,
          dA: ffn.dA,
          dZ: ffn.dZ,
          dW2: ffn.grads["b0.ff2_w"],
          db2: ffn.grads["b0.ff2_b"],
          dW1: ffn.grads["b0.ff1_w"],
          db1: ffn.grads["b0.ff1_b"],
        }),
      });
      return;
    }
    const { X, params, l, cfg } = item.inputs as {
      X: Matrix;
      params: FlatParams;
      l: number;
      cfg: ModelSizes;
    };
    const { cache, ...locals } = blockForward(X, params, l, cfg.num_heads, cfg.ln_eps);
    check(item, {
      ...locals,
      cache_keys: {
        block: Object.keys(cache),
        att: Object.keys(cache.att),
        ff: Object.keys(cache.ff),
      },
    });
  },
  "transformer.block_backward": (item) => {
    const { X, dY, params, l, cfg } = item.inputs as {
      X: Matrix;
      dY: Matrix;
      params: FlatParams;
      l: number;
      cfg: ModelSizes;
    };
    const { cache } = blockForward(X, params, l, cfg.num_heads, cfg.ln_eps);
    check(item, blockBackward(dY, cache, params, l, cfg.num_heads));
  },
  "transformer.forward": (item) => {
    const { params, ids, cfg } = item.inputs as {
      params: FlatParams;
      ids: number[];
      cfg: ModelSizes;
    };
    const { logits, stages, cache } = forward(params, ids, cfg);
    check(item, {
      stages,
      Xf: cache.Xf,
      logits,
      cache_keys: Object.keys(cache),
      block_caches: cache.blocks.length,
    });
  },
  "transformer.loss_and_grads": (item) => {
    const inputs = item.inputs as Record<string, unknown>;
    if (inputs.problem === "order") {
      const { calls } = item.outputs as { calls: { name: string; depth: number; l?: number }[] };
      const names = calls
        .filter((call) => call.depth === 1)
        .map((call) => (call.l === undefined ? call.name : `${call.name}(${call.l})`));
      const params = (loadGolden("model").find((entry) => entry.id === "loss_and_grads.tiny")!
        .inputs.params ?? {}) as FlatParams;
      const { ids, targets, cfg } = inputs as {
        ids: number[];
        targets: number[];
        cfg: ModelSizes;
      };
      expect(lossAndGrads(params, ids, targets, cfg).calls).toEqual(names);
      return;
    }
    if (inputs.problem === "config_shapes") {
      const { params, grads } = item.outputs as Record<string, Record<string, number[]>>;
      expect(Object.keys(grads!)).toEqual(Object.keys(params!));
      for (const [name, shape] of Object.entries(params!))
        expect(grads![name], name).toEqual(shape);
      return;
    }
    const { params, ids, targets, cfg } = inputs as {
      params: FlatParams;
      ids: number[];
      targets: number[];
      cfg: ModelSizes;
    };
    const { calls, ...result } = lossAndGrads(params, ids, targets, cfg);
    expect(calls.length).toBeGreaterThan(0);
    check(item, result);
  },
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
  "transformer.embed_forward": (item) => {
    const inputs = item.inputs as Record<string, unknown>;
    if (inputs.problem === "config_sizes") {
      const { V, seq_len, context_len, emb_dim } = inputs as Record<string, number>;
      expect(item.outputs.tok_emb_shape).toEqual([V, emb_dim]);
      expect(item.outputs.pos_emb_shape).toEqual([context_len, emb_dim]);
      expect(item.outputs.X0_shape).toEqual([seq_len, emb_dim]);
      expect(item.outputs.at_limit_shape).toEqual([context_len, emb_dim]);
      expect(item.outputs.too_long_message).toBe("sequence longer than the position table");
      return;
    }
    const { tok_emb, pos_emb, ids } = inputs as { tok_emb: Matrix; pos_emb: Matrix; ids: number[] };
    check(item, embedForward(tok_emb, pos_emb, ids));
  },
  "transformer.embed_backward": (item) => {
    const inputs = item.inputs as Record<string, unknown>;
    if (inputs.problem === "position_update") {
      const { gradient_nonzero, row_changed } = item.outputs as {
        gradient_nonzero: boolean[];
        row_changed: boolean[];
      };
      const { T, Tmax, weight_decay } = inputs as Record<string, number>;
      const inWindow = Array.from({ length: Tmax! }, (_, t) => t < T!);
      expect(gradient_nonzero).toEqual(inWindow);
      expect(row_changed).toEqual(weight_decay! > 0 ? inWindow.map(() => true) : inWindow);
      return;
    }
    const { dX0, ids, V, Tmax } = inputs as { dX0: Matrix; ids: number[]; V: number; Tmax: number };
    check(item, {
      ...embedBackward(dX0, ids, V, Tmax),
      overwritten: overwritingBackward(dX0, ids, V),
    });
  },
  "gradcheck.relative_error": (item) => {
    const { a, b } = item.inputs as { a: number; b: number };
    check(item, { error: relativeError(a, b) });
  },
  // NumPy's generator picks the entries and has no mirror, so this checks the report's form.
  "gradcheck.check_params": (item) => {
    const { worst, names, loss_evaluations } = item.outputs as {
      worst: Record<string, number>;
      names: string[];
      loss_evaluations: number;
    };
    expect(Object.keys(worst)).toEqual(names);
    expect(loss_evaluations).toBe(names.length * 5 * 2);
    for (const name of names) expect(worst[name], name).toBeGreaterThanOrEqual(0);
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
    } else if (inputs.problem === "ffn") {
      const { Xn, params, dOut } = inputs as { Xn: Matrix; params: FlatParams; dOut: Matrix };
      const scalar = (next: FlatParams, x = Xn) => dotted(ffnForward(x, next, "b0.").out, dOut);
      check(item, {
        numeric: {
          ...numericParams(params, Object.keys(params), scalar),
          Xn: numericalGrad((x) => scalar(params, x), Xn),
        },
      });
    } else if (inputs.problem === "block") {
      const { X, dY, params, l, cfg } = inputs as {
        X: Matrix;
        dY: Matrix;
        params: FlatParams;
        l: number;
        cfg: ModelSizes;
      };
      const scalar = (next: FlatParams, x = X) =>
        dotted(blockForward(x, next, l, cfg.num_heads, cfg.ln_eps).Y, dY);
      const stored = item.outputs.numeric as Record<string, unknown>;
      const names = Object.keys(stored).filter((name) => name !== "X");
      check(item, {
        numeric: {
          ...numericParams(params, names, scalar),
          X: numericalGrad((x) => scalar(params, x), X),
        },
      });
    } else if (inputs.problem === "model") {
      const { params, ids, targets, cfg } = inputs as {
        params: FlatParams;
        ids: number[];
        targets: number[];
        cfg: ModelSizes;
      };
      const names = Object.keys(item.outputs.numeric as Record<string, unknown>);
      check(item, {
        numeric: numericParams(params, names, (next) => lossAndGrads(next, ids, targets, cfg).loss),
      });
    } else if (inputs.problem === "embedding") {
      const { tok_emb, pos_emb, ids, dX0 } = inputs as {
        tok_emb: Matrix;
        pos_emb: Matrix;
        ids: number[];
        dX0: Matrix;
      };
      check(item, {
        tok: numericalGrad((v) => dotted(embedForward(v, pos_emb, ids).X0, dX0), tok_emb),
        pos: numericalGrad((v) => dotted(embedForward(tok_emb, v, ids).X0, dX0), pos_emb),
      });
    } else if (inputs.problem === "quadratic") {
      const { x } = inputs as { x: Vector };
      check(item, { g: row((v) => v.reduce((total, value) => total + value * value, 0), x) });
    } else if (inputs.problem === "power") {
      const { power, x, h } = inputs as { power: 2 | 3; x: number; h: number };
      const f = (v: Vector) => (power === 2 ? v[0]! * v[0]! : v[0]! * v[0]! * v[0]!);
      const analytic = power === 2 ? 2 * x : 3 * x * x;
      const g = row(f, [x], h);
      check(item, { g, analytic, error: relativeError(analytic, g[0]!) });
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
  "transformer.attention_backward": (item) => {
    const { Xn, dOut, H, ...params } = item.inputs as {
      Xn: Matrix;
      dOut: Matrix;
      H: number;
    } & AttentionParams;
    check(item, attentionBackward(dOut, { Xn, ...attentionForward(Xn, params, H) }, params, H));
  },
  "transformer.attention_forward": (item) => {
    if (item.inputs.problem === "config_shapes") {
      // The engine, run on zeros at the sizes in config.py, has the shapes of the real functions.
      const { seq_len, emb_dim, num_heads } = item.inputs as Record<string, number>;
      const T = seq_len!;
      const d = emb_dim!;
      const H = num_heads!;
      const zeros = (rows: number, cols: number): Matrix =>
        Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
      const params: AttentionParams = {
        wq: zeros(d, d),
        bq: new Array<number>(d).fill(0),
        wk: zeros(d, d),
        bk: new Array<number>(d).fill(0),
        wv: zeros(d, d),
        bv: new Array<number>(d).fill(0),
        wo: zeros(d, d),
        bo: new Array<number>(d).fill(0),
      };
      const Xn = zeros(T, d);
      const forward = attentionForward(Xn, params, H);
      const backward = attentionBackward(zeros(T, d), { Xn, ...forward }, params, H);
      const sized = (values: Record<string, unknown>) =>
        Object.fromEntries(Object.entries(values).map(([key, value]) => [key, shapeOf(value)]));
      const { Q, K, V, Qh, Kh, Vh, scaled, A, O, out } = forward;
      const { grads, ...locals } = backward;
      check(item, {
        forward: sized({
          Xn,
          Q,
          K,
          V,
          Qh,
          Kh,
          Vh,
          S: scaled,
          A,
          AVh: A.map((head, h) => matmul(head, Vh[h]!)),
          O,
          out,
        }),
        backward: sized({ dOut: Xn, ...locals }),
        grads: sized(grads),
      });
      return;
    }
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
    if (item.inputs.problem === "decay_rule") {
      // Ones and zero gradients, at the sizes of the real init_params: only decay can change a value.
      const { sizes, lr } = item.inputs as unknown as {
        sizes: Record<string, number[]>;
        lr: number;
      };
      const fill = (value: number) =>
        Object.fromEntries(
          Object.entries(sizes).map(([name, shape]) => [
            name,
            shape.length === 1
              ? new Array<number>(shape[0]!).fill(value)
              : matrixZeros(shape[0]!, shape[1]!).map((row) => row.map(() => value)),
          ]),
        ) as TensorMap;
      const ones = fill(1);
      const next = adamStep(createAdamState(ones, adamSettings), ones, fill(0), lr);
      check(item, {
        rows: Object.keys(sizes).map((name) => ({
          name,
          ndim: sizes[name]!.length,
          changed: next.params[name]!.flat().some((value) => value !== 1),
        })),
      });
      return;
    }
    if (item.inputs.problem === "one_weight") {
      const inputs = item.inputs as unknown as {
        params: TensorMap;
        grads: TensorMap[];
        max_norm: number;
        warmup: number;
        total: number;
        lr_max: number;
        lr_min: number;
      };
      let state = createAdamState(inputs.params, adamSettings);
      let current = inputs.params;
      let step = 0;
      const rows = inputs.grads.map((stepGrads) => {
        const next = loopStep(state, current, stepGrads, step, {
          maxNorm: inputs.max_norm,
          warmup: inputs.warmup,
          total: inputs.total,
          lrMax: inputs.lr_max,
          lrMin: inputs.lr_min,
        });
        state = next.state;
        current = next.params;
        step = next.step;
        const trace = next.trace.w!;
        return {
          step,
          norm: next.norm,
          clipped: next.clipped.w,
          lr: next.lr,
          m: state.m.w,
          v: state.v.w,
          t: state.t,
          mhat: trace.mhat,
          vhat: trace.vhat,
          raw: trace.raw,
          update: trace.update,
          w: current.w,
        };
      });
      check(item, { rows });
      return;
    }
    let state = createAdamState(params, adamSettings);
    let current = params;
    const steps = grads.map((stepGrads, i) => {
      const next = adamStep(state, current, stepGrads, lrs[i]!);
      state = next.state;
      current = next.params;
      if (item.inputs.problem === "trace") {
        const pick = (field: "mhat" | "vhat" | "raw" | "update") =>
          Object.fromEntries(Object.entries(next.trace).map(([name, t]) => [name, t[field]]));
        return {
          params: current,
          m: state.m,
          v: state.v,
          t: state.t,
          mhat: pick("mhat"),
          vhat: pick("vhat"),
          raw: pick("raw"),
          update: pick("update"),
        };
      }
      return { params: current, m: state.m, v: state.v, t: state.t };
    });
    check(item, { steps });
  },
  "train.batch_loss_and_grads": (item) => {
    if (item.inputs.problem === "learning_step") {
      const inputs = item.inputs as unknown as {
        text: string;
        num_merges: number;
        seq_len: number;
        batch_size: number;
        val_fraction: number;
        stride: number;
        epochs: number;
        params: FlatParams;
        settings: { beta1: number; beta2: number; eps: number; weight_decay: number };
        max_norm: number;
        warmup: number;
        lr_max: number;
        lr_min: number;
        cfg: ModelSizes;
      };
      const outputs = item.outputs as { xb: number[][]; yb: number[][]; ids: number[] };
      const tok = buildTokenizer(inputs.text, inputs.num_merges);
      const ids = encode(tok, inputs.text);
      const { train } = splitIds(ids, inputs.val_fraction);
      const { val } = splitIds(ids, inputs.val_fraction);
      const windows = makeWindows(train, inputs.seq_len, inputs.stride);
      const { totalSteps, stepsPerEpoch } = stepCounts(
        windows.starts.length,
        inputs.batch_size,
        inputs.epochs,
      );
      const { loss, grads } = batchLossAndGrads(inputs.params, outputs.xb, outputs.yb, inputs.cfg);
      const state = createAdamState(inputs.params as TensorMap, {
        beta1: inputs.settings.beta1,
        beta2: inputs.settings.beta2,
        eps: inputs.settings.eps,
        weightDecay: inputs.settings.weight_decay,
      });
      const next = loopStep(state, inputs.params as TensorMap, grads as TensorMap, 0, {
        maxNorm: inputs.max_norm,
        warmup: inputs.warmup,
        total: totalSteps,
        lrMax: inputs.lr_max,
        lrMin: inputs.lr_min,
      });
      const valWindows = makeWindows(val, inputs.seq_len, inputs.stride);
      const { logits } = forward(inputs.params, outputs.xb[0]!, inputs.cfg);
      check(item, {
        tokens: applyMerges(inputs.text, tok.merges),
        ids,
        vocab_size: tok.vocabSize,
        train_windows: windows.starts.length,
        val_windows: valWindows.starts.length,
        xb: outputs.xb,
        yb: outputs.yb,
        logits,
        probs: crossEntropy(logits, outputs.yb[0]!).probs,
        loss,
        grads,
        norm: next.norm,
        clipped: next.clipped,
        steps_per_epoch: stepsPerEpoch,
        total_steps: totalSteps,
        lr: next.lr,
        params: next.params,
        val_loss: validationLoss(
          next.params as FlatParams,
          inputs.cfg,
          valWindows.inputs,
          valWindows.targets,
        ).loss,
        seed_ids: valWindows.inputs[0]!.slice(0, 3),
        printed: item.outputs.printed,
      });
      return;
    }
    const { params, xb, yb, cfg } = item.inputs as unknown as {
      params: FlatParams;
      xb: number[][];
      yb: number[][];
      cfg: ModelSizes;
    };
    const { loss, grads, losses } = batchLossAndGrads(params, xb, yb, cfg);
    check(item, {
      loss,
      grads,
      losses,
      head_b_each: xb.map((x, i) => lossAndGrads(params, x, yb[i]!, cfg).grads.head_b),
    });
  },
  "train.validation_loss": (item) => {
    const { params, inputs, targets, cfg } = item.inputs as unknown as {
      params: FlatParams;
      inputs: number[][];
      targets: number[][];
      cfg: ModelSizes;
    };
    const { loss, losses } = validationLoss(params, cfg, inputs, targets);
    check(item, { loss, losses, perplexity: perplexity(loss) });
  },
  "train.main": (item) => {
    if (item.inputs.problem === "step_counts") {
      const rows = item.outputs.rows as {
        windows: number;
        batch_size: number;
        epochs: number;
        yielded: number;
      }[];
      check(item, {
        rows: rows.map((row) => {
          const { stepsPerEpoch, totalSteps, lastBatch } = stepCounts(
            row.windows,
            row.batch_size,
            row.epochs,
          );
          expect(batchSizes(row.windows, row.batch_size)).toHaveLength(row.yielded);
          return {
            windows: row.windows,
            batch_size: row.batch_size,
            epochs: row.epochs,
            steps_per_epoch: stepsPerEpoch,
            total_steps: totalSteps,
            yielded: batchSizes(row.windows, row.batch_size).length,
            last_size: lastBatch,
          };
        }),
      });
      return;
    }
    // The statements of the real batch loop, in order, with the engine's list after the gradient.
    const statements = item.outputs.statements as string[];
    const called = statements.flatMap((text) => {
      const match = LOOP_STEPS.find((name) => text.includes(name));
      return match ? [match] : [];
    });
    expect(called).toEqual([...LOOP_STEPS]);
    expect(statements[0]).toContain("batch_loss_and_grads");
  },
  "generate.sample_next": checkSampling,
  "generate.sample_next.draws": checkDraws,
  "generate.generate": checkGenerate,
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
  // The engine has no checkpoint code, so it is compared on what the tokenizer file holds.
  "checkpoint.save_checkpoint": (item) => {
    const { text, max_merges } = item.inputs as { text: string; max_merges: number };
    const { merges, itos } = buildTokenizer(text, max_merges);
    expect(JSON.parse(item.outputs.tokenizer_json as string)).toEqual({ merges, itos });
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
  "gradcheck.relative_error",
  "gradcheck.check_params",
  "transformer.split_heads",
  "transformer.merge_heads",
  "transformer.attention_forward",
  "transformer.attention_backward",
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
  "transformer.embed_forward",
  "transformer.embed_backward",
  "transformer.ffn_forward",
  "transformer.ffn_backward",
  "transformer.block_forward",
  "transformer.block_backward",
  "transformer.forward",
  "transformer.loss_and_grads",
  "train.batch_loss_and_grads",
  "train.validation_loss",
  "train.main",
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

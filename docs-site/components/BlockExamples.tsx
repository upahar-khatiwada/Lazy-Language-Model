import type { ReactNode } from "react";
import { CODE_REFS } from "@/data/code-refs";
import { BLOCK_CACHES, MODEL_CACHES, type CacheRow } from "@/data/caches";
import { BLOCK_FLOW, MODEL_FLOW, type FlowStep } from "@/data/gradient-flow";
import { formatLineRange, getSymbol, requireTarget, targetLines } from "@/lib/code-targets";
import { exampleAnchor } from "@/lib/concepts";
import { formatShape, resolveDimension } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { highlightPython } from "@/lib/highlight";
import {
  blockBackward,
  blockForward,
  ffnBackward,
  ffnForward,
  type FlatParams,
} from "@/lib/math/block";
import type { Matrix, Vector } from "@/lib/math/tensor";
import { forward, lossAndGrads, type ModelSizes } from "@/lib/math/model";
import { numericalGrad, numericalGradVector, outputDot } from "@/lib/math/gradcheck";
import { expandBlockName, loadInitParams } from "@/lib/parameters";
import { STREAM_STEPS } from "@/lib/residual-stream";
import { shapeOf } from "@/lib/viz/shape";
import { DataTable } from "./DataTable";
import examples from "./Examples.module.css";
import { GradientFlow, type FlowCode, type ResolvedFlowStep } from "./GradientFlow";
import labStyles from "./AttentionLab.module.css";
import { GradientCheck, Value, WorkedTable } from "./OpsExamples";
import { ResidualStream, type StreamLine } from "./ResidualStream";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases, and results come from lib/math, which
// tests/math.golden.test.ts compares with the real functions at 1e-9.

const TINY_CAPTION =
  "A tiny model: the parameters are random draws from the real init_params, not learned values. The other inputs are hand-picked.";

function Matrices({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className={labStyles.scroll} tabIndex={0} role="group" aria-label={label}>
      <div className={examples.matrices}>{children}</div>
    </div>
  );
}

function Anchored({ concept, children }: { concept: string; children: ReactNode }) {
  return (
    <div className={examples.stack} id={exampleAnchor(concept)}>
      {children}
    </div>
  );
}

const row = (value: Vector): Matrix => [value];

// Feed-forward ------------------------------------------------------------------------------------

const ffnInputs = () =>
  getGoldenCase("block", "ffn_backward.test_file").inputs as {
    Xn: Matrix;
    params: FlatParams;
    dOut: Matrix;
  };

export function FfnForwardExample({ concept }: { concept: string }) {
  const { Xn, params } = ffnInputs();
  const { out, cache } = ffnForward(Xn, params, "b0.");
  return (
    <Anchored concept={concept}>
      <VizFigure
        title="One row through the feed-forward layer"
        kind="illustrative"
        caption="The numbers of test_ffn_worked_example: two features widen to three and narrow to two. The model's sizes are in the next step."
      >
        <WorkedTable
          label="Feed-forward steps"
          testId="ffn-steps"
          rows={[
            { label: "Input", code: "Xn", value: Xn, testId: "ffn-xn" },
            {
              label: "Widen",
              code: 'Z = Xn @ params[p + "ff1_w"] + params[p + "ff1_b"]',
              value: cache.Z,
              testId: "ffn-z",
            },
            { label: "ReLU", code: "A = relu_forward(Z)", value: cache.A, testId: "ffn-a" },
            {
              label: "Narrow",
              code: 'A @ params[p + "ff2_w"] + params[p + "ff2_b"]',
              value: out,
              testId: "ffn-out",
            },
          ]}
        />
      </VizFigure>
    </Anchored>
  );
}

export function FfnBackwardExample({ concept }: { concept: string }) {
  const { Xn, params, dOut } = ffnInputs();
  const p = "b0.";
  const { cache } = ffnForward(Xn, params, p);
  const back = ffnBackward(dOut, cache, params, p);
  const scalar = (next: FlatParams) => outputDot(ffnForward(Xn, next, p).out, dOut);
  const matrix = (name: string): Matrix => params[`${p}${name}`] as Matrix;
  const vector = (name: string): Vector => params[`${p}${name}`] as Vector;
  return (
    <Anchored concept={concept}>
      <VizFigure
        title="The same row, backward"
        kind="illustrative"
        caption={
          <>
            The same forward pass as above. <code>dOut</code> is hand-picked, not computed from a
            loss. Two entries of <code>Z</code> are negative, so <code>dZ</code> is 0 there.
          </>
        }
      >
        <WorkedTable
          label="Feed-forward backward steps"
          testId="ffn-back-steps"
          rows={[
            { label: "Arrives", code: "dOut", value: dOut, testId: "ffn-back-dout" },
            {
              label: "Through the second layer",
              code: "dA",
              value: back.dA,
              testId: "ffn-back-da",
            },
            { label: "Through ReLU", code: "dZ", value: back.dZ, testId: "ffn-back-dz" },
            {
              label: "Through the first layer",
              code: "dXn",
              value: back.dXn,
              testId: "ffn-back-dxn",
            },
          ]}
        />
      </VizFigure>
      <GradientCheck
        title="Checking the backward numerically"
        testId="ffn-check"
        scalar={
          <>
            <code>(out * dOut).sum()</code>, whose gradient on <code>out</code> is exactly{" "}
            <code>dOut</code>,
          </>
        }
        rows={[
          {
            name: 'grads[p + "ff2_w"]',
            analytic: back.grads[`${p}ff2_w`] as Matrix,
            numeric: numericalGrad((W) => scalar({ ...params, [`${p}ff2_w`]: W }), matrix("ff2_w")),
          },
          {
            name: 'grads[p + "ff2_b"]',
            analytic: row(back.grads[`${p}ff2_b`] as Vector),
            numeric: row(
              numericalGradVector((b) => scalar({ ...params, [`${p}ff2_b`]: b }), vector("ff2_b")),
            ),
          },
          {
            name: 'grads[p + "ff1_w"]',
            analytic: back.grads[`${p}ff1_w`] as Matrix,
            numeric: numericalGrad((W) => scalar({ ...params, [`${p}ff1_w`]: W }), matrix("ff1_w")),
          },
          {
            name: 'grads[p + "ff1_b"]',
            analytic: row(back.grads[`${p}ff1_b`] as Vector),
            numeric: row(
              numericalGradVector((b) => scalar({ ...params, [`${p}ff1_b`]: b }), vector("ff1_b")),
            ),
          },
          {
            name: "dXn",
            analytic: back.dXn,
            numeric: numericalGrad((x) => outputDot(ffnForward(x, params, p).out, dOut), Xn),
          },
        ]}
      />
    </Anchored>
  );
}

// Block -------------------------------------------------------------------------------------------

type BlockInputs = {
  X: Matrix;
  params: FlatParams;
  l: number;
  cfg: ModelSizes;
};

const blockInputs = () => getGoldenCase("block", "block_forward.tiny").inputs as BlockInputs;

function runBlock() {
  const { X, params, l, cfg } = blockInputs();
  return { X, ...blockForward(X, params, l, cfg.num_heads, cfg.ln_eps) };
}

export function ResidualStreamExample({ concept }: { concept: string }) {
  const { X, Xn, att, X1, Xn2, ff, Y } = runBlock();
  const lines = Object.fromEntries(
    STREAM_STEPS.map(({ ref }) => {
      const target = CODE_REFS[ref];
      if (!target) throw new Error(`Unknown code reference: ${ref}`);
      const resolved = requireTarget(target);
      const line: StreamLine = {
        label: `${resolved.symbol.file}:${formatLineRange(resolved.startLine, resolved.endLine)}`,
        href: resolved.permalink,
        text: targetLines(resolved)
          .replace(/\s+#.*$/, "")
          .trim(),
      };
      return [ref, line];
    }),
  );
  return (
    <Anchored concept={concept}>
      <VizFigure title="The residual stream" kind="illustrative" caption={TINY_CAPTION}>
        <ResidualStream run={{ X, Xn, att, X1, Xn2, ff, Y }} lines={lines} defaultRow={1} />
      </VizFigure>
    </Anchored>
  );
}

export function BlockBackwardExample({ concept }: { concept: string }) {
  const { X, dY, l, cfg, params } = getGoldenCase("block", "block_backward.tiny").inputs as {
    X: Matrix;
    dY: Matrix;
    l: number;
    cfg: ModelSizes;
    params: FlatParams;
  };
  const { H, eps } = { H: cfg.num_heads, eps: cfg.ln_eps };
  const { cache } = blockForward(X, params, l, H, eps);
  const back = blockBackward(dY, cache, params, l, H);
  const off = getGoldenCase("block", "block_backward.branches_off").inputs.params as FlatParams;
  const noBranches = blockBackward(dY, blockForward(X, off, l, H, eps).cache, off, l, H);
  const scalar = (next: FlatParams) => outputDot(blockForward(X, next, l, H, eps).Y, dY);
  const check = (name: string) => {
    const value = params[name]!;
    const isMatrix = Array.isArray(value[0]);
    const analytic = back.grads[name]!;
    return {
      name: `grads["${name}"]`,
      analytic: (isMatrix ? analytic : row(analytic as Vector)) as Matrix,
      numeric: isMatrix
        ? numericalGrad((W) => scalar({ ...params, [name]: W }), value as Matrix)
        : row(numericalGradVector((v) => scalar({ ...params, [name]: v }), value as Vector)),
    };
  };
  return (
    <Anchored concept={concept}>
      <VizFigure
        title="One backward pass through a block"
        kind="illustrative"
        caption={`${TINY_CAPTION} dY is hand-picked, not computed from a loss.`}
      >
        <Matrices label="Backward through the feed-forward branch">
          <ResultMatrix label="dY" testId="block-dy" values={dY} />
          <ResultMatrix label="dXn2" testId="block-dxn2" values={back.dXn2} />
          <ResultMatrix label="dX1_branch" testId="block-dx1-branch" values={back.dX1_branch} />
          <ResultMatrix label="dX1 = dY + dX1_branch" testId="block-dx1" values={back.dX1} />
        </Matrices>
        <Matrices label="Backward through the attention branch">
          <ResultMatrix label="dXn" testId="block-dxn" values={back.dXn} />
          <ResultMatrix label="dX_branch" testId="block-dx-branch" values={back.dX_branch} />
          <ResultMatrix label="dX = dX1 + dX_branch" testId="block-dx" values={back.dX} />
        </Matrices>
      </VizFigure>

      <VizFigure
        title="Only the identity path"
        kind="illustrative"
        caption={
          <>
            The same block with <code>ln1_g</code> and <code>ln2_g</code> set to 0, so each branch
            sends back exactly 0. What leaves the block equals <code>dY</code>. This is an
            experiment for this page: <code>init_params</code> starts both at ones.
          </>
        }
      >
        <Matrices label="The block with both branches at zero">
          <ResultMatrix
            label="dX1_branch"
            testId="off-dx1-branch"
            values={noBranches.dX1_branch.map((values) => values.map((value) => value + 0))}
          />
          <ResultMatrix
            label="dX_branch"
            testId="off-dx-branch"
            values={noBranches.dX_branch.map((values) => values.map((value) => value + 0))}
          />
          <ResultMatrix label="dX" testId="off-dx" values={noBranches.dX} />
        </Matrices>
      </VizFigure>

      <GradientCheck
        title="Checking the backward numerically"
        testId="block-check"
        scalar={
          <>
            <code>(Y * dY).sum()</code>, whose gradient on <code>Y</code> is exactly <code>dY</code>
            ,
          </>
        }
        rows={[
          ...["b0.ln1_g", "b0.wq", "b0.wo", "b0.ln2_b", "b0.ff1_w", "b0.ff2_w"].map(check),
          {
            name: "dX",
            analytic: back.dX,
            numeric: numericalGrad((x) => outputDot(blockForward(x, params, l, H, eps).Y, dY), X),
          },
        ]}
      />
    </Anchored>
  );
}

// Caches ------------------------------------------------------------------------------------------

function Line({ target }: { target: CacheRow["built"] }) {
  const resolved = requireTarget(target);
  return (
    <a className="code-link" href={resolved.permalink}>
      {resolved.symbol.file}:{formatLineRange(resolved.startLine, resolved.endLine)}
    </a>
  );
}

export function CacheTable({ rows }: { rows: "block" | "model" }) {
  const list = rows === "block" ? BLOCK_CACHES : MODEL_CACHES;
  return (
    <DataTable
      label={rows === "block" ? "The four caches of a block" : "The cache of the whole model"}
      data-testid={`cache-table-${rows}`}
    >
      <thead>
        <tr>
          <th scope="col">Passed as</th>
          <th scope="col">Holds</th>
          <th scope="col">Built at</th>
          <th scope="col">Read by</th>
        </tr>
      </thead>
      <tbody>
        {list.map((cache) => (
          <tr key={cache.name} data-cache={cache.name}>
            <td>
              <code>{cache.name}</code>
            </td>
            <td>
              {cache.holds.map(({ field, meaning }) => (
                <span key={field} style={{ display: "block" }} data-field={field}>
                  <code>{field}</code>: {meaning}
                </span>
              ))}
            </td>
            <td>
              <Line target={cache.built} />
            </td>
            <td>
              <code>{cache.reader.split(".").at(-1)}</code>, at <Line target={cache.read} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

// Gradient flow -----------------------------------------------------------------------------------

function shapeText(step: FlowStep) {
  if (!step.dims) return { shape: null, values: null };
  return {
    shape: formatShape(step.dims),
    values: formatShape(step.dims.map((name) => resolveDimension(name).value ?? name)),
  };
}

function resolveStep(step: FlowStep, l?: number): ResolvedFlowStep {
  const target = requireTarget(step.target);
  const writeLines = (step.writeTargets ?? [])
    .map((write) => requireTarget(write))
    .filter(({ symbol }) => symbol.id === target.symbol.id);
  const lines = [target, ...writeLines].flatMap(({ startLine, endLine }) =>
    Array.from({ length: endLine - startLine + 1 }, (_, i) => startLine + i),
  );
  const perBlock = loadInitParams().parameters.filter((parameter) => parameter.perBlock);
  const writes = step.writesBlock
    ? `all ${perBlock.length} entries named b${l}.*`
    : (step.writes ?? []).join(", ");
  return {
    id: l === undefined ? step.id : `${step.id}-${l}`,
    output: step.output,
    label: l === undefined ? step.label : `${step.label} ${l}`,
    ...shapeText(step),
    statement: targetLines(target).trim(),
    location: `${target.symbol.file}:${formatLineRange(target.startLine, target.endLine)}`,
    href: target.permalink,
    isCall: Boolean(step.call),
    writes,
    note: step.note ?? null,
    caller: target.symbol.id,
    callee: step.call ?? null,
    lines,
  };
}

async function frame(id: string): Promise<FlowCode> {
  const symbol = getSymbol(id);
  return {
    lines: await highlightPython(symbol.source),
    source: symbol.source,
    filePath: symbol.file,
    functionName: symbol.name,
    startLine: symbol.startLine,
    permalink: symbol.permalink,
  };
}

// Server wrapper: resolves lines and highlights on the server, so Shiki stays out of the browser.
export async function GradientFlowDiagram({ flow }: { flow: "model" | "block" }) {
  const source = flow === "model" ? MODEL_FLOW : BLOCK_FLOW;
  const L = resolveDimension("L").value!;
  const steps = source.flatMap((step) =>
    step.perBlock
      ? Array.from({ length: L }, (_, i) => resolveStep(step, L - 1 - i))
      : [resolveStep(step)],
  );
  const ids = [...new Set(steps.flatMap(({ caller, callee }) => [caller, callee ?? caller]))];
  const code = Object.fromEntries(await Promise.all(ids.map(async (id) => [id, await frame(id)])));
  return (
    <GradientFlow
      title={
        flow === "model"
          ? "Gradient flow through the whole model, in the order of loss_and_grads"
          : "Gradient flow through one block, in the order of block_backward"
      }
      steps={steps}
      code={code}
      {...(flow === "block"
        ? {
            firstOutput: "dY",
            firstNote: "arrives from the next block, or from the final LayerNorm.",
          }
        : {})}
    />
  );
}

// Whole model -------------------------------------------------------------------------------------

type ModelInputs = { params: FlatParams; ids: number[]; targets: number[]; cfg: ModelSizes };

const modelInputs = () => getGoldenCase("model", "loss_and_grads.tiny").inputs as ModelInputs;

const SIZES_NOTE = (cfg: ModelSizes, params: FlatParams) => {
  const [V, d] = shapeOf(params.tok_emb);
  return `V = ${V}, T_max = ${cfg.context_len}, d = ${d}, H = ${cfg.num_heads}, L = ${cfg.num_layers}`;
};

export function ForwardExample({ concept }: { concept: string }) {
  const { params, ids, cfg } = modelInputs();
  const { stages, cache, logits } = forward(params, ids, cfg);
  return (
    <Anchored concept={concept}>
      <VizFigure
        title="The whole forward pass"
        kind="illustrative"
        caption={`${TINY_CAPTION} Sizes here: ${SIZES_NOTE(cfg, params)}. Logits from unlearned parameters mean nothing.`}
      >
        <p>
          <code>ids</code> = <Value value={ids} testId="model-ids" plain />
        </p>
        <Matrices label="The forward stages">
          <ResultMatrix label="X after embed_forward" testId="model-x0" values={stages[0]!} />
          {stages.slice(1).map((stage, i) => (
            <ResultMatrix
              key={i}
              label={`X after block ${i}`}
              testId={`model-x-block-${i}`}
              values={stage}
            />
          ))}
          <ResultMatrix label="Xf" testId="model-xf" values={cache.Xf} />
          <ResultMatrix label="logits" testId="model-logits" values={logits} />
        </Matrices>
      </VizFigure>
    </Anchored>
  );
}

export function LossAndGradsExample({ concept }: { concept: string }) {
  const { params, ids, targets, cfg } = modelInputs();
  const result = lossAndGrads(params, ids, targets, cfg);
  const scalar = (next: FlatParams) => lossAndGrads(next, ids, targets, cfg).loss;
  const check = (name: string) => {
    const value = params[name]!;
    const isMatrix = Array.isArray(value[0]);
    const analytic = result.grads[name]!;
    return {
      name: `grads["${name}"]`,
      analytic: (isMatrix ? analytic : row(analytic as Vector)) as Matrix,
      numeric: isMatrix
        ? numericalGrad((W) => scalar({ ...params, [name]: W }), value as Matrix)
        : row(numericalGradVector((v) => scalar({ ...params, [name]: v }), value as Vector)),
    };
  };
  const largest = (value: Matrix | Vector) =>
    Math.max(...(value as (number | number[])[]).flat().map(Math.abs));
  const names = Object.keys(params);
  return (
    <Anchored concept={concept}>
      <VizFigure
        title="The gradient on its way back"
        kind="illustrative"
        caption={`${TINY_CAPTION} ids = [${ids.join(", ")}], targets = [${targets.join(", ")}].`}
      >
        <p data-testid="model-loss" data-exact={result.loss}>
          <code>loss = {showNumber(result.loss)}</code>
        </p>
        <Matrices label="Gradients from the logits back">
          <ResultMatrix label="dZ" testId="model-dz" values={result.dZ} />
          <ResultMatrix label="dXf" testId="model-dxf" values={result.dXf} />
          {result.dX.map((value, i) => (
            <ResultMatrix
              key={i}
              label={
                i === 0 ? "dX after the final LayerNorm" : `dX after block ${cfg.num_layers - i}`
              }
              testId={`model-dx-${i}`}
              values={value}
            />
          ))}
        </Matrices>
      </VizFigure>

      <DataTable label="Every entry of grads after the run" data-testid="model-grads">
        <thead>
          <tr>
            <th scope="col">Key</th>
            <th scope="col">Shape</th>
            <th scope="col">Largest entry</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name} data-key={name}>
              <td>
                <code>{name}</code>
              </td>
              <td>
                <code>{formatShape(shapeOf(result.grads[name]))}</code>
              </td>
              <td data-exact={largest(result.grads[name]!)}>
                <code className={examples.nowrap}>
                  {largest(result.grads[name]!).toExponential(2)}
                </code>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>

      <GradientCheck
        title="Checking the backward numerically"
        testId="model-check"
        scalar={<>the loss</>}
        rows={[
          "tok_emb",
          "pos_emb",
          "b0.wk",
          "b0.ff1_w",
          "b1.ln2_g",
          "b1.wq",
          "lnf_g",
          "head_w",
          "head_b",
        ].map(check)}
      />
    </Anchored>
  );
}

// Parameter keys ----------------------------------------------------------------------------------

export function ParameterKeys() {
  const { parameters } = loadInitParams();
  const L = resolveDimension("L").value!;
  const d = resolveDimension("d").value!;
  const F = resolveDimension("F").value!;
  const Tmax = resolveDimension("Tmax").value!;
  const size = { d, F, Tmax, V: "V" } as const;
  const keys = expandBlockName(parameters, L);
  return (
    <details data-testid="parameter-keys">
      <summary>
        Every key as it appears in the dictionary: {keys.length} entries at{" "}
        <code>num_layers = {L}</code>
      </summary>
      <DataTable label="Every key of the parameter dictionary, in insertion order">
        <thead>
          <tr>
            <th scope="col">Key</th>
            <th scope="col">Shape with config</th>
          </tr>
        </thead>
        <tbody>
          {keys.map(({ key, dims }) => (
            <tr key={key} data-key={key}>
              <td>
                <code>{key}</code>
              </td>
              <td>
                <code>{formatShape(dims.map((dim) => size[dim]))}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </details>
  );
}

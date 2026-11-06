import type { ReactNode } from "react";
import { loadActual, provenanceLine } from "@/lib/actual";
import { LAB_STEPS, LAB_TOKENS, type LabInputs, type LabStepId } from "@/lib/attention-lab";
import { getSymbol } from "@/lib/code-targets";
import { exampleAnchor } from "@/lib/concepts";
import { resolveDimension } from "@/lib/dimensions";
import { getGoldenCase } from "@/lib/golden";
import { highlightPython } from "@/lib/highlight";
import { highlightedLines, lineEquations } from "@/lib/links";
import { attentionBackward, attentionForward, type AttentionParams } from "@/lib/math/attention";
import { numericalGrad, outputDot } from "@/lib/math/gradcheck";
import { matmul, type Matrix } from "@/lib/math/tensor";
import { ActualAttentionSummary } from "./ActualPanels";
import { AttentionLab } from "./AttentionLab";
import styles from "./AttentionLab.module.css";
import { Badge } from "./Badge";
import { DataTable } from "./DataTable";
import examples from "./Examples.module.css";
import { Heatmap } from "./Heatmap";
import { GradientCheck } from "./OpsExamples";
import { ResultMatrix, showNumber } from "./ResultMatrix";
import { ShapeFlow } from "./ShapeFlow";
import { VizFigure } from "./VizFigure";

// One run through the whole page. Inputs are the inputs of golden cases, and results come from
// lib/math, which tests/math.golden.test.ts compares with the real functions at 1e-9.

const forwardInputs = () =>
  getGoldenCase("attention", "attention_forward.cat_sat").inputs as LabInputs;

const backwardInputs = () =>
  getGoldenCase("attention", "attention_backward.cat_sat").inputs as LabInputs & { dOut: Matrix };

function paramsOf(inputs: LabInputs): AttentionParams {
  const { wq, bq, wk, bk, wv, bv, wo, bo } = inputs;
  return { wq, bq, wk, bk, wv, bv, wo, bo };
}

function run() {
  const inputs = forwardInputs();
  return {
    inputs,
    dh: inputs.Xn[0]!.length / inputs.H,
    ...attentionForward(inputs.Xn, paramsOf(inputs), inputs.H),
  };
}

const CAPTION =
  "Hand-picked numbers, one row per word of the sentence. They do not encode the words and are not learned values.";

const HEATMAP = {
  kind: "illustrative",
  rowLabels: LAB_TOKENS,
  colLabels: LAB_TOKENS,
  xLabel: "reads",
  yLabel: "row",
  caption: CAPTION,
} as const;

// Matrices can be wider than a phone, so they scroll inside their own container.
function Matrices({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className={styles.scroll} tabIndex={0} role="group" aria-label={label}>
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

const vector = (values: number[]) => `[${values.map(showNumber).join(", ")}]`;

export function AttentionProjectionExample({ concept }: { concept: string }) {
  const { inputs, Q, K, V } = run();
  return (
    <Anchored concept={concept}>
      <VizFigure title="Three projections of one input" kind="illustrative" caption={CAPTION}>
        <Matrices label="Input and projections">
          <ResultMatrix label="Xn" testId="ex-xn" values={inputs.Xn} />
          <ResultMatrix label="Q" testId="ex-q" values={Q} />
          <ResultMatrix label="K" testId="ex-k" values={K} />
          <ResultMatrix label="V" testId="ex-v" values={V} />
        </Matrices>
      </VizFigure>
    </Anchored>
  );
}

export function AttentionSplitExample({ concept }: { concept: string }) {
  const { Q, Qh } = run();
  return (
    <Anchored concept={concept}>
      <VizFigure
        title="Q, cut into two heads"
        kind="illustrative"
        caption={`${CAPTION} Head 0 holds the first two columns of Q, and head 1 the last two.`}
      >
        <Matrices label="Q and its heads">
          <ResultMatrix label="Q" testId="ex-split-q" values={Q} />
          <ResultMatrix label="Qh[0]" testId="ex-qh0" values={Qh[0]!} />
          <ResultMatrix label="Qh[1]" testId="ex-qh1" values={Qh[1]!} />
        </Matrices>
      </VizFigure>
    </Anchored>
  );
}

export function AttentionScoresExample({ concept }: { concept: string }) {
  const { Qh, Kh, raw } = run();
  const i = 2;
  const j = 1;
  const query = Qh[0]![i]!;
  const key = Kh[0]![j]!;
  return (
    <Anchored concept={concept}>
      <Heatmap {...HEATMAP} title="Qh @ Kh^T, head 0" values={raw[0]!} />
      <p data-testid="ex-score-entry" data-exact={raw[0]![i]![j]}>
        One cell, row {i} and column {j}:{" "}
        <code>
          {vector(query)} . {vector(key)} ={" "}
          {query.map((value, c) => `${showNumber(value)} * ${showNumber(key[c]!)}`).join(" + ")} ={" "}
          {showNumber(raw[0]![i]![j]!)}
        </code>
      </p>
    </Anchored>
  );
}

export function AttentionScaleExample({ concept }: { concept: string }) {
  const { dh, raw, scaled } = run();
  const real = resolveDimension("dh").value!;
  return (
    <Anchored concept={concept}>
      <Heatmap {...HEATMAP} title="S, head 0, after scaling" values={scaled[0]!} />
      <p data-testid="ex-scale" data-dh={dh}>
        This example has <code>d_h = {dh}</code>, so the scale is{" "}
        <code>
          sqrt({dh}) = {showNumber(Math.sqrt(dh))}
        </code>
        . One cell:{" "}
        <code>
          {showNumber(raw[0]![2]![1]!)} / {showNumber(Math.sqrt(dh))} ={" "}
          {showNumber(scaled[0]![2]![1]!)}
        </code>
        . The model has <code data-testid="ex-real-dh">d_h = {real}</code>, so its scale is{" "}
        <code>
          sqrt({real}) = {showNumber(Math.sqrt(real))}
        </code>
        .
      </p>
    </Anchored>
  );
}

export function AttentionMaskExample({ concept }: { concept: string }) {
  const { masked } = run();
  return (
    <Anchored concept={concept}>
      <Heatmap {...HEATMAP} title="S, head 0, after the mask" values={masked[0]!} />
    </Anchored>
  );
}

export function AttentionSoftmaxExample({ concept }: { concept: string }) {
  const { A, masked } = run();
  const off = masked[0]!.map((row) => row.map((value) => value === -Infinity));
  return (
    <Anchored concept={concept}>
      <Heatmap {...HEATMAP} title="A, head 0" values={A[0]!} masked={off} precision={3} />
      <DataTable label="Row sums of A, head 0" data-testid="ex-row-sums">
        <thead>
          <tr>
            <th scope="col">Row</th>
            <th scope="col">Sum of the row</th>
            <th scope="col">Masked cells</th>
          </tr>
        </thead>
        <tbody>
          {A[0]!.map((row, i) => (
            <tr key={i}>
              <th scope="row">{LAB_TOKENS[i]}</th>
              <td>
                <code>{row.reduce((total, value) => total + value, 0).toFixed(4)}</code>
              </td>
              <td>{off[i]!.filter(Boolean).length}</td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </Anchored>
  );
}

export function AttentionWeightedExample({ concept }: { concept: string }) {
  const { A, Vh } = run();
  const i = 2;
  const product = matmul(A[0]!, Vh[0]!);
  return (
    <Anchored concept={concept}>
      <VizFigure title="Weighted values, head 0" kind="illustrative" caption={CAPTION}>
        <Matrices label="Weighted values">
          <ResultMatrix label="Vh[0]" testId="ex-vh0" values={Vh[0]!} />
          <ResultMatrix label="(A @ Vh)[0]" testId="ex-avh0" values={product} />
        </Matrices>
        <p data-testid="ex-weighted-row">
          Row {i}, &quot;{LAB_TOKENS[i]}&quot;:{" "}
          <code>
            {A[0]!
              [i]!.slice(0, i + 1)
              .map((weight, j) => `${showNumber(weight)} * ${vector(Vh[0]![j]!)}`)
              .join(" + ")}{" "}
            = {vector(product[i]!)}
          </code>
        </p>
      </VizFigure>
    </Anchored>
  );
}

export function AttentionMergeExample({ concept }: { concept: string }) {
  const { A, Vh, O } = run();
  return (
    <Anchored concept={concept}>
      <VizFigure title="Two heads, side by side" kind="illustrative" caption={CAPTION}>
        <Matrices label="Heads and merged values">
          <ResultMatrix label="(A @ Vh)[0]" testId="ex-merge-0" values={matmul(A[0]!, Vh[0]!)} />
          <ResultMatrix label="(A @ Vh)[1]" testId="ex-merge-1" values={matmul(A[1]!, Vh[1]!)} />
          <ResultMatrix label="O" testId="ex-o" values={O} />
        </Matrices>
      </VizFigure>
    </Anchored>
  );
}

export function AttentionOutputExample({ concept }: { concept: string }) {
  const { inputs, O, out } = run();
  return (
    <Anchored concept={concept}>
      <VizFigure title="The output projection" kind="illustrative" caption={CAPTION}>
        <Matrices label="Output projection">
          <ResultMatrix label="O" testId="ex-output-o" values={O} />
          <ResultMatrix label="wo" testId="ex-wo" values={inputs.wo} />
          <ResultMatrix label="out" testId="ex-out" values={out} />
        </Matrices>
        <p data-testid="ex-bo" data-exact={JSON.stringify(inputs.bo)}>
          <code>out = O @ wo + bo</code>, with <code>bo = {vector(inputs.bo)}</code> added to every
          row.
        </p>
      </VizFigure>
    </Anchored>
  );
}

export function AttentionBackwardExample({ concept }: { concept: string }) {
  const inputs = backwardInputs();
  const params = paramsOf(inputs);
  const { Xn, H, dOut } = inputs;
  const forward = attentionForward(Xn, params, H);
  const back = attentionBackward(dOut, { Xn, ...forward }, params, H);
  const i = 1;
  const j = 2;
  const biasLargest = Math.max(...back.grads.bk.map(Math.abs));
  const outFrom = (next: Partial<AttentionParams>) =>
    outputDot(attentionForward(Xn, { ...params, ...next }, H).out, dOut);

  return (
    <Anchored concept={concept}>
      <VizFigure
        title="One backward pass, head 0"
        kind="illustrative"
        caption="The same hand-picked forward pass as above. dOut is also hand-picked, not computed from a loss."
      >
        <Matrices label="Backward, down to the scores">
          <ResultMatrix label="dOut" testId="back-dout" values={dOut} />
          <ResultMatrix label="dOh[0]" testId="back-doh" values={back.dOh[0]!} />
          <ResultMatrix label="dVh[0]" testId="back-dvh" values={back.dVh[0]!} />
          <ResultMatrix label="dA[0]" testId="back-da" values={back.dA[0]!} />
          <ResultMatrix label="dS[0]" testId="back-ds" values={back.dS[0]!} />
        </Matrices>
        <Matrices label="Backward, queries, keys and the input">
          <ResultMatrix label="dQh[0]" testId="back-dqh" values={back.dQh[0]!} />
          <ResultMatrix label="dKh[0]" testId="back-dkh" values={back.dKh[0]!} />
          <ResultMatrix label="dXn" testId="back-dxn" values={back.dXn} />
        </Matrices>
      </VizFigure>

      <VizFigure
        title="A masked cell gets no gradient"
        kind="illustrative"
        caption={`Row ${i}, column ${j} of head 0. Position ${i} may not read position ${j}, so this cell is masked.`}
      >
        <DataTable label="One masked cell, through softmax backward" data-testid="back-masked">
          <thead>
            <tr>
              <th scope="col">Quantity</th>
              <th scope="col">Code</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Weight</td>
              <td>
                <code>
                  A[0, {i}, {j}]
                </code>
              </td>
              <td data-testid="masked-a" data-exact={forward.A[0]![i]![j]}>
                <code>{showNumber(forward.A[0]![i]![j]!)}</code>
              </td>
            </tr>
            <tr>
              <td>Gradient on the weight</td>
              <td>
                <code>
                  dA[0, {i}, {j}]
                </code>
              </td>
              <td data-testid="masked-da" data-exact={back.dA[0]![i]![j]}>
                <code>{showNumber(back.dA[0]![i]![j]!)}</code>
              </td>
            </tr>
            <tr>
              <td>Gradient on the score</td>
              <td>
                <code>
                  dS[0, {i}, {j}]
                </code>
              </td>
              <td data-testid="masked-ds" data-exact={back.dS[0]![i]![j]}>
                <code>{showNumber(back.dS[0]![i]![j]! + 0)}</code>
              </td>
            </tr>
          </tbody>
        </DataTable>
      </VizFigure>

      <VizFigure
        title="Where the results go"
        kind="illustrative"
        caption="The same run. grads holds the parameter gradients, and dXn is returned to block_backward."
      >
        <Matrices label="Parameter gradients">
          <ResultMatrix label={'grads[p + "wo"]'} testId="back-wo" values={back.grads.wo} />
          <ResultMatrix label={'grads[p + "wq"]'} testId="back-wq" values={back.grads.wq} />
          <ResultMatrix label={'grads[p + "wk"]'} testId="back-wk" values={back.grads.wk} />
          <ResultMatrix label={'grads[p + "wv"]'} testId="back-wv" values={back.grads.wv} />
        </Matrices>
        <p data-testid="back-bk" data-largest={biasLargest}>
          The largest entry of <code>grads[p + &quot;bk&quot;]</code> is{" "}
          <code>{biasLargest.toExponential(1)}</code>, which is round-off around 0.
        </p>
      </VizFigure>

      <GradientCheck
        title="Checking the backward numerically"
        testId="attention-check"
        scalar={
          <>
            <code>(out * dOut).sum()</code>, whose gradient on <code>out</code> is exactly{" "}
            <code>dOut</code>,
          </>
        }
        rows={[
          {
            name: 'grads[p + "wo"]',
            analytic: back.grads.wo,
            numeric: numericalGrad((wo) => outFrom({ wo }), params.wo),
          },
          {
            name: 'grads[p + "wq"]',
            analytic: back.grads.wq,
            numeric: numericalGrad((wq) => outFrom({ wq }), params.wq),
          },
          {
            name: 'grads[p + "wk"]',
            analytic: back.grads.wk,
            numeric: numericalGrad((wk) => outFrom({ wk }), params.wk),
          },
          {
            name: 'grads[p + "wv"]',
            analytic: back.grads.wv,
            numeric: numericalGrad((wv) => outFrom({ wv }), params.wv),
          },
          {
            name: "dXn",
            analytic: back.dXn,
            numeric: numericalGrad((x) => outputDot(attentionForward(x, params, H).out, dOut), Xn),
          },
        ]}
      />
    </Anchored>
  );
}

// Server wrapper: highlights on the server, so Shiki stays out of the browser, and resolves each
// step's lines from the link records.
export async function AttentionLabExample() {
  const symbol = getSymbol("transformer.attention_forward");
  const stepLines = Object.fromEntries(
    LAB_STEPS.map((step) => [
      step.id,
      step.equations.flatMap((equation) => [...highlightedLines(symbol.id, equation)]),
    ]),
  ) as Record<LabStepId, number[]>;
  const lines = await highlightPython(symbol.source);
  const actual = loadActual();
  return (
    <AttentionLab
      inputs={forwardInputs()}
      stepLines={stepLines}
      actual={
        actual.available
          ? { data: actual.attention, provenance: provenanceLine(actual.meta.provenance) }
          : null
      }
      code={{
        lines,
        source: symbol.source,
        filePath: symbol.file,
        functionName: symbol.name,
        startLine: symbol.startLine,
        permalink: symbol.permalink,
        lineEquations: lineEquations(symbol.file, symbol.startLine, lines.length),
      }}
    />
  );
}

export function ActualModelOutputSlot() {
  const actual = loadActual();
  return (
    <section
      className={styles.slot}
      aria-label="Actual Model Output slot"
      data-testid="actual-output-slot"
    >
      <div className={styles.slotTop}>
        <Badge kind="actual" />
        <span>{actual.available ? "One real window" : "No data yet"}</span>
      </div>
      {actual.available ? <ActualAttentionSummary /> : <EmptyActualSlot />}
    </section>
  );
}

function EmptyActualSlot() {
  return (
    <>
      <p>
        This slot is empty, and the matching option in the lab is disabled. Nothing on this page
        comes from a run of the model.
      </p>
      <p>
        What will appear here: the weights <code>A</code> that <code>attention_forward</code>{" "}
        computes when the repository runs on text encoded by its real tokenizer, for each block and
        head. Those tokens are sub-word pieces, not the words of the sentence used above.
      </p>
      <p>
        The panel will state where its numbers came from. Any value read from a checkpoint must say
        that the checkpoint is an ignored local file, trained on a corpus that is not in the pinned
        commit, with training settings that by inference differ from <code>config.py</code>.
      </p>
    </>
  );
}

// Stored shapes of the real functions, run at the sizes in config.py on random inputs.
export function AttentionShapes() {
  const { outputs } = getGoldenCase("attention", "attention_forward.config_shapes");
  const { forward, backward, grads } = outputs as Record<string, Record<string, number[]>>;
  const stages = (order: [string, number[]][]) => order.map(([name, shape]) => ({ name, shape }));
  const caption =
    "Sizes from config.py. These shapes are what the real functions returned on random inputs. No value from that run is shown.";
  const f = forward!;
  const b = backward!;
  const g = grads!;
  return (
    <div className={examples.stack} data-testid="attention-shapes">
      <ShapeFlow
        title="Forward, with the real config"
        kind="illustrative"
        caption={caption}
        stages={stages([
          ["Xn", f.Xn!],
          ["Q", f.Q!],
          ["K", f.K!],
          ["V", f.V!],
          ["Qh", f.Qh!],
          ["Kh", f.Kh!],
          ["Vh", f.Vh!],
          ["S", f.S!],
          ["A", f.A!],
          ["A @ Vh", f.AVh!],
          ["O", f.O!],
          ["out", f.out!],
        ])}
      />
      <ShapeFlow
        title="Backward, with the real config"
        kind="illustrative"
        caption={caption}
        stages={stages([
          ["dOut", b.dOut!],
          ["dOh", b.dOh!],
          ["dVh", b.dVh!],
          ["dA", b.dA!],
          ["dS", b.dS!],
          ["dQh", b.dQh!],
          ["dKh", b.dKh!],
          ["dQ", b.dQ!],
          ["dK", b.dK!],
          ["dV", b.dV!],
          ["dXn", b.dXn!],
        ])}
      />
      <DataTable label="Gradient shapes for the attention parameters" data-testid="grad-shapes">
        <thead>
          <tr>
            <th scope="col">Entry of grads</th>
            <th scope="col">Shape</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(g).map(([name, shape]) => (
            <tr key={name}>
              <td>
                <code>{`grads[p + "${name}"]`}</code>
              </td>
              <td>
                <code>
                  ({shape.join(", ")}
                  {shape.length === 1 ? "," : ""})
                </code>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </div>
  );
}

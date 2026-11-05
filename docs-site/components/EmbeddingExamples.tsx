import { exampleAnchor } from "@/lib/concepts";
import { getGoldenCase } from "@/lib/golden";
import { embedBackward, embedForward, overwritingBackward } from "@/lib/math/embeddings";
import { numericalGrad, outputDot } from "@/lib/math/gradcheck";
import type { Matrix } from "@/lib/math/tensor";
import { loadConfig } from "@/lib/repo";
import { DataTable } from "./DataTable";
import { EmbeddingLab } from "./EmbeddingLab";
import { GradientCheck, Value, WorkedTable } from "./OpsExamples";
import styles from "./Examples.module.css";
import { VizFigure } from "./VizFigure";

// Inputs are the inputs of golden cases. Results come from lib/math, which tests/math.golden.test.ts
// compares with the real functions at 1e-9.

const forwardInputs = () =>
  getGoldenCase("embeddings", "embed_forward.test_file").inputs as {
    tok_emb: Matrix;
    pos_emb: Matrix;
    ids: number[];
  };

const backwardInputs = () =>
  getGoldenCase("embeddings", "embed_backward.test_file").inputs as {
    dX0: Matrix;
    ids: number[];
    V: number;
    Tmax: number;
  };

export function TokenLookupExample({ concept }: { concept: string }) {
  const { tok_emb, pos_emb, ids } = forwardInputs();
  const { tokens } = embedForward(tok_emb, pos_emb, ids);
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Looking up three ids"
        kind="illustrative"
        caption="Hand-picked numbers from test_embedding_worked_example. A real table has Xavier uniform values at first and learned values later, and it is much larger."
      >
        <WorkedTable
          label="Token lookup, step by step"
          testId="token-lookup"
          plain
          rows={[
            { label: "Token table", code: "tok_emb", value: tok_emb },
            { label: "Ids", code: "ids", value: ids },
            { label: "One row per id", code: "tok_emb[ids]", value: tokens, testId: "tokens" },
          ]}
        />
      </VizFigure>
    </div>
  );
}

export function PositionExample({ concept }: { concept: string }) {
  const { tok_emb, pos_emb, ids } = forwardInputs();
  const { positions } = embedForward(tok_emb, pos_emb, ids);
  const short = embedForward(tok_emb, pos_emb, [ids[0]!, ids[1]!]).positions;
  return (
    <div id={exampleAnchor(concept)}>
      <VizFigure
        title="Slicing the position table"
        kind="illustrative"
        caption="Hand-picked numbers from test_embedding_worked_example. The rows depend only on how long the sequence is, not on which ids it holds."
      >
        <WorkedTable
          label="Position rows, step by step"
          testId="position-lookup"
          plain
          rows={[
            { label: "Position table", code: "pos_emb", value: pos_emb },
            { label: "T = 3", code: "pos_emb[: 3]", value: positions, testId: "positions" },
            { label: "T = 2", code: "pos_emb[: 2]", value: short, testId: "positions-short" },
          ]}
        />
      </VizFigure>
    </div>
  );
}

export function EmbeddingLabExample({ concept }: { concept: string }) {
  const { tok_emb, pos_emb, ids } = forwardInputs();
  return (
    <div id={exampleAnchor(concept)}>
      <EmbeddingLab tokEmb={tok_emb} posEmb={pos_emb} ids={ids} dX0={backwardInputs().dX0} />
    </div>
  );
}

export function EmbeddingBackwardExample({ concept }: { concept: string }) {
  const { tok_emb, pos_emb } = forwardInputs();
  const { dX0, ids, V, Tmax } = backwardInputs();
  const { tok, pos } = embedBackward(dX0, ids, V, Tmax);
  const overwritten = overwritingBackward(dX0, ids, V);
  const repeated = ids[0]!;
  const places = ids.flatMap((id, t) => (id === repeated ? [t] : []));

  return (
    <div className={styles.stack} id={exampleAnchor(concept)}>
      <VizFigure
        title="A repeated id accumulates"
        kind="illustrative"
        caption="Hand-picked numbers from test_embedding_backward_sums_repeated_ids. dX0 is chosen by hand, not computed from a loss."
      >
        <WorkedTable
          label="Embedding backward, step by step"
          testId="embed-backward"
          plain
          rows={[
            { label: "Ids", code: "ids", value: ids },
            { label: "Gradient on the sum", code: "dX0", value: dX0 },
            {
              label: "Token table gradient",
              code: 'np.add.at(grads["tok_emb"], ids, dX0)',
              value: tok,
              testId: "dtok",
            },
            {
              label: "Position table gradient",
              code: 'grads["pos_emb"][: len(ids)] += dX0',
              value: pos,
              testId: "dpos",
            },
            {
              label: "Not in the code, for contrast",
              code: "g[ids] += dX0",
              value: overwritten,
              testId: "overwritten",
            },
          ]}
        />
        <p data-testid="dtok-entry">
          One row:{" "}
          <code>
            grads[&quot;tok_emb&quot;][{repeated}] = {places.map((t) => `dX0[${t}]`).join(" + ")} ={" "}
            {places.map((t) => `[${dX0[t]!.join(", ")}]`).join(" + ")} = [
            {tok[repeated]!.join(", ")}]
          </code>
        </p>
      </VizFigure>
      <GradientCheck
        title="Checking the backward numerically"
        testId="embed-check"
        scalar={
          <>
            <code>(X0 * dX0).sum()</code>, whose gradient on <code>X0</code> is exactly{" "}
            <code>dX0</code>,
          </>
        }
        rows={[
          {
            name: 'grads["tok_emb"]',
            analytic: tok,
            numeric: numericalGrad(
              (v) => outputDot(embedForward(v, pos_emb, ids).X0, dX0),
              tok_emb,
            ),
          },
          {
            name: 'grads["pos_emb"]',
            analytic: pos,
            numeric: numericalGrad(
              (v) => outputDot(embedForward(tok_emb, v, ids).X0, dX0),
              pos_emb,
            ),
          },
        ]}
      />
    </div>
  );
}

// The rows of pos_emb that no sequence reaches at the values in config.py.
export function UnusedPositionRows() {
  const value = (name: string) =>
    loadConfig().constants.find((constant) => constant.name === name)!.value as number;
  const first = value("seq_len");
  const last = value("context_len") - 1;
  return (
    <code data-testid="unused-rows" data-first={first} data-last={last}>
      {first} to {last}
    </code>
  );
}

export function PositionUpdateExample() {
  const none = getGoldenCase("embeddings", "position_update.no_decay");
  const decay = getGoldenCase("embeddings", "position_update.with_decay");
  const { T, Tmax } = none.inputs as { T: number; Tmax: number };
  const { weight_decay } = decay.inputs as { weight_decay: number };
  const gradient = none.outputs.gradient_nonzero as boolean[];
  const unchanged = none.outputs.row_changed as boolean[];
  const decayed = decay.outputs.row_changed as boolean[];
  const yesNo = (flag: boolean) => (flag ? "yes" : "no");

  return (
    <VizFigure
      title="Which rows of pos_emb change in one step"
      kind="illustrative"
      caption={`Stored results of one real loss_and_grads and one real Adam step on the hand-picked tiny model of tests/test_gradcheck.py, which has T = ${T} and ${Tmax} rows. This is not the trained model.`}
    >
      <DataTable label="Which rows of pos_emb change in one step" data-testid="position-update">
        <thead>
          <tr>
            <th scope="col">Row of pos_emb</th>
            <th scope="col">Non-zero gradient</th>
            <th scope="col">Changed, weight_decay = 0</th>
            <th scope="col">
              Changed, weight_decay = <Value value={weight_decay} plain />
            </th>
          </tr>
        </thead>
        <tbody>
          {gradient.map((flag, t) => (
            <tr key={t} data-testid={`position-row-${t}`}>
              <th scope="row">{t}</th>
              <td>{yesNo(flag)}</td>
              <td>{yesNo(unchanged[t]!)}</td>
              <td>{yesNo(decayed[t]!)}</td>
            </tr>
          ))}
        </tbody>
      </DataTable>
    </VizFigure>
  );
}

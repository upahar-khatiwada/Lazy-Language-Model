"use client";

import { useId, useState } from "react";
import { generationSteps, type GenerationStep } from "@/lib/math/sampling";
import type { Matrix } from "@/lib/math/tensor";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { ProbabilityBars } from "./ProbabilityBars";
import { showNumber } from "./ResultMatrix";
import { StepPlayer, type PlayerStep } from "./StepPlayer";
import { VizFigure } from "./VizFigure";

export type StepperInputs = {
  tokens: string[];
  table: Matrix;
  seedIds: number[];
  maxNewTokens: number;
  window: number;
  settings: { temperature: number; topK: number; topP: number };
  // The draws from running the real generate.generate. Empty for the greedy mode.
  draws: number[];
};

export type StepperMode = { id: string; label: string; note: string; inputs: StepperInputs };

// The engine's mirror of generate.generate, with the hand-picked table in place of the model.
export function stepperSteps({
  table,
  seedIds,
  maxNewTokens,
  window,
  settings,
  draws,
}: StepperInputs): GenerationStep[] {
  return generationSteps({
    seedIds,
    maxNewTokens,
    window,
    settings,
    logitsRows: (ids) => ids.map((id) => table[id]!),
    pick: (step) => draws[step]!,
  });
}

function Sequence({
  ids,
  windowLength,
  added,
  tokens,
  testId,
}: {
  ids: number[];
  windowLength: number;
  added: boolean;
  tokens: string[];
  testId: string;
}) {
  const firstInWindow = ids.length - windowLength - (added ? 1 : 0);
  return (
    <ul className={styles.tokens} data-testid={testId}>
      {ids.map((id, i) => {
        const isNew = added && i === ids.length - 1;
        const inWindow = !isNew && i >= firstInWindow;
        return (
          <li key={i}>
            <code
              className={isNew ? styles.tokenNew : inWindow ? styles.tokenWindow : styles.tokenOut}
              data-state={isNew ? "new" : inWindow ? "window" : "outside"}
            >
              {tokens[id]}
            </code>
          </li>
        );
      })}
    </ul>
  );
}

function StepPanel({ step, inputs }: { step: GenerationStep; inputs: StepperInputs }) {
  const { tokens } = inputs;
  const names = (ids: number[]) => ids.map((id) => tokens[id]).join(" ");
  const last = step.window.at(-1)!;
  return (
    <div className={styles.stack}>
      <p>
        <code>{"ids[-window:]"}</code> is{" "}
        <code data-testid="stepper-window">{names(step.window)}</code>, so <code>forward</code>{" "}
        receives {step.window.length} id
        {step.window.length === 1 ? "" : "s"} and returns logits of shape{" "}
        <code>
          ({step.window.length}, {tokens.length})
        </code>
        . Only <code>logits[-1]</code> is used, the row for the last token,{" "}
        <code>{tokens[last]}</code>.
      </p>
      {step.result.kind === "greedy" ? (
        <DataTable label="logits[-1]" data-testid="stepper-logits">
          <thead>
            <tr>
              {tokens.map((token) => (
                <th scope="col" key={token}>
                  {token}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              {step.lastLogits.map((value, i) => (
                <td key={i} className={i === step.next ? styles.chosen : undefined}>
                  <code>{showNumber(value)}</code>
                </td>
              ))}
            </tr>
          </tbody>
        </DataTable>
      ) : (
        <ProbabilityBars
          title="What rng.choice receives on this step"
          kind="illustrative"
          caption="From the table of hand-picked logits below. Tokens marked removed were cut by top-p."
          labels={tokens}
          probabilities={step.result.p}
          showRemoved
        />
      )}
      <p>
        {step.result.kind === "greedy" ? (
          <>
            <code>np.argmax</code> picks{" "}
          </>
        ) : (
          <>
            <code>rng.choice</code> drew{" "}
          </>
        )}
        <code data-testid="stepper-next">{tokens[step.next]}</code>, which is appended.
      </p>
      <Sequence
        ids={[...step.ids, step.next]}
        windowLength={step.window.length}
        added
        tokens={tokens}
        testId="stepper-sequence"
      />
    </div>
  );
}

export function GenerationStepper({ modes }: { modes: StepperMode[] }) {
  const [modeId, setModeId] = useState(modes[0]!.id);
  const selectId = useId();
  const mode = modes.find((candidate) => candidate.id === modeId)!;
  const { inputs } = mode;
  const { tokens, table } = inputs;

  const steps = stepperSteps(inputs);
  const player: PlayerStep[] = [
    {
      id: "seed",
      title: "Seed",
      content: (
        <div className={styles.stack}>
          <p>
            The list starts as a copy of <code>seed_ids</code>. Nothing has been sampled yet.
          </p>
          <Sequence
            ids={inputs.seedIds}
            windowLength={inputs.seedIds.length}
            added={false}
            tokens={tokens}
            testId="stepper-sequence"
          />
        </div>
      ),
    },
    ...steps.map((step, i) => ({
      id: `step-${i}`,
      title: `Append ${tokens[step.next]}`,
      content: <StepPanel step={step} inputs={inputs} />,
    })),
  ];

  return (
    <VizFigure
      title="The sequence growing, one token per step"
      kind="illustrative"
      caption={mode.note}
    >
      <div className={styles.stack} data-testid="generation-stepper">
        <div className={styles.control}>
          <label htmlFor={selectId}>Settings</label>
          <select
            id={selectId}
            data-testid="stepper-mode"
            value={modeId}
            onChange={(event) => setModeId(event.target.value)}
          >
            {modes.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.label}
              </option>
            ))}
          </select>
        </div>
        <StepPlayer key={modeId} label="Generation steps" steps={player} />
        <p className={styles.legend}>
          In each sequence: <code className={styles.tokenWindow}>in the window</code>{" "}
          <code className={styles.tokenOut}>outside the window</code>{" "}
          <code className={styles.tokenNew}>appended on this step</code>
        </p>
        <DataTable
          label="The hand-picked table used in place of the model"
          data-testid="stepper-table"
        >
          <thead>
            <tr>
              <th scope="col">Last token</th>
              {tokens.map((token) => (
                <th scope="col" key={token}>
                  {token}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.map((row, r) => (
              <tr key={r}>
                <th scope="row">{tokens[r]}</th>
                {row.map((value, c) => (
                  <td key={c}>
                    <code>{showNumber(value)}</code>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </DataTable>
      </div>
    </VizFigure>
  );
}

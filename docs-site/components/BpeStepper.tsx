"use client";

import { useState, type ReactNode } from "react";
import { bpeSteps, type BpeStep, type PairCount, type WordSplit } from "@/lib/math/bpe";
import { DataTable } from "./DataTable";
import styles from "./Examples.module.css";
import { StepPlayer, type PlayerStep } from "./StepPlayer";
import { showToken, Tokens } from "./Tokens";
import { VizFigure } from "./VizFigure";

const samePair = (a: PairCount, b: PairCount) => a.pair[0] === b.pair[0] && a.pair[1] === b.pair[1];

function WordTable({ splits, weights }: { splits: WordSplit[]; weights: number[] }) {
  return (
    <DataTable label="Word table" data-testid="bpe-words">
      <thead>
        <tr>
          <th scope="col">Pre-token</th>
          <th scope="col">Count</th>
          <th scope="col">Current split</th>
        </tr>
      </thead>
      <tbody>
        {splits.map(([word, tokens], i) => (
          <tr key={word}>
            <td>
              <code>{showToken(word)}</code>
            </td>
            <td>{weights[i]}</td>
            <td>
              <Tokens tokens={tokens} />
            </td>
          </tr>
        ))}
      </tbody>
    </DataTable>
  );
}

function PairTable({ step }: { step: BpeStep }) {
  return (
    <DataTable label="Pair counts" data-testid="bpe-pairs">
      <thead>
        <tr>
          <th scope="col">Pair</th>
          <th scope="col">Frequency</th>
          <th scope="col">Result</th>
        </tr>
      </thead>
      <tbody>
        {step.counts.map((entry) => {
          const chosen = samePair(entry, step.best);
          const tied = step.tied.some((other) => samePair(entry, other));
          return (
            <tr
              key={JSON.stringify(entry.pair)}
              className={chosen ? styles.chosen : undefined}
              data-chosen={chosen ? "true" : undefined}
            >
              <td>
                <Tokens tokens={entry.pair} />
              </td>
              <td>{entry.count}</td>
              <td>{chosen ? "Chosen" : tied ? "Tied, but found later" : ""}</td>
            </tr>
          );
        })}
      </tbody>
    </DataTable>
  );
}

function Stack({ children }: { children: ReactNode }) {
  return <div className={styles.stack}>{children}</div>;
}

// Steps come from lib/math/bpe.ts. tests/math.golden.test.ts compares them with the real train_bpe.
function buildSteps(text: string): PlayerStep[] {
  const { wordFreqs, initial, steps } = bpeSteps(text);
  const weights = wordFreqs.map(([, weight]) => weight);
  const merges = (count: number) => steps.slice(0, count).map(({ merged }) => merged);

  return [
    {
      id: "start",
      title: "Start",
      content: (
        <Stack>
          <p>
            Pre-tokenize the text, count each pre-token, and split each one into characters. Each
            pre-token keeps its own list, so a pair can only come from inside one pre-token.
          </p>
          <WordTable splits={initial} weights={weights} />
        </Stack>
      ),
    },
    ...steps.map((step, i) => ({
      id: `merge-${i + 1}`,
      title: `Merge ${i + 1}`,
      content: (
        <Stack>
          <p data-testid="bpe-chosen">
            Chosen pair:{" "}
            <code>
              {showToken(step.best.pair[0])} + {showToken(step.best.pair[1])}
            </code>{" "}
            with frequency {step.best.count}, which gives <code>{showToken(step.merged)}</code>.
            {step.tied.length > 0
              ? " Other pairs have the same frequency. The strict > keeps the first one found."
              : ""}
          </p>
          <PairTable step={step} />
          <p>Splits after this merge:</p>
          <WordTable splits={step.after} weights={weights} />
          <p>Merge list so far:</p>
          <Tokens tokens={merges(i + 1)} testId="bpe-merge-list" />
        </Stack>
      ),
    })),
    {
      id: "stop",
      title: "Stop",
      content: (
        <Stack>
          <p data-testid="bpe-stop">
            Every pre-token is now one token, so no adjacent pair is left and <code>pair_freq</code>{" "}
            is empty. Training stops here with {steps.length} merges, even though more were allowed.
          </p>
          <WordTable splits={steps.at(-1)?.after ?? initial} weights={weights} />
          <p>The merge list, in order:</p>
          <Tokens tokens={merges(steps.length)} testId="bpe-merge-list" />
        </Stack>
      ),
    },
  ];
}

export function BpeStepper({ texts, numMerges }: { texts: string[]; numMerges: number }) {
  const [text, setText] = useState(texts[0]!);
  return (
    <VizFigure
      title="BPE, one merge at a time"
      kind="illustrative"
      caption={`A tiny text chosen by hand, not the corpus. These are not the merges of the real model, which trains on the corpus with up to ${numMerges} merges. A test compares these steps with the real train_bpe on the same text.`}
    >
      <label className={styles.control}>
        Text
        <select
          className={styles.fieldWide}
          data-testid="bpe-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
        >
          {texts.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      </label>
      {/* A new text starts again from the first step. */}
      <StepPlayer key={text} label="BPE steps" steps={buildSteps(text)} />
    </VizFigure>
  );
}

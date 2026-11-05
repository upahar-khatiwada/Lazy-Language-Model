"use client";

import { useState } from "react";
import { CodeFrame, type CodeFrameProps } from "./CodeFrame";
import styles from "./GradientFlow.module.css";

export type FlowCode = Omit<CodeFrameProps, "highlightLines">;

// A flow step with everything resolved on the server, so no file access or highlighter runs here.
export type ResolvedFlowStep = {
  id: string;
  output: string;
  label: string;
  shape: string | null;
  values: string | null;
  // The real source of the statement, and where it is.
  statement: string;
  location: string;
  href: string;
  isCall: boolean;
  writes: string;
  note: string | null;
  caller: string;
  callee: string | null;
  lines: number[];
};

export function GradientFlow({
  title,
  steps,
  code,
  firstOutput,
  firstNote,
}: {
  title: string;
  steps: ResolvedFlowStep[];
  code: Record<string, FlowCode>;
  firstOutput?: string;
  firstNote?: string;
}) {
  const [selected, setSelected] = useState(steps[0]!.id);
  const current = steps.find((step) => step.id === selected)!;
  const callee = current.callee ? code[current.callee] : null;
  const caller = code[current.caller]!;

  return (
    <figure className={styles.flow} data-testid="gradient-flow">
      <figcaption className={styles.title}>{title}</figcaption>
      {firstOutput ? (
        <p className={styles.start} data-testid="flow-start">
          <code>{firstOutput}</code> {firstNote}
        </p>
      ) : null}
      <ol className={styles.steps}>
        {steps.map((step) => (
          <li className={styles.step} key={step.id} data-testid={`flow-step-${step.id}`}>
            <div className={styles.arrow}>
              <span className={styles.head} aria-hidden="true">
                ↓
              </span>
              <span className={styles.call}>
                <code data-testid="flow-statement">{step.statement}</code>
                <a className="code-link" href={step.href}>
                  {step.location}
                </a>
                {step.isCall ? null : <span className={styles.inlineMark}>No function call</span>}
              </span>
            </div>
            <button
              type="button"
              className={styles.node}
              aria-pressed={step.id === selected}
              data-testid={`flow-node-${step.id}`}
              onClick={() => setSelected(step.id)}
            >
              <span className={styles.nodeName}>
                <code>{step.output}</code> {step.label}
              </span>
              {step.shape ? (
                <span className={styles.shape}>
                  <code>{step.shape}</code>
                  {step.values && step.values !== step.shape ? <code>{step.values}</code> : null}
                </span>
              ) : null}
            </button>
            {step.writes ? (
              <p className={styles.writes} data-testid="flow-writes">
                Fills <code>grads</code>: {step.writes}
              </p>
            ) : null}
            {step.note ? <p className={styles.writes}>{step.note}</p> : null}
          </li>
        ))}
      </ol>
      <section className={styles.panel} aria-label="Selected step" data-testid="flow-panel">
        <p className={styles.panelTitle} aria-live="polite" data-testid="flow-selected">
          Selected: <code>{current.output}</code>, {current.label}, at{" "}
          <a className="code-link" href={current.href}>
            {current.location}
          </a>
        </p>
        {callee ? (
          <div data-testid="flow-callee">
            <CodeFrame {...callee} />
          </div>
        ) : null}
        <div data-testid="flow-caller">
          <CodeFrame {...caller} highlightLines={new Set(current.lines)} />
        </div>
      </section>
    </figure>
  );
}

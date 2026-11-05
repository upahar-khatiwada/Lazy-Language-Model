"use client";

import { useState, type ReactNode } from "react";
import { buttonClass } from "./Button";
import styles from "./StepPlayer.module.css";

export type PlayerStep = { id: string; title: string; content: ReactNode };

// Steps change only when the reader asks. Nothing plays on its own.
export function StepPlayer({ label, steps }: { label: string; steps: PlayerStep[] }) {
  const [index, setIndex] = useState(0);
  const current = steps[index];
  if (!current) throw new Error(`StepPlayer "${label}" has no step ${index}`);
  const last = steps.length - 1;

  return (
    <section className={styles.player} aria-label={label} data-testid="step-player">
      <ol className={styles.list}>
        {steps.map((step, i) => (
          <li key={step.id}>
            <button
              type="button"
              className={styles.stepButton}
              aria-current={i === index ? "step" : undefined}
              onClick={() => setIndex(i)}
            >
              <span aria-hidden="true">{i + 1}.</span> {step.title}
            </button>
          </li>
        ))}
      </ol>
      <div className={styles.controls}>
        {/* aria-disabled, not disabled: a disabled button would drop keyboard focus at either end. */}
        <button
          type="button"
          className={`${buttonClass()} ${styles.control}`}
          aria-disabled={index === 0}
          onClick={() => setIndex(Math.max(0, index - 1))}
        >
          Previous
        </button>
        <button
          type="button"
          className={`${buttonClass()} ${styles.control}`}
          aria-disabled={index === last}
          onClick={() => setIndex(Math.min(last, index + 1))}
        >
          Next
        </button>
        <span className={styles.status} aria-live="polite" data-testid="step-status">
          Step {index + 1} of {steps.length}: {current.title}
        </span>
      </div>
      <div className={styles.panel}>{current.content}</div>
    </section>
  );
}

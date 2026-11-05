"use client";

import Link from "next/link";
import { useState } from "react";
import { Badge } from "./Badge";
import { InlineCode } from "./InlineCode";
import { CodeFrame, type CodeFrameProps } from "./CodeFrame";
import type { FlowCode } from "./GradientFlow";
import styles from "./LearningPipeline.module.css";

export type ResolvedLearningLink = { statement: string; location: string; href: string };

// A stage with everything resolved on the server, so no file access or highlighter runs here.
export type ResolvedLearningStage = {
  id: string;
  label: string;
  summary: string;
  page: { title: string; href: string | null };
  links: ResolvedLearningLink[];
  also: (ResolvedLearningLink & { note: string }) | null;
  symbol: string;
  lines: number[];
  value: { label: string; text: string; exact?: string };
};

export function LearningPipeline({
  stages,
  code,
}: {
  stages: ResolvedLearningStage[];
  code: Record<string, FlowCode & Pick<CodeFrameProps, "source">>;
}) {
  const [selected, setSelected] = useState(stages[0]!.id);
  const index = stages.findIndex((stage) => stage.id === selected);
  const current = stages[index]!;

  return (
    <figure className={styles.figure} data-testid="learning-pipeline">
      <figcaption className={styles.title}>
        One learning step, in the order of <code>train.main</code>. Select a stage.
      </figcaption>
      <ol className={styles.stages}>
        {stages.map((stage, i) => (
          <li key={stage.id}>
            <button
              type="button"
              className={styles.stage}
              aria-pressed={stage.id === selected}
              data-testid={`learning-stage-${stage.id}`}
              onClick={() => setSelected(stage.id)}
            >
              <span className={styles.number}>{i + 1}</span>
              {stage.label}
            </button>
          </li>
        ))}
      </ol>
      <section
        className={styles.panel}
        aria-label={`Selected stage: ${current.label}`}
        data-testid="learning-panel"
      >
        <h3 className={styles.panelTitle} aria-live="polite" data-testid="learning-selected">
          {index + 1}. {current.label}
        </h3>
        <p>
          <InlineCode text={current.summary} />
        </p>
        <p data-testid="learning-page">
          Page:{" "}
          {current.page.href ? (
            <Link href={current.page.href}>{current.page.title}</Link>
          ) : (
            <span aria-disabled="true" title="This page is not written yet">
              {current.page.title} (page not written yet)
            </span>
          )}
        </p>
        <div>
          <p className={styles.path}>Where it runs, starting from train.main:</p>
          <ol className={styles.links} data-testid="learning-links">
            {current.links.map((link) => (
              <li key={`${link.location}:${link.statement}`} data-testid="learning-link">
                <code>{link.statement}</code>{" "}
                <a className="code-link" href={link.href}>
                  {link.location}
                </a>
              </li>
            ))}
          </ol>
          {current.also ? (
            <p className={styles.also} data-testid="learning-also">
              {current.also.note} <code>{current.also.statement}</code>{" "}
              <a className="code-link" href={current.also.href}>
                {current.also.location}
              </a>
            </p>
          ) : null}
        </div>
        <p className={styles.value} data-testid="learning-value">
          <Badge kind="illustrative" /> {current.value.label}:{" "}
          <code data-exact={current.value.exact}>{current.value.text}</code>
        </p>
        <div data-testid="learning-code">
          <CodeFrame {...code[current.symbol]!} highlightLines={new Set(current.lines)} />
        </div>
      </section>
    </figure>
  );
}

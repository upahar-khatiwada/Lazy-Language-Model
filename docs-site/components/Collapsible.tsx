import type { ReactNode } from "react";
import styles from "./Collapsible.module.css";
import { ConceptStep } from "./ConceptStep";

// lib/remark-collapse.ts places these on a page, so MDX never names this component. The summary keeps
// the step's real h3, so the heading order and the nine steps of the page protocol do not change.
// No JavaScript here: <details> works by itself and keeps its content in the page.
export function Collapsible({
  step,
  title,
  open,
  children,
}: {
  step?: string;
  title?: string;
  open?: boolean;
  children: ReactNode;
}) {
  return (
    <details className={styles.section} data-collapsible="" open={open}>
      <summary className={styles.summary}>
        {step ? <ConceptStep>{step}</ConceptStep> : title}
      </summary>
      <div className={styles.body}>{children}</div>
    </details>
  );
}

import type { ReactNode } from "react";
import styles from "./Callout.module.css";

const LABELS = { note: "Note", warning: "Warning", source: "Source" } as const;

export type CalloutKind = keyof typeof LABELS;

export function Callout({ kind, children }: { kind: CalloutKind; children: ReactNode }) {
  return (
    <div className={`${styles.callout} ${styles[kind]}`}>
      <span className={styles.label}>{LABELS[kind]}</span>
      {children}
    </div>
  );
}

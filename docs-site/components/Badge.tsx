import styles from "./Badge.module.css";

const LABELS = {
  illustrative: "Illustrative Example",
  actual: "Actual Model Output",
  derived: "Derived from implementation",
  implemented: "Implemented",
} as const;

export type BadgeKind = keyof typeof LABELS;

export function Badge({ kind }: { kind: BadgeKind }) {
  return <span className={`${styles.badge} ${styles[kind]}`}>{LABELS[kind]}</span>;
}

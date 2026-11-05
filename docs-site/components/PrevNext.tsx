import Link from "next/link";
import type { ResolvedPage } from "@/lib/nav";
import styles from "./PrevNext.module.css";

export function PrevNext({ prev, next }: { prev?: ResolvedPage; next?: ResolvedPage }) {
  if (!prev && !next) return null;
  return (
    <nav className={styles.nav} aria-label="Previous and next pages">
      {prev ? (
        <Link className={styles.link} href={prev.href} rel="prev">
          <span className={styles.label}>Previous</span>
          {prev.title}
        </Link>
      ) : null}
      {next ? (
        <Link className={`${styles.link} ${styles.next}`} href={next.href} rel="next">
          <span className={styles.label}>Next</span>
          {next.title}
        </Link>
      ) : null}
    </nav>
  );
}

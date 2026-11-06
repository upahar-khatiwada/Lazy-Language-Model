import type { ReactNode } from "react";
import styles from "./DocMain.module.css";

// Not "main": train.py and talk.py define a function by that name, which the Code Explorer uses as an
// anchor, and the skip link would then jump to the wrong place.
// `wide` also takes the on-this-page column, for a page that has no such rail.
export function DocMain({ wide = false, children }: { wide?: boolean; children: ReactNode }) {
  return (
    <main
      className={wide ? `${styles.main} ${styles.wide}` : styles.main}
      id="main-content"
      tabIndex={-1}
    >
      {children}
    </main>
  );
}

export function DocHeader({
  title,
  description,
  sourceFile,
}: {
  title: string;
  description: string;
  sourceFile?: string;
}) {
  return (
    <header className={styles.header}>
      <h1>{title}</h1>
      <p className={styles.description}>{description}</p>
      {sourceFile ? <p className={styles.sourceFile}>{sourceFile}</p> : null}
    </header>
  );
}

export function PageNotice({ children }: { children: ReactNode }) {
  return <div className={styles.notice}>{children}</div>;
}

export function SiteFooter({
  sha,
  shortSha,
  baseUrl,
}: {
  sha: string;
  shortSha: string;
  baseUrl: string;
}) {
  return (
    <footer className={styles.footer}>
      Generated from commit <a href={`${baseUrl}/commit/${sha}`}>{shortSha}</a>
    </footer>
  );
}

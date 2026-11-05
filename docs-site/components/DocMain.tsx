import type { ReactNode } from "react";
import styles from "./DocMain.module.css";

export function DocMain({ children }: { children: ReactNode }) {
  return (
    <main className={styles.main} id="main" tabIndex={-1}>
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

import type { Metadata } from "next";
import { Specimens } from "@/components/Specimens";
import { ThemeToggle } from "@/components/ThemeToggle";
import styles from "./design.module.css";

// Hidden style guide: not in data/nav.ts, not in content/, and marked noindex.
export const metadata: Metadata = {
  title: "Design system",
  robots: { index: false, follow: false },
};

export default function DesignPage() {
  return (
    <main className={styles.page} id="design-top">
      <header className={styles.header}>
        <div>
          <h1>Design system</h1>
          <p>Every element, in both themes. The toggle changes the page behind the panels.</p>
        </div>
        <ThemeToggle />
      </header>
      <div className={styles.panels}>
        <Specimens theme="light" />
        <Specimens theme="dark" />
      </div>
    </main>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useRef, useState, type ReactNode } from "react";
import type { ResolvedGroup } from "@/lib/nav";
import { Button, buttonClass } from "./Button";
import styles from "./DocsChrome.module.css";
import { SearchDialog } from "./SearchDialog";
import { Sidebar } from "./Sidebar";
import { ThemeToggle } from "./ThemeToggle";

const SIDEBAR_ID = "docs-sidebar";

type Props = {
  nav: ResolvedGroup[];
  githubUrl: string;
  children: ReactNode;
};

export function DocsChrome({ nav, githubUrl, children }: Props) {
  const pathname = usePathname();
  // Remembers which page the drawer was opened on, so moving to another page closes it.
  const [openOn, setOpenOn] = useState<string | null>(null);
  const open = openOn === pathname;
  const toggleRef = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => setOpenOn(null), []);

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className={styles.topbar}>
        <span className={styles.contents}>
          <Button
            ref={toggleRef}
            aria-expanded={open}
            aria-controls={SIDEBAR_ID}
            onClick={() => setOpenOn(open ? null : pathname)}
          >
            Contents
          </Button>
        </span>
        <Link className={styles.brand} href="/">
          <span className={styles.brandLong}>LLM Developer Docs</span>
          <span className={styles.brandShort}>LLM Docs</span>
        </Link>
        <span className={styles.spacer} />
        <SearchDialog />
        <a className={buttonClass()} href={githubUrl}>
          GitHub
        </a>
        <ThemeToggle />
      </header>
      <div className={styles.shell}>
        <Sidebar
          id={SIDEBAR_ID}
          nav={nav}
          pathname={pathname}
          open={open}
          onClose={close}
          toggleRef={toggleRef}
        />
        {children}
      </div>
    </>
  );
}

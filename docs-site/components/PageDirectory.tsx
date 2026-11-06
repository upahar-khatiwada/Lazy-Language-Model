import Link from "next/link";
import { NAV } from "@/data/nav";
import { explorerFiles, explorerHref } from "@/lib/code-explorer";
import { PageLink } from "./PageLink";
import styles from "./PageDirectory.module.css";

// Every page a reader can open, built from the navigation and the code index so it cannot drift.
// PageLink shows a page with no content file as disabled text.
export function PageDirectory() {
  return (
    <div className={styles.directory} data-testid="page-directory">
      {NAV.map((group) => (
        <section className={styles.group} key={group.title}>
          <h3>{group.title}</h3>
          <ul>
            {group.title === NAV[0]!.title ? (
              <li>
                <Link prefetch={false} href="/">
                  Home
                </Link>
              </li>
            ) : null}
            {group.pages.map((page) => (
              <li key={page.slug}>
                <PageLink page={page.slug}>{page.title}</PageLink>
              </li>
            ))}
          </ul>
        </section>
      ))}
      <section className={styles.group}>
        <h3>Source files</h3>
        <ul>
          {explorerFiles().map(({ path }) => (
            <li key={path}>
              <Link prefetch={false} href={explorerHref(path)}>
                <code>{path}</code>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

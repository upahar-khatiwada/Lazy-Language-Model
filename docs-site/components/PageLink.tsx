import Link from "next/link";
import type { ReactNode } from "react";
import { writtenHref } from "@/lib/pages";
import styles from "./PageLink.module.css";

// A link to a documentation page. A page that is not written yet has no route, so it shows as
// disabled text: not a link and not a tab stop.
export function PageLink({
  page,
  hash,
  children,
}: {
  page: string;
  hash?: string;
  children: ReactNode;
}) {
  const href = writtenHref(page);
  if (href) return <Link href={hash ? `${href}#${hash}` : href}>{children}</Link>;
  return (
    <span className={styles.disabled} aria-disabled="true" title="This page is not written yet">
      {children}
      <span className="visually-hidden"> (page not written yet)</span>
    </span>
  );
}

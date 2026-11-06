import Link from "next/link";
import type { Crumb } from "@/lib/nav";
import styles from "./Breadcrumbs.module.css";

export function Breadcrumbs({ crumbs }: { crumbs: Crumb[] }) {
  return (
    <nav aria-label="Breadcrumb">
      <ol className={styles.list}>
        {crumbs.map((crumb, index) => (
          <li className={styles.item} key={crumb.title}>
            {crumb.href ? (
              <Link prefetch={false} className={styles.link} href={crumb.href}>
                {crumb.title}
              </Link>
            ) : (
              <span aria-current={index === crumbs.length - 1 ? "page" : undefined}>
                {crumb.title}
              </span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}

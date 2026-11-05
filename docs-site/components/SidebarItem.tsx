import Link from "next/link";
import styles from "./SidebarItem.module.css";

type Props = {
  title: string;
  sourceFile?: string;
  href: string;
  state: "default" | "active" | "disabled";
  // Set only on the /design page, to show a state outside real navigation.
  preview?: { hover?: boolean };
};

export function SidebarItem({ title, sourceFile, href, state, preview }: Props) {
  const content = (
    <>
      <span className={styles.title}>{title}</span>
      {sourceFile ? <span className={styles.file}>{sourceFile}</span> : null}
    </>
  );

  // A page that is not written yet has no route, so it must not be a link or a tab stop.
  if (state === "disabled") {
    return (
      <span className={`${styles.item} ${styles.disabled}`} aria-disabled="true">
        {content}
      </span>
    );
  }

  const stateClass = preview?.hover ? styles.hover : styles[state];
  return (
    <Link
      className={`${styles.item} ${stateClass}`}
      href={href}
      aria-current={state === "active" && !preview ? "page" : undefined}
    >
      {content}
    </Link>
  );
}

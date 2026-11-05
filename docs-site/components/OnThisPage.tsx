"use client";

import { useEffect, useState } from "react";
import type { PageHeading } from "@/lib/remark-headings";
import styles from "./OnThisPage.module.css";

// A heading counts as reached once it is above this fraction of the viewport height.
const READING_LINE = 0.34;

export function OnThisPage({ headings }: { headings: PageHeading[] }) {
  const [activeId, setActiveId] = useState(headings[0]?.id);

  useEffect(() => {
    const elements = headings.flatMap((heading) => {
      const element = document.getElementById(heading.id);
      return element ? [element] : [];
    });

    // The observer only signals that a heading crossed the reading line. Positions decide which
    // one is current, so the result is also right when scrolling upward through a long section.
    const observer = new IntersectionObserver(
      () => {
        const line = window.innerHeight * READING_LINE;
        const reached = elements.filter((element) => element.getBoundingClientRect().top <= line);
        setActiveId((reached.at(-1) ?? elements[0])?.id);
      },
      { rootMargin: `0px 0px -${(1 - READING_LINE) * 100}% 0px` },
    );
    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [headings]);

  if (headings.length === 0) return null;

  return (
    <nav className={styles.rail} aria-label="On this page">
      <div className={styles.title}>On this page</div>
      <ul className={styles.list}>
        {headings.map((heading) => {
          const active = heading.id === activeId;
          return (
            <li className={heading.depth === 3 ? styles.nested : undefined} key={heading.id}>
              <a
                className={`${styles.link} ${active ? styles.active : ""}`}
                href={`#${heading.id}`}
                aria-current={active ? "location" : undefined}
                onClick={() => setActiveId(heading.id)}
              >
                {heading.text}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

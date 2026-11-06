"use client";

import { useEffect, useRef, type FocusEvent, type RefObject } from "react";
import { activeHref, type ResolvedGroup } from "@/lib/nav";
import styles from "./Sidebar.module.css";
import { SidebarItem } from "./SidebarItem";

type Props = {
  id: string;
  nav: ResolvedGroup[];
  pathname: string;
  open: boolean;
  onClose: () => void;
  toggleRef: RefObject<HTMLButtonElement | null>;
};

// Moves only the sidebar's own scroll position, so the page does not jump.
function revealActiveItem(sidebar: HTMLElement) {
  const item = sidebar.querySelector<HTMLElement>('[aria-current="page"]');
  if (!item || sidebar.clientHeight === 0) return;
  const sidebarBox = sidebar.getBoundingClientRect();
  const itemBox = item.getBoundingClientRect();
  if (itemBox.top < sidebarBox.top || itemBox.bottom > sidebarBox.bottom) {
    sidebar.scrollTop += itemBox.top - sidebarBox.top - (sidebar.clientHeight - itemBox.height) / 2;
  }
}

export function Sidebar({ id, nav, pathname, open, onClose, toggleRef }: Props) {
  const sidebarRef = useRef<HTMLElement>(null);
  const active = activeHref(nav, pathname);

  useEffect(() => {
    if (sidebarRef.current) revealActiveItem(sidebarRef.current);
  }, [pathname, open]);

  useEffect(() => {
    if (!open) return;
    sidebarRef.current?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        toggleRef.current?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose, toggleRef]);

  // Tabbing out of the drawer closes it, so focus never lands on content hidden underneath.
  function onBlur(event: FocusEvent<HTMLElement>) {
    const next = event.relatedTarget;
    if (!open || !next) return;
    if (!event.currentTarget.contains(next) && next !== toggleRef.current) onClose();
  }

  return (
    <nav
      className={`${styles.sidebar} ${open ? styles.open : ""}`}
      id={id}
      aria-label="Documentation"
      ref={sidebarRef}
      tabIndex={open ? -1 : undefined}
      onBlur={onBlur}
    >
      {nav.map((group) => (
        <div className={styles.group} key={group.title}>
          <div className={styles.groupTitle}>{group.title}</div>
          <ul className={styles.list}>
            {group.pages.map((page) => (
              <li key={page.slug}>
                <SidebarItem
                  title={page.title}
                  sourceFile={page.sourceFile}
                  href={page.href}
                  state={!page.enabled ? "disabled" : page.href === active ? "active" : "default"}
                />
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

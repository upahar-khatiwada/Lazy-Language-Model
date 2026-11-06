"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useEffect, useEffectEvent, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  activeEquation,
  announcement,
  detailUrl,
  EQ_PARAM,
  lineCountText,
  ROLE_LABELS,
  scrollTargetIndex,
  splitEquations,
  withEquation,
  type EquationDetail,
} from "@/lib/math-code";
import { revealEquation } from "@/lib/reveal";
import styles from "./MathCodeNavigator.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const TRIGGER = "[data-eq], [data-eqs], [data-eq-hover]";

type Popover = {
  pathname: string;
  slugs: string[];
  line: number | null;
  top: number;
  left: number;
};

// Next keeps useSearchParams in step with replaceState, but only when the state passed in is not its
// own. Passing window.history.state here would leave the highlight behind the address.
function pin(slug: string | null) {
  const { pathname, search, hash } = window.location;
  window.history.replaceState(null, "", withEquation(pathname + search + hash, slug));
}

// The first equation named by an element that takes part in linking.
function slugOf(element: Element | null): string | null {
  const trigger = element?.closest<HTMLElement>(TRIGGER);
  if (!trigger) return null;
  return (
    trigger.dataset.eq ?? trigger.dataset.eqHover ?? splitEquations(trigger.dataset.eqs)[0] ?? null
  );
}

// display: none hides the other steps of a step player, and those lines are not on screen.
function visibleLines(slug: string): HTMLElement[] {
  return [...document.querySelectorAll<HTMLElement>(`[data-eqs~="${slug}"]`)].filter(
    (element) => element.getClientRects().length > 0,
  );
}

function scrollToCode(slug: string) {
  revealEquation(slug);
  const lines = visibleLines(slug);
  const equation = document.querySelector(`[data-eq="${slug}"]`);
  const after = lines.map(
    (line) =>
      equation !== null &&
      (equation.compareDocumentPosition(line) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0,
  );
  lines[scrollTargetIndex(after)]?.scrollIntoView({ block: "center", inline: "nearest" });
}

// One instance for the whole site. It reads what the server rendered: data-eq on an equation,
// data-eqs on a code line. The address holds the pinned equation, so ?eq= restores it on load.
export function MathCodeNavigator() {
  const pathname = usePathname();
  const pinned = useSearchParams().get(EQ_PARAM);
  const [focused, setFocused] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [popover, setPopover] = useState<Popover | null>(null);
  const [details, setDetails] = useState<Record<string, EquationDetail | null>>({});
  const countRef = useRef<HTMLSpanElement>(null);
  const liveRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  const openPopover = popover?.pathname === pathname ? popover : null;
  const active = activeEquation(pinned, focused, hovered);
  const pinnedDetail = pinned ? details[pinned] : undefined;
  // The chip appears once its details arrive, and its count is written by the effect below.
  const chipShown = Boolean(pinnedDetail);

  const clearAll = () => {
    pin(null);
    setFocused(null);
    setHovered(null);
    setPopover(null);
    returnFocusRef.current?.focus({ preventScroll: true });
    returnFocusRef.current = null;
  };
  const onEscape = useEffectEvent(clearAll);

  // A pinned equation is meant to be seen, so its folded sections open, before the count below.
  useEffect(() => {
    if (pinned) revealEquation(pinned);
  }, [pinned, pathname]);

  // Marks the page. A step player can replace code later, so the marks follow changes to the page.
  useEffect(() => {
    const apply = () => {
      const marked = new Set(
        active
          ? document.querySelectorAll(`[data-eq="${active}"], [data-eqs~="${active}"]`)
          : undefined,
      );
      document.querySelectorAll("[data-eq-active]").forEach((element) => {
        if (!marked.has(element)) element.removeAttribute("data-eq-active");
      });
      marked.forEach((element) => {
        if (!element.hasAttribute("data-eq-active")) {
          element.setAttribute("data-eq-active", "true");
        }
      });
      if (countRef.current && pinned) {
        countRef.current.textContent = lineCountText(visibleLines(pinned).length);
      }
    };
    apply();
    const main = document.getElementById("main-content");
    if (!main) return;
    const observer = new MutationObserver(apply);
    observer.observe(main, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [active, pinned, pathname, chipShown]);

  // A deep link: scroll to the code, unless the address already names a place on the page.
  useEffect(() => {
    if (pinned && !window.location.hash) scrollToCode(pinned);
    // Only on arrival: pressing a control scrolls by itself, and focus must never scroll.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  // Details load on demand. Hover alone does not fetch, so moving the mouse stays cheap.
  const wanted = [pinned, focused, ...(openPopover?.slugs ?? [])].filter(
    (slug): slug is string => slug !== null,
  );
  const wantedKey = wanted.join(" ");
  useEffect(() => {
    for (const slug of splitEquations(wantedKey)) {
      if (slug in details) continue;
      fetch(detailUrl(slug, BASE_PATH))
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null)
        .then((detail: EquationDetail | null) => {
          setDetails((known) => ({ ...known, [slug]: detail }));
        });
    }
  }, [wantedKey, details]);

  // A screen reader hears the change, for a pinned or focused equation.
  const spoken = focused ?? pinned;
  const spokenLabel = spoken ? details[spoken]?.label : undefined;
  useEffect(() => {
    if (!liveRef.current) return;
    liveRef.current.textContent =
      spoken && spokenLabel ? announcement(spokenLabel, visibleLines(spoken).length) : "";
  }, [spoken, spokenLabel, pathname]);

  useEffect(() => {
    if (openPopover) popoverRef.current?.focus({ preventScroll: true });
  }, [openPopover]);

  useEffect(() => {
    const onPointerOver = (event: PointerEvent) => {
      if (event.pointerType !== "mouse") return;
      const element = event.target as Element;
      setHovered(element.closest("[data-eq-popover]") ? null : slugOf(element));
    };
    const onFocusIn = (event: FocusEvent) => {
      const element = event.target as Element;
      setFocused(element.closest("[data-eq-popover]") ? null : slugOf(element));
    };
    const onFocusOut = () => setFocused(null);

    const onClick = (event: MouseEvent) => {
      const element = event.target as Element;
      if (element.closest("[data-eq-popover]")) return;
      setPopover(null);

      const show = element.closest<HTMLElement>("[data-eq-show]");
      if (show) {
        const slug = slugOf(show);
        if (slug) {
          pin(slug);
          scrollToCode(slug);
        }
        return;
      }
      if (element.closest("a")) return;

      const line = element.closest<HTMLElement>("[data-eqs]");
      if (line) {
        // Selecting code to copy it is not a click on the line.
        if (!element.closest("[data-eq-mark]") && !window.getSelection()?.isCollapsed) return;
        const slugs = splitEquations(line.dataset.eqs);
        const rect = line.getBoundingClientRect();
        pin(slugs[0]!);
        returnFocusRef.current = element.closest<HTMLElement>("[data-eq-mark]");
        setPopover({
          pathname: window.location.pathname,
          slugs,
          line: Number(line.dataset.line) || null,
          top: rect.bottom + window.scrollY + 4,
          left: Math.max(16, Math.min(rect.left + window.scrollX + 48, window.innerWidth - 496)),
        });
        return;
      }

      if (element.closest("button")) return;
      const equation = element.closest<HTMLElement>("[data-eq]");
      if (equation?.dataset.eq) {
        pin(equation.dataset.eq);
        scrollToCode(equation.dataset.eq);
      }
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onEscape();
        return;
      }
      // The equation is focusable so a wide one can scroll. Enter does what a click does.
      const element = event.target as Element;
      if (event.key === "Enter" && element.matches("[data-eq] [role='group']")) {
        const slug = slugOf(element);
        if (slug) {
          pin(slug);
          scrollToCode(slug);
        }
      }
    };

    document.addEventListener("pointerover", onPointerOver);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerover", onPointerOver);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const firstCode = pinnedDetail?.locations.find(
    (location) => location.role === "implements" || location.role === "inline",
  );

  return (
    <>
      <div className="visually-hidden" aria-live="polite" ref={liveRef} data-testid="eq-live" />
      {pinnedDetail ? (
        <div
          className={styles.chip}
          role="group"
          aria-label="Active equation"
          data-testid="eq-chip"
        >
          <strong>{pinnedDetail.label}</strong>
          <span ref={countRef} data-testid="eq-chip-count" />
          {firstCode ? (
            <Link href={firstCode.explorerHref} prefetch={false}>
              {firstCode.file}:{firstCode.startLine}
            </Link>
          ) : null}
          <button type="button" className={styles.clear} onClick={clearAll}>
            Clear<span className="visually-hidden"> the highlighted equation</span>
          </button>
        </div>
      ) : null}
      {openPopover
        ? createPortal(
            <div
              className={styles.popover}
              style={{ top: openPopover.top, left: openPopover.left }}
              role="dialog"
              aria-label={
                openPopover.line ? `Equations for line ${openPopover.line}` : "Linked equations"
              }
              tabIndex={-1}
              ref={popoverRef}
              data-eq-popover
              data-testid="eq-popover"
            >
              {openPopover.slugs.map((slug) => {
                const detail = details[slug];
                if (detail === undefined) return <p key={slug}>Loading the equation.</p>;
                if (detail === null) {
                  return <p key={slug}>This equation could not be loaded. Reload to try again.</p>;
                }
                const here = detail.locations.filter(
                  (location) =>
                    openPopover.line !== null &&
                    location.startLine <= openPopover.line &&
                    openPopover.line <= location.endLine,
                );
                return (
                  <section className={styles.entry} key={slug} data-eq-entry={slug}>
                    <div className={styles.entryTop}>
                      <strong>{detail.label}</strong>
                      {slug === pinned ? null : (
                        <button type="button" className={styles.clear} onClick={() => pin(slug)}>
                          Highlight<span className="visually-hidden"> {detail.label}</span>
                        </button>
                      )}
                    </div>
                    <div
                      className={styles.math}
                      dangerouslySetInnerHTML={{ __html: detail.html }}
                    />
                    {here.length > 0 ? (
                      <p>
                        This line:{" "}
                        {[...new Set(here.map((location) => ROLE_LABELS[location.role]))].join(
                          ", ",
                        )}
                        .
                      </p>
                    ) : null}
                    {detail.inlineNote ? <p>{detail.inlineNote}</p> : null}
                    <p>
                      {detail.homeHref ? (
                        <Link href={detail.homeHref} prefetch={false}>
                          Concept: {detail.conceptTitle}
                        </Link>
                      ) : (
                        <>Concept: {detail.conceptTitle}</>
                      )}
                    </p>
                  </section>
                );
              })}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}

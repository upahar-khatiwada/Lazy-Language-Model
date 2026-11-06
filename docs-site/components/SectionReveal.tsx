"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { openAllForPrint, revealHash } from "@/lib/reveal";

// One instance for the whole site. It opens folded sections when an anchor points into one, and all
// of them while printing. MathCodeNavigator does the same for ?eq= highlights.
export function SectionReveal() {
  const pathname = usePathname();

  // Arriving on a page with a hash. The browser could not scroll to something folded away.
  useEffect(() => {
    revealHash(window.location.hash)?.scrollIntoView();
  }, [pathname]);

  useEffect(() => {
    const onHashChange = () => revealHash(window.location.hash)?.scrollIntoView();

    // Before the browser or Next.js moves, so the target has a position to move to.
    const onClick = (event: MouseEvent) => {
      const link = (event.target as Element).closest?.("a");
      if (link?.hash && link.pathname === window.location.pathname) revealHash(link.hash);
    };

    let opened: HTMLDetailsElement[] = [];
    const onBeforePrint = () => {
      opened = openAllForPrint();
    };
    const onAfterPrint = () => {
      opened.forEach((section) => {
        section.open = false;
      });
      opened = [];
    };

    window.addEventListener("hashchange", onHashChange);
    document.addEventListener("click", onClick, { capture: true });
    window.addEventListener("beforeprint", onBeforePrint);
    window.addEventListener("afterprint", onAfterPrint);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
      document.removeEventListener("click", onClick, { capture: true });
      window.removeEventListener("beforeprint", onBeforePrint);
      window.removeEventListener("afterprint", onAfterPrint);
    };
  }, []);

  return null;
}

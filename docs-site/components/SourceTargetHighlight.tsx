"use client";

import { useEffect } from "react";

// A deep link such as /code/ops/#softmax_forward. A plain load or a same-page link already sets
// :target, but a next/link from another page uses pushState, which does not. This marks the same
// element in that case, and scrolls both the page and the code block to it.
export function SourceTargetHighlight() {
  useEffect(() => {
    let active: HTMLElement | null = null;

    function apply(scroll: boolean) {
      active?.removeAttribute("data-active");
      const id = decodeURIComponent(window.location.hash.slice(1));
      const element = id ? document.getElementById(id) : null;
      active = element?.closest("[data-source-listing]") ? element : null;
      active?.setAttribute("data-active", "true");
      if (scroll) active?.scrollIntoView({ block: "start" });
    }

    const onHashChange = () => apply(false);
    apply(true);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return null;
}

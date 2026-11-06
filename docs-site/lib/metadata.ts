import type { Metadata } from "next";

export const SITE_NAME = "LLM Developer Docs";

export const SITE_DESCRIPTION =
  "How a from-scratch language model in Python and NumPy works: every equation tied to the code that computes it, with worked examples and the hand-written backward pass.";

// A page's own <title> and description, and the same text for link previews. Next replaces a parent's
// openGraph and twitter objects instead of merging them, so each page needs its own. There is no card
// image, because none is available from the repository, so the card is the plain "summary" type.
export function pageMetadata(title: string, description: string): Metadata {
  return {
    title,
    description,
    openGraph: { title, description, type: "article", siteName: SITE_NAME },
    twitter: { card: "summary", title, description },
  };
}

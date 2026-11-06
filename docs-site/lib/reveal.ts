// Folded sections (components/Collapsible.tsx) must open when something points inside them. A browser
// does this for find-in-page and for some anchors, but not for ?eq= highlights, client-side navigation
// or every browser, so the site does it itself.

export const COLLAPSIBLE_SELECTOR = "details[data-collapsible]";

// Opens every <details> around an element, and says whether any was closed.
export function openAncestors(element: Element): boolean {
  let opened = false;
  for (let node = element.parentElement; node; node = node.parentElement) {
    if (node instanceof HTMLDetailsElement && !node.open) {
      node.open = true;
      opened = true;
    }
  }
  return opened;
}

function targetOfHash(hash: string, doc: Document): Element | null {
  if (hash.length < 2) return null;
  let id: string;
  try {
    id = decodeURIComponent(hash.slice(1));
  } catch {
    return null;
  }
  return doc.getElementById(id);
}

// Returns the target only when it had to be opened, because only then does it still need scrolling.
export function revealHash(hash: string, doc: Document = document): Element | null {
  const target = targetOfHash(hash, doc);
  return target && openAncestors(target) ? target : null;
}

// For a link that may lead to another place on this page, before the browser moves there.
export function revealHref(href: string, doc: Document = document): void {
  const hashAt = href.indexOf("#");
  if (hashAt >= 0) revealHash(href.slice(hashAt), doc);
}

// An equation and the code lines linked to it, so a highlight can be seen and a count stays right.
export function revealEquation(slug: string, doc: Document = document): void {
  doc
    .querySelectorAll(`[data-eq="${slug}"], [data-eqs~="${slug}"]`)
    .forEach((element) => openAncestors(element));
}

// Printing shows every folded section. Returns what it opened, so it can be closed again afterwards.
export function openAllForPrint(doc: Document = document): HTMLDetailsElement[] {
  const closed = [...doc.querySelectorAll<HTMLDetailsElement>(COLLAPSIBLE_SELECTOR)].filter(
    (section) => !section.open,
  );
  closed.forEach((section) => {
    section.open = true;
  });
  return closed;
}

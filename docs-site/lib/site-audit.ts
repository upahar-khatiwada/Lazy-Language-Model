// Checks on the built pages in out/, with no browser. Each page is parsed once and its problems are
// plain strings, so tests can feed in small fixtures and see each rule fail. This does not replace a
// real accessibility tool: it cannot see colour, layout or anything rendered only by JavaScript.

const FOCUSABLE = "a[href], button, input, select, textarea, summary, [tabindex]";

export type LinkRef = { href: string; text: string };

const textOf = (element: Element | null) => element?.textContent?.trim() ?? "";

function accessibleName(element: Element, doc: Document): string {
  const label = element.getAttribute("aria-label")?.trim();
  if (label) return label;
  const labelledBy = element.getAttribute("aria-labelledby");
  if (labelledBy) {
    return labelledBy
      .split(/\s+/)
      .map((id) => textOf(doc.getElementById(id)))
      .join(" ")
      .trim();
  }
  return textOf(element) || element.getAttribute("title")?.trim() || "";
}

function hasFormLabel(field: Element, doc: Document): boolean {
  if (field.getAttribute("aria-label")?.trim()) return true;
  if (field.getAttribute("aria-labelledby")) return accessibleName(field, doc) !== "";
  if (field.closest("label")) return true;
  const id = field.getAttribute("id");
  return (
    id !== null && [...doc.querySelectorAll("label")].some((l) => l.getAttribute("for") === id)
  );
}

export function auditMetadata(doc: Document): string[] {
  const problems: string[] = [];
  const meta = (selector: string) => doc.querySelector(selector)?.getAttribute("content")?.trim();
  if (!textOf(doc.querySelector("title"))) problems.push("no <title>");
  if (!meta('meta[name="description"]')) problems.push("no meta description");
  if (!meta('meta[property="og:title"]')) problems.push("no og:title");
  if (!meta('meta[property="og:description"]')) problems.push("no og:description");
  if (!meta('meta[name="twitter:card"]')) problems.push("no twitter:card");
  if (!doc.querySelector('link[rel="icon"]')) problems.push("no favicon link");
  if (!doc.documentElement.getAttribute("lang")) problems.push("no lang on <html>");
  return problems;
}

// Reader pages only. The hidden /design style guide has no site chrome, so it has no skip link.
export function auditSkipLink(doc: Document): string[] {
  const id = doc.querySelector("main")?.getAttribute("id");
  if (!id) return ["<main> has no id for the skip link"];
  const first = doc.querySelector(FOCUSABLE);
  return first?.getAttribute("href") === `#${id}`
    ? []
    : ["the first focusable element is not a skip link to <main>"];
}

export function auditAccessibility(doc: Document): string[] {
  const problems: string[] = [];

  // Landmarks and the way in.
  if (doc.querySelectorAll("main").length !== 1) problems.push("page needs exactly one <main>");

  // Heading order.
  const headings = [...doc.querySelectorAll("h1, h2, h3, h4, h5, h6")];
  if (doc.querySelectorAll("h1").length !== 1) {
    problems.push(`page needs exactly one h1, found ${doc.querySelectorAll("h1").length}`);
  }
  let level = 0;
  for (const heading of headings) {
    const next = Number(heading.tagName[1]);
    if (level > 0 && next > level + 1) {
      problems.push(`heading skips a level: h${level} to h${next} "${textOf(heading)}"`);
    }
    if (!textOf(heading)) problems.push(`empty h${next}`);
    level = next;
  }

  // Ids and what points at them.
  const seen = new Set<string>();
  for (const element of doc.querySelectorAll("[id]")) {
    const id = element.getAttribute("id")!;
    if (seen.has(id)) problems.push(`duplicate id "${id}"`);
    seen.add(id);
  }
  for (const attribute of ["aria-labelledby", "aria-describedby", "aria-controls"]) {
    for (const element of doc.querySelectorAll(`[${attribute}]`)) {
      for (const id of element.getAttribute(attribute)!.split(/\s+/)) {
        if (!seen.has(id)) problems.push(`${attribute} points at missing id "${id}"`);
      }
    }
  }

  // Controls have names.
  for (const element of doc.querySelectorAll("a[href], button, summary")) {
    if (!accessibleName(element, doc)) {
      problems.push(`<${element.tagName.toLowerCase()}> has no accessible name`);
    }
  }
  for (const field of doc.querySelectorAll("input, select, textarea")) {
    if (field.getAttribute("type") !== "hidden" && !hasFormLabel(field, doc)) {
      problems.push(`<${field.tagName.toLowerCase()}> has no label`);
    }
  }
  for (const element of doc.querySelectorAll("[tabindex]")) {
    if (Number(element.getAttribute("tabindex")) > 0) problems.push("positive tabindex");
  }
  for (const element of doc.querySelectorAll('[aria-hidden="true"]')) {
    if (element.matches(FOCUSABLE) || element.querySelector(FOCUSABLE)) {
      problems.push("focusable element inside aria-hidden");
    }
  }

  // Scrollable and structural pieces.
  for (const summary of doc.querySelectorAll("summary")) {
    if (summary.parentElement?.firstElementChild !== summary) {
      problems.push("<summary> is not the first child of its <details>");
    }
  }

  // Pictures and equations have text alternatives.
  for (const svg of doc.querySelectorAll("svg")) {
    const named = svg.getAttribute("role") === "img" && svg.getAttribute("aria-label")?.trim();
    // Everything inside role=img is one picture to assistive technology, including a nested <svg>.
    const covered =
      svg.closest('[aria-hidden="true"]') || svg.parentElement?.closest("svg[role=img]");
    if (!named && !covered) {
      problems.push("<svg> has neither role=img with a label nor aria-hidden");
    }
  }
  for (const image of doc.querySelectorAll("img")) {
    if (image.getAttribute("alt") === null) problems.push("<img> has no alt");
  }
  if (doc.querySelector(".katex-error")) problems.push("KaTeX error node");
  for (const math of doc.querySelectorAll(".katex")) {
    if (!math.querySelector("math")) problems.push("KaTeX output has no MathML");
  }
  return problems;
}

export function collectLinks(doc: Document): LinkRef[] {
  return [...doc.querySelectorAll("a[href]")].map((link) => ({
    href: link.getAttribute("href")!,
    text: textOf(link).slice(0, 60),
  }));
}

export const collectIds = (doc: Document) =>
  new Set([...doc.querySelectorAll("[id]")].map((element) => element.getAttribute("id")!));

export type ParsedPermalink = {
  sha: string;
  file: string;
  start: number | null;
  end: number | null;
};

// https://github.com/<owner>/<repo>/blob/<sha>/<file>#L3-L9
export function parsePermalink(href: string, baseUrl: string): ParsedPermalink | null {
  if (!href.startsWith(`${baseUrl}/blob/`)) return null;
  const [path, hash] = href.slice(`${baseUrl}/blob/`.length).split("#");
  const slash = path!.indexOf("/");
  const lines = hash?.match(/^L(\d+)(?:-L(\d+))?$/);
  return {
    sha: path!.slice(0, slash),
    file: path!.slice(slash + 1),
    start: lines ? Number(lines[1]) : null,
    end: lines ? Number(lines[2] ?? lines[1]) : null,
  };
}

export function auditPermalink(
  link: ParsedPermalink,
  sha: string,
  lineCount: (file: string) => number | null,
): string | null {
  if (link.sha !== sha) return `permalink is pinned to ${link.sha}, not ${sha}: ${link.file}`;
  const count = lineCount(link.file);
  if (count === null) return `permalink file does not exist: ${link.file}`;
  if (link.end !== null && (link.start! < 1 || link.start! > link.end || link.end > count)) {
    return `permalink lines L${link.start}-L${link.end} are outside ${link.file} (${count} lines)`;
  }
  return null;
}

// /a/b/#x, relative to nothing because the export writes root-relative links. Returns the page's
// folder path and the fragment, or null for a link that leaves the site.
export function resolveInternal(
  href: string,
  fromPath: string,
  basePath = "",
): { path: string; hash: string; outsideBase?: boolean } | null {
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("//")) return null;
  const [beforeHash, hash = ""] = href.split("#");
  const withoutQuery = beforeHash!.split("?")[0]!;
  if (withoutQuery === "") return { path: fromPath, hash };
  const absolute = withoutQuery.startsWith("/")
    ? withoutQuery
    : new URL(withoutQuery, `http://x${fromPath}`).pathname;
  if (!basePath) return { path: absolute, hash };
  // Under a base path, a root-relative link that lacks it leaves the site.
  if (absolute !== basePath && !absolute.startsWith(`${basePath}/`)) {
    return { path: absolute, hash, outsideBase: true };
  }
  return { path: absolute.slice(basePath.length) || "/", hash };
}

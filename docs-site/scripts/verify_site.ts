// Checks the built site in out/: internal links and anchors, GitHub permalinks, metadata and the
// static accessibility rules in lib/site-audit.ts. Run it after `bun run build`.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { Window } from "happy-dom";
import { loadRepoMeta } from "../lib/repo";
import {
  auditAccessibility,
  auditMetadata,
  auditPermalink,
  auditSkipLink,
  collectIds,
  collectLinks,
  parsePermalink,
  resolveInternal,
} from "../lib/site-audit";

const OUT = join(process.cwd(), "out");
const REPO_ROOT = join(process.cwd(), "..");
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

if (!existsSync(OUT)) {
  console.error("FAILED: out/ does not exist. Run `bun run build` first.");
  process.exit(1);
}

function htmlFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === "_next" ? [] : htmlFiles(path);
    return name.endsWith(".html") ? [path] : [];
  });
}

const routeOf = (file: string) => {
  const rel = relative(OUT, file).split(sep).join("/");
  return rel === "index.html"
    ? "/"
    : rel.endsWith("/index.html")
      ? `/${rel.slice(0, -10)}`
      : `/${rel}`;
};

const window = new Window();
const parser = new window.DOMParser();

// The inline scripts hold the same page again as data, and only slow the parse down.
const parse = (html: string) =>
  parser.parseFromString(
    html
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/g, ""),
    "text/html",
  ) as unknown as Document;

const files = htmlFiles(OUT);
const pages = new Map<string, ReturnType<typeof read>>();

function read(file: string) {
  const doc = parse(readFileSync(file, "utf-8"));
  const robots = doc.querySelector('meta[name="robots"]')?.getAttribute("content") ?? "";
  return {
    ids: collectIds(doc),
    links: collectLinks(doc),
    title: doc.querySelector("title")?.textContent?.trim() ?? "",
    hidden: robots.includes("noindex"),
    metadata: auditMetadata(doc),
    skipLink: auditSkipLink(doc),
    accessibility: auditAccessibility(doc),
  };
}

for (const file of files) pages.set(routeOf(file), read(file));

const lineCounts = new Map<string, number | null>();
function lineCount(file: string): number | null {
  if (!lineCounts.has(file)) {
    const path = join(REPO_ROOT, file);
    lineCounts.set(file, existsSync(path) ? readFileSync(path, "utf-8").split("\n").length : null);
  }
  return lineCounts.get(file)!;
}

const repo = loadRepoMeta();
const problems: string[] = [];
let internal = 0;
let anchors = 0;
const permalinks = new Set<string>();

function pageExists(path: string): boolean {
  if (pages.has(path)) return true;
  // A file the export writes that is not a page, such as /math-code/x.json.
  return existsSync(join(OUT, ...path.split("/").filter(Boolean)));
}

const titles = new Map<string, string>();
for (const [route, page] of pages) {
  const report = (text: string) => problems.push(`${route}: ${text}`);

  // The hidden style guides and the 404 page are not meant to be found or shared.
  if (!page.hidden) [...page.metadata, ...page.skipLink].forEach(report);
  page.accessibility.forEach(report);

  if (!page.hidden) {
    const other = titles.get(page.title);
    if (other) report(`same title as ${other}: "${page.title}"`);
    titles.set(page.title, route);
  }

  for (const { href, text } of page.links) {
    const permalink = parsePermalink(href, repo.githubBaseUrl);
    if (permalink) {
      if (!permalinks.has(href)) {
        permalinks.add(href);
        const problem = auditPermalink(permalink, repo.sha, lineCount);
        if (problem) report(problem);
      }
      continue;
    }
    const target = resolveInternal(href, route.endsWith("/") ? route : `${route}/`, BASE_PATH);
    if (!target) continue;
    internal++;
    if (target.outsideBase) {
      report(`link does not carry the base path ${BASE_PATH}: ${href} ("${text}")`);
      continue;
    }
    if (!pageExists(target.path)) {
      report(`broken link to ${href} ("${text}")`);
      continue;
    }
    if (target.hash) {
      anchors++;
      const destination = pages.get(target.path);
      if (destination && !destination.ids.has(decodeURIComponent(target.hash))) {
        report(`broken anchor ${href} ("${text}")`);
      }
    }
  }
}

for (const problem of problems) console.error(`problem: ${problem}`);
const summary = `${pages.size} pages, ${internal} internal links, ${anchors} anchors, ${permalinks.size} GitHub permalinks`;
if (problems.length > 0) {
  console.error(`FAILED: ${problems.length} problem(s) in ${summary}.`);
  process.exit(1);
}
console.log(`OK: ${summary}. No problems.`);
await window.happyDOM.close();

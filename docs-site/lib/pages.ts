import { NAV } from "@/data/nav";
import { listContentSlugs } from "./content";
import { pageHref } from "./nav";

// A page in the navigation is "written" once its content file exists.

export function navSlugs(): string[] {
  return NAV.flatMap((group) => group.pages.map((page) => page.slug));
}

export function pageTitle(slug: string): string {
  const page = NAV.flatMap((group) => group.pages).find((candidate) => candidate.slug === slug);
  if (!page) throw new Error(`Page is not in the navigation: ${slug}`);
  return page.title;
}

// The route when the page is written, null otherwise. Unknown slugs fail the build.
export function writtenHref(slug: string): string | null {
  pageTitle(slug);
  return listContentSlugs().includes(slug) ? pageHref(slug) : null;
}

export function firstWrittenInGroup(title: string): { slug: string; title: string } | null {
  const group = NAV.find((candidate) => candidate.title === title);
  if (!group) throw new Error(`Navigation has no group named ${title}`);
  const written = new Set(listContentSlugs());
  return group.pages.find((page) => written.has(page.slug)) ?? null;
}

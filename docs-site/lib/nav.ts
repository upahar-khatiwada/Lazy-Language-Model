import type { NavGroup, NavPage } from "@/data/nav";

// No file system access here: client components use these types and helpers.

export type ResolvedPage = NavPage & { href: string; enabled: boolean };
export type ResolvedGroup = { title: string; pages: ResolvedPage[] };
export type Crumb = { title: string; href?: string };

export function pageHref(slug: string): string {
  return `/${slug}/`;
}

export function resolveNav(tree: NavGroup[], available: Iterable<string>): ResolvedGroup[] {
  const slugs = new Set(available);
  return tree.map((group) => ({
    title: group.title,
    pages: group.pages.map((page) => ({
      ...page,
      href: pageHref(page.slug),
      enabled: slugs.has(page.slug),
    })),
  }));
}

export function flattenNav(tree: ResolvedGroup[]): ResolvedPage[] {
  return tree.flatMap((group) => group.pages);
}

// Groups have no page of their own, so the group crumb is plain text.
export function breadcrumbs(tree: ResolvedGroup[], slug: string): Crumb[] {
  for (const group of tree) {
    const page = group.pages.find((candidate) => candidate.slug === slug);
    if (page) return [{ title: "Docs", href: "/" }, { title: group.title }, { title: page.title }];
  }
  throw new Error(`Page is not in the navigation: ${slug}`);
}

// Disabled pages have no route, so previous and next skip them.
export function prevNext(
  tree: ResolvedGroup[],
  slug: string,
): { prev?: ResolvedPage; next?: ResolvedPage } {
  const enabled = flattenNav(tree).filter((page) => page.enabled);
  const index = enabled.findIndex((page) => page.slug === slug);
  if (index === -1) throw new Error(`Page is not an enabled navigation entry: ${slug}`);
  return { prev: enabled[index - 1], next: enabled[index + 1] };
}

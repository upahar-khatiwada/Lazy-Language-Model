import type { ReactNode } from "react";
import { DocsChrome } from "@/components/DocsChrome";
import { SiteFooter } from "@/components/DocMain";
import { NAV } from "@/data/nav";
import { listContentSlugs } from "@/lib/content";
import { resolveNav } from "@/lib/nav";
import { loadRepoMeta } from "@/lib/repo";

export default function DocsLayout({ children }: { children: ReactNode }) {
  const repo = loadRepoMeta();
  return (
    <DocsChrome nav={resolveNav(NAV, listContentSlugs())} githubUrl={repo.githubBaseUrl}>
      {children}
      <SiteFooter sha={repo.sha} shortSha={repo.shortSha} baseUrl={repo.githubBaseUrl} />
    </DocsChrome>
  );
}

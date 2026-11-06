import { Suspense, type ReactNode } from "react";
import { DocsChrome } from "@/components/DocsChrome";
import { SiteFooter } from "@/components/DocMain";
import { MathCodeNavigator } from "@/components/MathCodeNavigator";
import { SectionReveal } from "@/components/SectionReveal";
import { NAV } from "@/data/nav";
import { listContentSlugs } from "@/lib/content";
import { resolveNav } from "@/lib/nav";
import { loadRepoMeta } from "@/lib/repo";

export default function DocsLayout({ children }: { children: ReactNode }) {
  const repo = loadRepoMeta();
  return (
    <DocsChrome nav={resolveNav(NAV, listContentSlugs())} githubUrl={repo.githubBaseUrl}>
      {children}
      {/* It reads ?eq= from the address, which a static page only knows in the browser. */}
      <Suspense fallback={null}>
        <MathCodeNavigator />
      </Suspense>
      <SectionReveal />
      <SiteFooter sha={repo.sha} shortSha={repo.shortSha} baseUrl={repo.githubBaseUrl} />
    </DocsChrome>
  );
}

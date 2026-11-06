import type { Metadata } from "next";
import Link from "next/link";
import { DocHeader, DocMain, SiteFooter } from "@/components/DocMain";
import { DocsChrome } from "@/components/DocsChrome";
import { NAV } from "@/data/nav";
import { listContentSlugs } from "@/lib/content";
import { resolveNav } from "@/lib/nav";
import { loadRepoMeta } from "@/lib/repo";

export const metadata: Metadata = {
  title: "Page not found",
  description: "This address has no page in the LLM Developer Docs.",
  robots: { index: false },
};

// The export writes this to out/404.html, which GitHub Pages serves for an unknown address. It keeps
// the sidebar and search, so a reader can still find the page.
export default function NotFound() {
  const repo = loadRepoMeta();
  return (
    <DocsChrome nav={resolveNav(NAV, listContentSlugs())} githubUrl={repo.githubBaseUrl}>
      <DocMain wide>
        <DocHeader title="Page not found" description="This address has no page." />
        <div className="prose">
          <p>
            Use the search or the sidebar, or go to the{" "}
            <Link prefetch={false} href="/">
              home page
            </Link>
            .
          </p>
        </div>
      </DocMain>
      <SiteFooter sha={repo.sha} shortSha={repo.shortSha} baseUrl={repo.githubBaseUrl} />
    </DocsChrome>
  );
}

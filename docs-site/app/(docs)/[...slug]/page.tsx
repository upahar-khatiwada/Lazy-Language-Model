import type { Metadata } from "next";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { Callout } from "@/components/Callout";
import { CodeFileBody, explorerNeighbours, SourceListing } from "@/components/CodeExplorer";
import { DocHeader, DocMain, PageNotice } from "@/components/DocMain";
import { mdxComponents } from "@/components/mdx";
import { OnThisPage } from "@/components/OnThisPage";
import { PrevNext } from "@/components/PrevNext";
import { SourceTargetHighlight } from "@/components/SourceTargetHighlight";
import { NAV } from "@/data/nav";
import {
  explorerFiles,
  fileSlug,
  filePurpose,
  findExplorerFile,
  isTestFile,
  type ExplorerFile,
} from "@/lib/code-explorer";
import { listContentSlugs, loadPage, readPageMeta } from "@/lib/content";
import { breadcrumbs, prevNext, resolveNav } from "@/lib/nav";
import { loadRepoMeta } from "@/lib/repo";

type Params = { params: Promise<{ slug: string[] }> };

export const dynamicParams = false;

// Files of the Code Explorer share this route, so /code/<file> never competes with /code/<page>.
function explorerFileOf(slug: string[]): ExplorerFile | undefined {
  return slug.length === 2 && slug[0] === "code" ? findExplorerFile(slug[1]!) : undefined;
}

export function generateStaticParams() {
  return [
    ...listContentSlugs().map((slug) => ({ slug: slug.split("/") })),
    ...explorerFiles().map(({ path }) => ({ slug: ["code", fileSlug(path)] })),
  ];
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slug } = await params;
  const file = explorerFileOf(slug);
  if (file) return { title: file.path, description: filePurpose(file) };
  const { title, description } = readPageMeta(slug.join("/"));
  return { title, description };
}

function CodeFilePage({ file }: { file: ExplorerFile }) {
  const repo = loadRepoMeta();
  return (
    <DocMain wide>
      <Breadcrumbs
        crumbs={[
          { title: "Docs", href: "/" },
          { title: "Code" },
          { title: "Code Explorer", href: "/code/" },
          { title: file.path },
        ]}
      />
      <DocHeader
        title={file.path}
        description={filePurpose(file)}
        sourceFile={isTestFile(file.path) ? "Test file" : undefined}
      />
      <CodeFileBody file={file}>
        <SourceListing
          file={file}
          permalink={`${repo.githubBaseUrl}/blob/${repo.sha}/${file.path}`}
        />
      </CodeFileBody>
      <SourceTargetHighlight key={file.path} />
      <PrevNext {...explorerNeighbours(file)} />
    </DocMain>
  );
}

export default async function DocPage({ params }: Params) {
  const segments = (await params).slug;
  const file = explorerFileOf(segments);
  if (file) return <CodeFilePage file={file} />;

  const slug = segments.join("/");
  const { meta, headings, Content } = await loadPage(slug);
  const nav = resolveNav(NAV, listContentSlugs());

  return (
    <>
      <DocMain>
        <Breadcrumbs crumbs={breadcrumbs(nav, slug)} />
        <DocHeader title={meta.title} description={meta.description} sourceFile={meta.sourceFile} />
        {meta.status === "placeholder" ? (
          <PageNotice>
            <Callout kind="warning">
              <p>Placeholder page. It only tests the layout and says nothing about the model.</p>
            </Callout>
          </PageNotice>
        ) : null}
        <div className="prose">
          <Content components={mdxComponents} />
        </div>
        <PrevNext {...prevNext(nav, slug)} />
      </DocMain>
      <OnThisPage headings={headings} />
    </>
  );
}

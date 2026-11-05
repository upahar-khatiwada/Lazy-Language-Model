import type { Metadata } from "next";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { Callout } from "@/components/Callout";
import { DocHeader, DocMain, PageNotice } from "@/components/DocMain";
import { mdxComponents } from "@/components/mdx";
import { OnThisPage } from "@/components/OnThisPage";
import { PrevNext } from "@/components/PrevNext";
import { NAV } from "@/data/nav";
import { listContentSlugs, loadPage, readPageMeta } from "@/lib/content";
import { breadcrumbs, prevNext, resolveNav } from "@/lib/nav";

type Params = { params: Promise<{ slug: string[] }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return listContentSlugs().map((slug) => ({ slug: slug.split("/") }));
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slug } = await params;
  const { title, description } = readPageMeta(slug.join("/"));
  return { title, description };
}

export default async function DocPage({ params }: Params) {
  const slug = (await params).slug.join("/");
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

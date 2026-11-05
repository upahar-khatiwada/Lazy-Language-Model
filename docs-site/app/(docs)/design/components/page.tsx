import type { Metadata } from "next";
import { DocHeader, DocMain } from "@/components/DocMain";
import { mdxComponents } from "@/components/mdx";
import { OnThisPage } from "@/components/OnThisPage";
import { loadPage } from "@/lib/content";

const SAMPLE = "_samples/softmax-forward";

// Hidden, like /design: not in data/nav.ts and marked noindex.
export const metadata: Metadata = {
  title: "Component samples",
  robots: { index: false, follow: false },
};

export default async function ComponentSamplesPage() {
  const { meta, headings, Content } = await loadPage(SAMPLE);
  return (
    <>
      <DocMain>
        <DocHeader title={meta.title} description={meta.description} />
        <div className="prose">
          <Content components={mdxComponents} />
        </div>
      </DocMain>
      <OnThisPage headings={headings} />
    </>
  );
}

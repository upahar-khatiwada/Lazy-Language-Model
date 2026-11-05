import { readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { evaluate } from "@mdx-js/mdx";
import matter from "gray-matter";
import type { MDXContent } from "mdx/types";
import * as runtime from "react/jsx-runtime";
import remarkGfm from "remark-gfm";
import { z } from "zod";
import { remarkHeadings, type PageHeading } from "./remark-headings";

const CONTENT_DIR = join(process.cwd(), "content");

const FrontmatterSchema = z.strictObject({
  title: z.string().min(1),
  description: z.string().min(1),
  sourceFile: z.string().min(1).optional(),
  status: z.enum(["placeholder", "draft", "published"]),
});

export type PageMeta = z.infer<typeof FrontmatterSchema>;

export type LoadedPage = {
  meta: PageMeta;
  headings: PageHeading[];
  Content: MDXContent;
};

// Slugs use forward slashes on every platform: mathematics/tokenization.
// Folders starting with "_" hold samples that get no route of their own and stay out of the nav.
export function listContentSlugs(): string[] {
  return readdirSync(CONTENT_DIR, { recursive: true, encoding: "utf-8" })
    .filter((path) => path.endsWith(".mdx") && !path.split(/[\\/]/).some((p) => p.startsWith("_")))
    .map((path) =>
      relative(CONTENT_DIR, join(CONTENT_DIR, path))
        .split(sep)
        .join("/")
        .replace(/\.mdx$/, ""),
    )
    .sort();
}

function readSource(slug: string): { meta: PageMeta; body: string } {
  const raw = readFileSync(join(CONTENT_DIR, `${slug}.mdx`), "utf-8");
  const { data, content } = matter(raw);
  const result = FrontmatterSchema.safeParse(data);
  if (!result.success) {
    throw new Error(`Invalid frontmatter in ${slug}.mdx: ${z.prettifyError(result.error)}`);
  }
  return { meta: result.data, body: content };
}

export function readPageMeta(slug: string): PageMeta {
  return readSource(slug).meta;
}

// Pages are compiled at build time only, because the site is a static export.
// remark-math is left out on purpose: equations come from structured data through MathBlock.
export async function loadPage(slug: string): Promise<LoadedPage> {
  const { meta, body } = readSource(slug);
  const headings: PageHeading[] = [];
  const { default: Content } = await evaluate(body, {
    ...runtime,
    remarkPlugins: [remarkGfm, remarkHeadings(headings)],
  });
  return { meta, headings, Content };
}

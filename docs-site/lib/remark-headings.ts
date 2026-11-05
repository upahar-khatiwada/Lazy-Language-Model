import GithubSlugger from "github-slugger";
import type { Root } from "mdast";
import { toString } from "mdast-util-to-string";
import { visit } from "unist-util-visit";

export type PageHeading = { id: string; text: string; depth: 2 | 3 };

// Gives each heading an id and reports the h2 and h3 headings for the on-this-page rail.
export function remarkHeadings(collected: PageHeading[]) {
  return () => (tree: Root) => {
    const slugger = new GithubSlugger();
    visit(tree, "heading", (node) => {
      const text = toString(node);
      const id = slugger.slug(text);
      node.data = { ...node.data, hProperties: { id } };
      if (node.depth === 2 || node.depth === 3) {
        collected.push({ id, text, depth: node.depth });
      }
    });
  };
}

import type { Root, RootContent } from "mdast";
import { toString } from "mdast-util-to-string";

// Long pages stay short by folding their longest parts. The MDX source is not touched, so the page
// protocol's nine steps, the search index and verify:content all keep reading what the author wrote.
// Only the page's first section that has something to fold starts open. <details> keeps the folded
// content in the page, and lib/reveal.ts opens it for anchors, ?eq= links and printing.

// The steps of the page protocol that hold the longest content.
export const FOLDED_STEPS = new Set([
  "Tiny numerical example",
  "Actual implementation",
  "Code and math connection",
]);
// A backward section is long throughout, so only its introduction stays open.
const OPEN_BACKWARD_STEPS = new Set(["What is it?", "Why does the model need it?"]);
const BACKWARD_SECTION = /backward/i;

type Attribute = { type: "mdxJsxAttribute"; name: string; value: string | null };
type JsxNode = {
  type: "mdxJsxFlowElement";
  name: string | null;
  attributes: Attribute[];
  children: RootContent[];
};

const isJsx = (node: RootContent, name: string): boolean =>
  (node as unknown as JsxNode).type === "mdxJsxFlowElement" &&
  (node as unknown as JsxNode).name === name;

const attribute = (name: string, value: string | null): Attribute => ({
  type: "mdxJsxAttribute",
  name,
  value,
});

function collapsible(attributes: Attribute[], children: RootContent[]): RootContent {
  const node: JsxNode = { type: "mdxJsxFlowElement", name: "Collapsible", attributes, children };
  return node as unknown as RootContent;
}

const symbolOf = (node: RootContent): string =>
  (node as unknown as JsxNode).attributes.find(({ name }) => name === "symbol")?.value ?? "";

export function collapseChildren(children: RootContent[]): RootContent[] {
  // A page without the protocol, such as a walkthrough, reads from top to bottom. It stays as written.
  if (!children.some((node) => isJsx(node, "ConceptStep"))) return children;

  const out: RootContent[] = [];
  let section = "";
  let sectionIndex = 0;
  let openSection: number | null = null;

  const openAttribute = (): Attribute[] => {
    openSection ??= sectionIndex;
    return openSection === sectionIndex ? [attribute("open", null)] : [];
  };

  for (let i = 0; i < children.length; i++) {
    const node = children[i]!;
    if (node.type === "heading") {
      if (node.depth === 2) {
        section = toString(node);
        sectionIndex++;
      }
      out.push(node);
      continue;
    }

    if (isJsx(node, "ConceptStep")) {
      const step = toString(node);
      const folded =
        FOLDED_STEPS.has(step) ||
        (BACKWARD_SECTION.test(section) && !OPEN_BACKWARD_STEPS.has(step));
      if (folded) {
        const body: RootContent[] = [];
        while (
          i + 1 < children.length &&
          children[i + 1]!.type !== "heading" &&
          !isJsx(children[i + 1]!, "ConceptStep")
        ) {
          body.push(children[++i]!);
        }
        // A step with nothing under it has nothing to fold.
        if (body.length > 0) {
          out.push(collapsible([attribute("step", step), ...openAttribute()], body));
          continue;
        }
      }
      out.push(node);
      continue;
    }

    // A full listing outside the protocol's steps.
    if (isJsx(node, "CodeReference")) {
      const title = `Code listing: ${symbolOf(node)}`;
      out.push(collapsible([attribute("title", title), ...openAttribute()], [node]));
      continue;
    }

    out.push(node);
  }
  return out;
}

export function remarkCollapse() {
  return (tree: Root) => {
    tree.children = collapseChildren(tree.children);
  };
}

import type { MDXComponents } from "mdx/types";
import { Badge } from "./Badge";
import { Callout } from "./Callout";
import { CodeBlock } from "./CodeBlock";
import { DataTable } from "./DataTable";
import { MathBlock } from "./MathBlock";

export const mdxComponents: MDXComponents = {
  table: (props) => <DataTable {...props} />,
  Badge,
  Callout,
  CodeBlock,
  MathBlock,
};

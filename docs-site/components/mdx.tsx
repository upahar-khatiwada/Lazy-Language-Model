import type { MDXComponents } from "mdx/types";
import { Badge } from "./Badge";
import { Callout } from "./Callout";
import { CodeReference } from "./CodeReference";
import { ConceptCard } from "./ConceptCard";
import { DataTable } from "./DataTable";
import { MathBlock } from "./MathBlock";
import { MathCodeLink } from "./MathCodeLink";
import { NumericExample } from "./NumericExample";
import { PipelineDiagram } from "./PipelineDiagram";
import { SourceBadge } from "./SourceBadge";
import { Symbols } from "./Symbols";
import { TensorShape } from "./TensorShape";
import { Trace } from "./Trace";

// Only id-based components are offered for equations and code, so a page cannot inline either.
export const mdxComponents: MDXComponents = {
  table: (props) => <DataTable {...props} />,
  Badge,
  Callout,
  CodeReference,
  ConceptCard,
  MathBlock,
  MathCodeLink,
  NumericExample,
  PipelineDiagram,
  SourceBadge,
  Symbols,
  TensorShape,
  Trace,
};

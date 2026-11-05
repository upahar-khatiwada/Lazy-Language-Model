import type { MDXComponents } from "mdx/types";
import { ParameterCount, ParameterTable, RepoMap } from "./ArchitectureTables";
import { Badge } from "./Badge";
import { Callout } from "./Callout";
import { CodeReference } from "./CodeReference";
import { CodeLine, CodeStep } from "./CodeStep";
import { ConceptCard } from "./ConceptCard";
import { ConfigValue } from "./ConfigValue";
import { DataTable } from "./DataTable";
import { MathBlock } from "./MathBlock";
import { MathCodeLink } from "./MathCodeLink";
import { MathReference } from "./MathReference";
import { NumericExample } from "./NumericExample";
import { PageLink } from "./PageLink";
import { FullPipelineDiagram, PipelineDiagram } from "./PipelineDiagram";
import { SourceBadge } from "./SourceBadge";
import { Symbols } from "./Symbols";
import { TensorShape } from "./TensorShape";
import { Trace } from "./Trace";

// Only id-based components are offered for equations and code, so a page cannot inline either.
export const mdxComponents: MDXComponents = {
  table: (props) => <DataTable {...props} />,
  Badge,
  Callout,
  CodeLine,
  CodeReference,
  CodeStep,
  ConceptCard,
  ConfigValue,
  FullPipelineDiagram,
  MathBlock,
  MathCodeLink,
  MathReference,
  NumericExample,
  PageLink,
  ParameterCount,
  ParameterTable,
  PipelineDiagram,
  RepoMap,
  SourceBadge,
  Symbols,
  TensorShape,
  Trace,
};

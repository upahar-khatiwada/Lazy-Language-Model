import type { MDXComponents } from "mdx/types";
import { InitialValues, ParameterCount, ParameterTable, RepoMap } from "./ArchitectureTables";
import { Badge } from "./Badge";
import { BpeExample } from "./BpeExample";
import { Callout } from "./Callout";
import { CodeReference } from "./CodeReference";
import { CodeLine, CodeStep } from "./CodeStep";
import { ConceptCard, ConceptWhy } from "./ConceptCard";
import { ConceptStep } from "./ConceptStep";
import { ConfigValue } from "./ConfigValue";
import {
  BatchExample,
  CorpusExample,
  SplitExample,
  WindowCountExample,
  WindowExample,
} from "./DataExamples";
import { DataTable } from "./DataTable";
import { EquationCode } from "./EquationCode";
import {
  LayerNormBackwardExample,
  LayerNormDerivationExample,
  LayerNormLabExample,
} from "./LayerNormExamples";
import { MathBlock } from "./MathBlock";
import { MathCodeLink } from "./MathCodeLink";
import { MathReference } from "./MathReference";
import { MatrixExample } from "./MatrixExample";
import { NotationTable } from "./NotationTable";
import { NumericExample } from "./NumericExample";
import {
  CrossEntropyBackwardExample,
  CrossEntropyExample,
  InitExample,
  LinearBackwardExample,
  LinearPlaygroundExample,
  PerplexityExample,
  ReluExample,
  SoftmaxBackwardExample,
  SoftmaxLabExample,
  SoftmaxShiftTable,
  XavierExample,
} from "./OpsExamples";
import { PageLink } from "./PageLink";
import { ParameterDerivation, ParameterExample } from "./ParameterDerivation";
import { FullPipelineDiagram, PipelineDiagram } from "./PipelineDiagram";
import { SourceBadge } from "./SourceBadge";
import { SymbolCoverage } from "./SymbolCoverage";
import { Symbols } from "./Symbols";
import { TensorShape } from "./TensorShape";
import {
  EncodeExample,
  MergeExample,
  VocabularyExample,
  WordCountExample,
} from "./TokenizerExamples";
import { Trace } from "./Trace";

// Only id-based components are offered for equations and code, so a page cannot inline either.
export const mdxComponents: MDXComponents = {
  table: (props) => <DataTable {...props} />,
  Badge,
  BatchExample,
  BpeExample,
  Callout,
  CodeLine,
  CodeReference,
  CodeStep,
  ConceptCard,
  ConceptStep,
  ConceptWhy,
  ConfigValue,
  CorpusExample,
  CrossEntropyBackwardExample,
  CrossEntropyExample,
  EncodeExample,
  EquationCode,
  FullPipelineDiagram,
  InitExample,
  InitialValues,
  LayerNormBackwardExample,
  LayerNormDerivationExample,
  LayerNormLabExample,
  LinearBackwardExample,
  LinearPlaygroundExample,
  MathBlock,
  MathCodeLink,
  MathReference,
  MatrixExample,
  MergeExample,
  NotationTable,
  NumericExample,
  PageLink,
  ParameterCount,
  ParameterDerivation,
  ParameterExample,
  ParameterTable,
  PerplexityExample,
  PipelineDiagram,
  ReluExample,
  RepoMap,
  SoftmaxBackwardExample,
  SoftmaxLabExample,
  SoftmaxShiftTable,
  SourceBadge,
  SplitExample,
  SymbolCoverage,
  Symbols,
  TensorShape,
  Trace,
  VocabularyExample,
  WindowCountExample,
  WindowExample,
  WordCountExample,
  XavierExample,
};

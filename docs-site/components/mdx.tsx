import type { MDXComponents } from "mdx/types";
import { ActualEmbeddingsPanel, ActualGenerationPanel, ActualTokenizerPanel } from "./ActualPanels";
import { InitialValues, ParameterCount, ParameterTable, RepoMap } from "./ArchitectureTables";
import {
  ActualModelOutputSlot,
  AttentionBackwardExample,
  AttentionLabExample,
  AttentionMaskExample,
  AttentionMergeExample,
  AttentionOutputExample,
  AttentionProjectionExample,
  AttentionScaleExample,
  AttentionScoresExample,
  AttentionShapes,
  AttentionSoftmaxExample,
  AttentionSplitExample,
  AttentionWeightedExample,
} from "./AttentionExamples";
import { Badge } from "./Badge";
import {
  BlockBackwardExample,
  CacheTable,
  FfnBackwardExample,
  FfnForwardExample,
  ForwardExample,
  GradientFlowDiagram,
  LossAndGradsExample,
  ParameterKeys,
  ResidualStreamExample,
} from "./BlockExamples";
import { BpeExample } from "./BpeExample";
import { Callout } from "./Callout";
import {
  CheckpointExample,
  CheckpointFiles,
  EntryLine,
  ModelConfigFields,
  SymbolUsage,
  TestCoverage,
  TestTotals,
  UntestedSymbols,
} from "./CheckpointExamples";
import { CodeFileLink, CodeTree } from "./CodeExplorer";
import { Collapsible } from "./Collapsible";
import { CodeReference } from "./CodeReference";
import { CodeLine, CodeStep } from "./CodeStep";
import { ConceptCard, ConceptWhy } from "./ConceptCard";
import { ConceptStep } from "./ConceptStep";
import { ConfigValue } from "./ConfigValue";
import { AbsentTechniques, DiscrepancyPreamble, DiscrepancySection } from "./ContractPages";
import {
  BatchExample,
  CorpusExample,
  SplitExample,
  WindowCountExample,
  WindowExample,
} from "./DataExamples";
import { DataTable } from "./DataTable";
import {
  EmbeddingBackwardExample,
  EmbeddingLabExample,
  PositionExample,
  PositionUpdateExample,
  TokenLookupExample,
  UnusedPositionRows,
} from "./EmbeddingExamples";
import { EquationCode } from "./EquationCode";
import {
  ChoiceExample,
  GenerationStepperExample,
  GreedyExample,
  ProbabilityLabExample,
  SamplingDefaults,
  TemperatureExample,
  TopKExample,
  TopPExample,
} from "./GenerationExamples";
import {
  LayerNormBackwardExample,
  LayerNormDerivationExample,
  LayerNormLabExample,
} from "./LayerNormExamples";
import {
  CentralDifferenceExample,
  CheckParamsExample,
  RelativeErrorExample,
} from "./GradcheckExamples";
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
import {
  AdamCalculatorExample,
  AdamTwoStepExample,
  BatchAverageExample,
  ClipLabExample,
  DecayTable,
  LearningPipelineDiagram,
  LogLines,
  LogMeanings,
  ScheduleLabExample,
  StepCountExample,
  ValidationExample,
  WeightWalkthroughExample,
} from "./TrainingExamples";

// Only id-based components are offered for equations and code, so a page cannot inline either.
export const mdxComponents: MDXComponents = {
  table: (props) => <DataTable {...props} />,
  AbsentTechniques,
  ActualEmbeddingsPanel,
  ActualGenerationPanel,
  ActualModelOutputSlot,
  ActualTokenizerPanel,
  AdamCalculatorExample,
  AdamTwoStepExample,
  AttentionBackwardExample,
  AttentionLabExample,
  AttentionMaskExample,
  AttentionMergeExample,
  AttentionOutputExample,
  AttentionProjectionExample,
  AttentionScaleExample,
  AttentionScoresExample,
  AttentionShapes,
  AttentionSoftmaxExample,
  AttentionSplitExample,
  AttentionWeightedExample,
  Badge,
  BatchAverageExample,
  BatchExample,
  BlockBackwardExample,
  BpeExample,
  CacheTable,
  Callout,
  CentralDifferenceExample,
  CheckParamsExample,
  CheckpointExample,
  CheckpointFiles,
  ChoiceExample,
  ClipLabExample,
  Collapsible,
  CodeFileLink,
  CodeLine,
  CodeReference,
  CodeStep,
  CodeTree,
  ConceptCard,
  ConceptStep,
  ConceptWhy,
  ConfigValue,
  CorpusExample,
  CrossEntropyBackwardExample,
  CrossEntropyExample,
  DecayTable,
  DiscrepancyPreamble,
  DiscrepancySection,
  EmbeddingBackwardExample,
  EmbeddingLabExample,
  EncodeExample,
  EntryLine,
  EquationCode,
  FfnBackwardExample,
  FfnForwardExample,
  ForwardExample,
  FullPipelineDiagram,
  GradientFlowDiagram,
  GenerationStepperExample,
  GreedyExample,
  InitExample,
  InitialValues,
  LayerNormBackwardExample,
  LayerNormDerivationExample,
  LayerNormLabExample,
  LearningPipelineDiagram,
  LinearBackwardExample,
  LinearPlaygroundExample,
  LogLines,
  LogMeanings,
  LossAndGradsExample,
  MathBlock,
  MathCodeLink,
  MathReference,
  MatrixExample,
  MergeExample,
  ModelConfigFields,
  NotationTable,
  NumericExample,
  PageLink,
  ParameterCount,
  ParameterDerivation,
  ParameterExample,
  ParameterKeys,
  ParameterTable,
  PerplexityExample,
  PipelineDiagram,
  PositionExample,
  PositionUpdateExample,
  ProbabilityLabExample,
  RelativeErrorExample,
  ReluExample,
  RepoMap,
  ResidualStreamExample,
  SamplingDefaults,
  ScheduleLabExample,
  SoftmaxBackwardExample,
  SoftmaxLabExample,
  SoftmaxShiftTable,
  SourceBadge,
  SplitExample,
  StepCountExample,
  SymbolCoverage,
  Symbols,
  SymbolUsage,
  TensorShape,
  TemperatureExample,
  TestCoverage,
  TestTotals,
  TokenLookupExample,
  TopKExample,
  TopPExample,
  Trace,
  UnusedPositionRows,
  UntestedSymbols,
  ValidationExample,
  VocabularyExample,
  WeightWalkthroughExample,
  WindowCountExample,
  WindowExample,
  WordCountExample,
  XavierExample,
};

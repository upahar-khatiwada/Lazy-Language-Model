import type { CodeTarget } from "@/lib/code-targets";
import type { DimensionName } from "@/lib/dimensions";

// The symbol table of the Notation page. `code` must appear in the lines `target` resolves to, and
// `config` must exist in data/generated/config.json; lib/site-verify.ts checks both. Values are not
// written here: they come from config.json through lib/dimensions.ts.

export type NotationRow = {
  dimension: DimensionName;
  tex: string;
  meaning: string;
  config: string | null;
  code: string;
  target: CodeTarget;
  note?: string;
};

const INIT = { file: "transformer.py", symbol: "transformer.init_params" };
const ATTENTION = { file: "transformer.py", symbol: "transformer.attention_forward" };
const SIZES = "d, F, V = cfg.emb_dim, cfg.ff_dim, cfg.vocab_size";

export const NOTATION: NotationRow[] = [
  {
    dimension: "T",
    tex: "T",
    meaning: "Tokens in one sequence",
    config: "seq_len",
    code: "T, d = Xn.shape",
    target: { ...ATTENTION, match: "T, d = Xn.shape" },
    note: "Inside the model, T is read from the array, not from config. Generation can use fewer than seq_len.",
  },
  {
    dimension: "Tmax",
    tex: "T_{\\max}",
    meaning: "Rows in the position table, the longest allowed sequence",
    config: "context_len",
    code: "cfg.context_len",
    target: { ...INIT, match: '"pos_emb": xavier_uniform(cfg.context_len, d, rng)' },
  },
  {
    dimension: "d",
    tex: "d",
    meaning: "Width of each token's vector",
    config: "emb_dim",
    code: "cfg.emb_dim",
    target: { ...INIT, match: SIZES },
  },
  {
    dimension: "H",
    tex: "H",
    meaning: "Attention heads",
    config: "num_heads",
    code: "cfg.num_heads",
    target: { ...ATTENTION, match: "H = cfg.num_heads" },
  },
  {
    dimension: "dh",
    tex: "d_h",
    meaning: "Width of one head",
    config: "head_dim",
    code: "dh = d // H",
    target: { ...ATTENTION, match: "dh = d // H" },
    note: "Nothing reads config.head_dim. The code computes the same value in place.",
  },
  {
    dimension: "F",
    tex: "F",
    meaning: "Width of the feed-forward hidden layer",
    config: "ff_dim",
    code: "cfg.ff_dim",
    target: { ...INIT, match: SIZES },
  },
  {
    dimension: "L",
    tex: "L",
    meaning: "Transformer blocks",
    config: "num_layers",
    code: "cfg.num_layers",
    target: { ...INIT, match: "for l in range(cfg.num_layers):" },
  },
  {
    dimension: "V",
    tex: "V",
    meaning: "Vocabulary size",
    config: null,
    code: "tok.vocab_size",
    target: {
      file: "train.py",
      symbol: "train.main",
      match: "cfg = ModelConfig(tok.vocab_size,",
    },
    note: "It depends on the corpus, so config.py has no value for it. Inside attention_forward the local name V is the value matrix, not this.",
  },
];

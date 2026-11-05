# Hyperparameters

Pinned commit: `c17aa75`. Source: `config.py`. Comments are quoted verbatim from the code, including their spelling. "Read by" lists every `config.<name>` reference in the repository, found by searching root modules and `tests/`. No test reads `config`.

## 1. Values

| Name | Value | Line | Comment in `config.py` | Read by |
|---|---|---:|---|---|
| `context_len` | `32` | 2 | suppose 5 tokens then theres 5x32 rows, the model can look 32 texts at once | `train.py` (48) |
| `emb_dim` | `32` | 3 | size of the vector that flows through the entire model with same width at each layer | `train.py` (48) |
| `num_heads` | `2` | 4 | emb_dim must be divisible by this, model splits each vector into 2 halves of 16 numbers and runs attention separately | `train.py` (49) |
| `ff_dim` | `128` | 5 | in the feed forward step, vector is stretched from 32 to 128 temporarily to allow model compute richer things | `train.py` (49) |
| `num_layers` | `2` | 6 | number of layers of the LLM | `train.py` (49) |
| `seq_len` | `16` | 7 | each training example is a 16 rows table, we cut the input corpus into pieces of 16 tokens | `train.py` (44, 45, 73), `talk.py` (24) |
| `ln_eps` | `1e-5` | 8 | tiny number added inside layer norm so we never divide by zero | **Nothing** |
| `corpus_path` | `"data/corpus.csv"` | 11 | file the text is read from | `train.py` (40) |
| `corpus_column` | `"text"` | 12 | which column of the csv holds the text | `train.py` (40) |
| `max_rows` | `2000` | 13 | only read this many rows, keeps training fast | `train.py` (40) |
| `num_merges` | `500` | 14 | how many times BPE glues the most common pair into a new token, more merges = bigger vocab | `train.py` (41) |
| `val_fraction` | `0.1` | 15 | last 10% of the token stream is kept aside to test the model on text it never trained on | `train.py` (43) |
| `stride` | `1` | 16 | how far the window slides each time, 1 = every possible window (neighbours overlap by 15 tokens) | `train.py` (44, 45) |
| `batch_size` | `16` | 19 | how many windows we average the gradient over before one update | `train.py` (54, 60) |
| `epochs` | `20` | 20 | how many full passes over the training windows | `train.py` (55, 59) |
| `grad_clip` | `1.0` | 21 | if the whole gradient is longer than this, shrink it to this length so one bad batch cant blow up the model | `train.py` (62) |
| `lr_max` | `1e-3` | 24 | biggest learning rate, reached right after warmup | `train.py` (65) |
| `lr_min` | `1e-4` | 25 | smallest learning rate, where the cosine decay ends | `train.py` (65) |
| `warmup_steps` | `50` | 26 | number of steps spent ramping the learning rate up from near 0, a fresh model is fragile | `train.py` (64) |
| `beta1` | `0.9` | 27 | how much of the old gradient direction Adam keeps (momentum) | `train.py` (51) |
| `beta2` | `0.999` | 28 | how much of the old squared gradient Adam keeps (typical gradient size) | `train.py` (51) |
| `adam_eps` | `1e-8` | 29 | tiny number so Adam never divides by zero | `train.py` (51) |
| `weight_decay` | `0.01` | 30 | each step shrinks the weight matrices slightly, only matrices, not biases or layer norm | `train.py` (51) |
| `temperature` | `0.8` | 33 | below 1 = safer and more repetitive, above 1 = more random, 0 = always pick the top token | `train.py` (74), `talk.py` (25) |
| `top_k` | `0` | 34 | only sample from the k most likely tokens, 0 = turned off | `train.py` (74), `talk.py` (25) |
| `top_p` | `0.9` | 35 | only sample from the smallest group of tokens whose probabilities add up to 0.9 | `train.py` (74), `talk.py` (25) |
| `max_new_tokens` | `50` | 36 | how many tokens talk.py writes after your prompt | `talk.py` (24) |
| `seed` | `42` | 39 | fixes the random numbers so a run can be repeated exactly | `train.py` (37), `talk.py` (9) |
| `head_dim` | `emb_dim // num_heads` = `16` | 44 | width of one head, 32 / 2 = 16 | **Nothing** |

Assertions that run when `config` is imported:

- Line 42: `emb_dim % num_heads == 0`, message "emb_dim must be divisible by num_heads"
- Line 43: `seq_len <= context_len`, message "a training window cannot be longer than the position table"

## 2. Values with no effect

- **`ln_eps`:** `train.py` lines 48-49 do not pass it to `ModelConfig`, so `ModelConfig.ln_eps` keeps its own default of `1e-5` (`transformer.py` line 15). Changing `config.ln_eps` changes nothing.
- **`head_dim`:** nothing reads it. `attention_forward` (line 73) and `attention_backward` (line 94) compute the same value locally.

## 3. Values that exist outside `config.py`

| Value | Location | Note |
|---|---|---|
| `vocab_size` | `tok.vocab_size`, `train.py` line 48 | Depends on the corpus; not a config value |
| `30` new tokens per sample | `train.py` line 73 | `config.max_new_tokens` is not used during training |
| `3` seed tokens per sample | `train.py` line 72 | Literal |
| `50` logging interval | `train.py` line 67 | Literal; unrelated to `warmup_steps` |
| `"checkpoints/model"` | `train.py` line 36, `talk.py` line 7 | Default prefix |
| `1e-6` greedy threshold | `generate.py` line 8 | Literal |
| `shuffle=True` | `data.py` line 36 | Function default; `train.py` does not change it |
| `epsilon=1e-5` | `ops.py` line 42 | Function default; the model always passes `cfg.ln_eps` |
| `ln_eps=1e-5` | `transformer.py` line 15 | Dataclass default, and the value training actually uses |
| `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `weight_decay=0.0` | `optim.py` line 7 | Class defaults. `train.py` passes config values, so decay in the pipeline is `0.01`, not `0.0` |
| `max_merges=1000` | `tokenizer.py` lines 24, 76, 99 | Function defaults. `train.py` passes `500` |
| `h=1e-5`, `eps=1e-12`, `samples_per_param=5` | `gradcheck.py` lines 3, 16, 19 | Test-time only |

## 4. Values that depend on the corpus

These follow from the config and from the data file. They are not stored anywhere.

**Actual Model Output** (computed this session from the modified working-tree `data/corpus.csv`, by running `load_corpus`, `build_tokenizer` and `encode_corpus`; no training):

| Quantity | Value |
|---|---|
| Rows read | 1,978 (fewer than `max_rows`, so the limit has no effect on this file) |
| Characters in the joined text | 457,575 |
| Vocabulary size `V` | 575 |
| Tokens `N` | 250,128 |
| Train/validation cut | 225,115 |
| Training windows at `stride = 1` | 225,099 |
| Steps per epoch at `batch_size = 16` | 14,069 |
| Total steps at `epochs = 20` | 281,380 |

The committed `data/corpus.csv` gives `V = 564` instead. The existing checkpoint was not trained for 281,380 steps; see `DISCREPANCIES.md`, section 7.

## 5. Inaccurate wording in code comments

Reported only. Do not quote these on the website as explanations.

| Location | Comment says | Code does |
|---|---|---|
| `config.py` line 2 | "the model can look 32 texts at once" | `context_len` is the number of `pos_emb` rows, the longest allowed sequence in tokens. Training and generation use only 16 |
| `transformer.py` line 22 | `pos_emb` is a "weighted average of values" | It is a learned table with one row per position |
| `optim.py` line 1 | "takes the gradients ... and updates them" | It updates parameters; only `clip_grad_norm` changes gradients |

# Repository inventory

Pinned commit: `c17aa75afd287dc623009168f50c4544fc1d4a32` (committed 2026-11-03T16:04:10+05:45).

Working tree at inspection: dirty. `data/corpus.csv` is modified and `docs-site/` is untracked. Every `.py` file is unchanged from the pinned commit, so the line numbers below apply to that commit.

Line ranges come from parsing each file with Python's `ast` module. They run from the first line (including any decorator) to the last line of the body. Later tasks must generate line numbers from code, not copy them by hand from this file.

"Reachable" means reachable from `train.main` or `talk.main`, using a static call graph of the root modules.

## 1. Files

| Path | Tracked | Lines | Role |
|---|---|---:|---|
| `config.py` | yes | 44 | Hyperparameters as module-level constants, plus two import-time assertions |
| `tokenizer.py` | yes | 119 | Byte-pair-encoding (BPE) tokenizer, working on characters |
| `data.py` | yes | 43 | Corpus loading, encoding, split, windows, batches |
| `ops.py` | yes | 60 | Differentiable operations with hand-written backward functions |
| `gradcheck.py` | yes | 33 | Finite-difference gradient checker, used only by tests |
| `transformer.py` | yes | 190 | Parameters, forward pass, backward pass |
| `optim.py` | yes | 50 | Adam with decoupled weight decay, learning-rate schedule, gradient clipping |
| `generate.py` | yes | 35 | Sampling and the generation loop |
| `checkpoint.py` | yes | 46 | Save and load weights, model shape and tokenizer |
| `train.py` | yes | 85 | Training entry point |
| `talk.py` | yes | 30 | Interactive inference entry point |
| `tests/test_checkpoint.py` | yes | 76 | Checkpoint tests |
| `tests/test_data.py` | yes | 83 | Data pipeline tests |
| `tests/test_generate.py` | yes | 82 | Sampling tests |
| `tests/test_gradcheck.py` | yes | 39 | Gradient-check tests |
| `tests/test_ops.py` | yes | 88 | Operation tests |
| `tests/test_optim.py` | yes | 49 | Optimizer, schedule and clipping tests |
| `tests/test_tokenizer.py` | yes | 50 | Tokenizer tests |
| `tests/test_transformer.py` | yes | 110 | Model tests |
| `data/corpus.csv` | yes, modified | 1979 (`wc -l`) | Training text; one column named `text` |
| `.gitignore` | yes | 6 | Ignores `__pycache__`, `.vscode`, `.venv`, `.pytest_cache`, `checkpoints/` |
| `checkpoints/model.npz`, `.json`, `.tok.json` | no, ignored | n/a | Existing trained checkpoint; not part of any commit |

There is no `conftest.py`, `pytest.ini`, `pyproject.toml`, `requirements.txt` or README at the pinned commit. Tests import root modules by bare name, so they are run with `python -m pytest tests` from the repository root.

`data/corpus.csv` (working tree): the header is `text` and `pandas` reads 1,978 rows. The first row begins "There's a garden that I ken, Full of little gentlemen;". The committed version is different text; it begins "First Citizen:". See `DISCREPANCIES.md`, section 7.

The reference PDF is not in the repository. It was read from `C:\Users\upahar\Downloads\LLM_study_guide_v2_streamlined.pdf`.

## 2. Modules

### config.py

No functions or classes. No imports. Every constant is listed in `HYPERPARAMETERS.md`.

Assertions: `assert emb_dim % num_heads == 0` (line 42) and `assert seq_len <= context_len` (line 43).

### tokenizer.py

Imports: `re`. Constants: `UNK = "<unk>"` (line 3); `PRE_TOKEN_RE = re.compile(r"\w+|\s|[^\w\s]")` (line 4).

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `count_words(text)` | 6-10 | Counts pre-token frequencies | train |
| `merge_tokens(tokens, pair, merged)` | 12-22 | Replaces adjacent `pair` occurrences with `merged`, scanning left to right | train, talk |
| `train_bpe(word_freqs, max_merges=1000)` | 24-61 | Learns merges; returns dicts with `pair`, `merged`, `frequency` | train |
| `apply_merges(text, merges)` | 63-73 | Pre-tokenizes text and replays all merges on each pre-token | train, talk |
| `train_bpe_on_text(text, max_merges=1000)` | 76-77 | Runs `train_bpe(count_words(text), max_merges)` | train |
| `class Tokenizer` | 79-96 | Holds merges and vocabulary; encodes and decodes | train, talk |
| `Tokenizer.__init__(self, merges, stoi, itos)` | 80-84 | Stores its arguments and sets `vocab_size = len(itos)` | train, talk |
| `Tokenizer.encode(self, text)` | 86-90 | Text to ids; tokens not in the vocabulary map to the `<unk>` id | train, talk |
| `Tokenizer.decode(self, ids)` | 92-96 | Ids to text by joining `itos` entries | train, talk |
| `build_tokenizer(text, max_merges=1000)` | 99-119 | Trains merges, builds `itos` (`<unk>`, then characters, then merged tokens) and `stoi` | train |

### data.py

Imports: `numpy as np`, `pandas as pd`, `tokenizer.Tokenizer` (annotation only). No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `load_corpus(path, column, max_rows) -> str` | 5-12 | Reads up to `max_rows` CSV rows, drops missing values, strips, joins with newlines, adds a final newline | train |
| `encode_corpus(text: str, tok: Tokenizer) -> np.ndarray` | 15-16 | Encodes text into a 1-D `int64` array | train |
| `split_ids(ids, val_fraction)` | 19-21 | Splits the id stream without shuffling: leading part is train, trailing part is validation | train |
| `make_windows(ids, seq_len, stride=1)` | 24-33 | Returns input and target `pandas` DataFrames of sliding windows | train |
| `iterate_batches(inputs, targets, batch_size, rng, shuffle=True)` | 36-43 | Yields batches as NumPy arrays; keeps a final short batch | train |

### ops.py

Imports: `numpy as np`. No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `xavier_uniform(fan_in, fan_out, rng)` | 3-5 | Samples a `(fan_in, fan_out)` matrix from a uniform distribution | train |
| `linear_forward(X, W, b)` | 7-8 | Returns `X @ W + b` | **No: tests only** |
| `linear_backward(X, W, dY)` | 10-14 | Returns `dX, dW, db` | train |
| `relu_forward(X)` | 16-17 | Element-wise `max(0, X)` | train, talk |
| `relu_backward(Z, dA)` | 19-20 | Keeps `dA` where `Z > 0` | train |
| `softmax_forward(x)` | 22-24 | Softmax on the last axis, with the maximum subtracted first | train, talk |
| `softmax_backward(y, dY)` | 26-27 | Softmax backward product | train |
| `cross_entropy_forward(logits, targets)` | 29-34 | Returns the mean cross-entropy and the probabilities | train |
| `cross_entropy_backward(probs, targets)` | 36-40 | Returns the gradient with respect to the logits | train |
| `layer_norm_forward(X, gamma, beta, epsilon=1e-5)` | 42-47 | Layer norm; returns the output and cache `(Xhat, s)` | train, talk |
| `layer_norm_backward(dY, cache, gamma)` | 49-57 | Returns `dX, dgamma, dbeta` | train |
| `perplexity(loss)` | 59-60 | Returns `exp(loss)` | train |

### gradcheck.py

Imports: `numpy as np`. No constants. No root module imports this file.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `numerical_grad(f, x, h=1e-5)` | 3-14 | Central-difference gradient of every element of `x` | **No: tests only** |
| `relative_error(a, b, eps=1e-12)` | 16-17 | Relative difference between two scalars | **No: reached only through `check_params`** |
| `check_params(loss_fn, params, grads, rng, samples_per_param=5, h=1e-5)` | 19-33 | Worst relative error per parameter at randomly sampled indices | **No: tests only** |

### transformer.py

Imports: `dataclasses.dataclass`, `numpy as np`, and from `ops`: `cross_entropy_forward`, `cross_entropy_backward`, `layer_norm_backward`, `layer_norm_forward`, `linear_backward`, `relu_backward`, `relu_forward`, `softmax_forward`, `softmax_backward`, `xavier_uniform`. All are used. No module-level constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `class ModelConfig` | 7-15 | Frozen dataclass of the model shape; `ln_eps` defaults to `1e-5` | train, talk |
| `init_params(cfg: ModelConfig, rng)` | 17-42 | Builds the flat dictionary of parameter arrays | train |
| `count_params(params)` | 44-45 | Counts scalar parameters | **No: tests only** |
| `embed_forward(params, ids)` | 52-53 | Adds token and position embeddings | train, talk |
| `embed_backward(dX0, ids, grads)` | 55-57 | Accumulates embedding gradients | train |
| `split_heads(M, H)` | 60-62 | Reshapes `(T,d)` to `(H,T,dh)` | train, talk |
| `merge_heads(Mh)` | 65-67 | Reshapes `(H,T,dh)` to `(T,d)` | train, talk |
| `attention_forward(Xn, params, p, cfg: ModelConfig)` | 70-86 | Multi-head causal self-attention; returns output and cache | train, talk |
| `attention_backward(dOut, c, params, p, cfg: ModelConfig, grads)` | 92-109 | Fills attention gradients; returns `dXn` | train |
| `ffn_forward(Xn, params, p)` | 112-116 | Linear, ReLU, linear; returns output and cache | train, talk |
| `ffn_backward(dOut, c, params, p, grads)` | 118-128 | Fills feed-forward gradients; returns `dXn` | train |
| `block_forward(X, params, l: int, cfg: ModelConfig)` | 131-139 | One pre-LN block; returns output and cache | train, talk |
| `block_backward(dY, c, params, l, cfg: ModelConfig, grads)` | 141-152 | Backward through one block; returns the input gradient | train |
| `forward(params, ids, cfg: ModelConfig)` | 155-168 | Whole model on one sequence; returns logits and cache | train, talk |
| `loss_and_grads(params, ids, targets, cfg: ModelConfig)` | 171-190 | Forward, loss and full backward for one sequence | train |

### optim.py

Imports: `math`, `numpy as np`. No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `class Adam` | 6-30 | Optimizer state and update | train |
| `Adam.__init__(self, params, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)` | 7-11 | Creates zeroed `m` and `v`, sets `t = 0`, stores settings | train |
| `Adam.step(self, params, grads, lr)` | 13-30 | One in-place update, with decay only where `ndim >= 2` | train |
| `lr_schedule(step, warmup, total, lr_max, lr_min)` | 32-37 | Linear warmup, then cosine decay | train |
| `clip_grad_norm(grads, max_norm)` | 40-50 | Global-norm clipping in place; returns the norm before clipping | train |

### generate.py

Imports: `ops.softmax_forward`, `transformer.ModelConfig`, `transformer.forward`, `numpy as np`. No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `sample_next(logits, temperature, top_k, top_p, rng)` | 7-26 | Chooses one token id | train, talk |
| `generate(params, cfg: ModelConfig, seed_ids, max_new_tokens, window, temperature, top_k, top_p, rng)` | 28-35 | Appends sampled ids, feeding `forward` the last `window` ids | train, talk |

### checkpoint.py

Imports: `json`, `os`, `dataclasses.asdict`, `numpy as np`, `tokenizer.Tokenizer`, `transformer.ModelConfig`. No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `save_checkpoint(prefix, params, cfg: ModelConfig, tok: Tokenizer)` | 16-30 | Writes `prefix.npz`, `prefix.json` and `prefix.tok.json` | train |
| `load_checkpoint(prefix)` | 32-46 | Reads the three files and rebuilds `stoi` from `itos` | talk |

### train.py

Imports: `math`, `numpy as np`, `config`; `data.load_corpus`, `encode_corpus`, `split_ids`, `make_windows`, `iterate_batches`; `checkpoint.save_checkpoint`; `generate.generate`; `ops.cross_entropy_forward`, `perplexity`; `optim.Adam`, `clip_grad_norm`, `lr_schedule`; `tokenizer.build_tokenizer`; `transformer.ModelConfig`, `init_params`, `forward`, `loss_and_grads`. No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `batch_loss_and_grads(params, xb, yb, cfg: ModelConfig)` | 12-24 | Loops over a batch and averages loss and gradients | train |
| `validation_loss(params, cfg, inputs, targets)` | 26-34 | Mean cross-entropy over all validation windows | train |
| `main(prefix="checkpoints/model")` | 36-81 | Full training pipeline; entry point at lines 84-85 | train |

### talk.py

Imports: `sys`, `numpy as np`, `config`, `checkpoint.load_checkpoint`, `generate.generate`. No constants.

| Symbol | Lines | Purpose | Reachable |
|---|---|---|---|
| `main(prefix="checkpoints/model")` | 7-27 | Loads a checkpoint and runs the prompt loop; entry point at lines 29-30 | talk |

## 3. Tests

Result of `python -m pytest tests -q -p no:cacheprovider` in this session: `67 passed in 4.11s`. That is 64 test functions; `test_round_trip` has 4 parameter cases. Tests write only under pytest's temporary directory.

### tests/test_checkpoint.py

Imports: `json`, `numpy`, `pytest`, `checkpoint`, `tokenizer`, `transformer`. Constant: `CORPUS` (line 10).

| Symbol | Lines | Purpose |
|---|---|---|
| `saved(tmp_path)` | 13-21 | Fixture: saves a checkpoint and returns its parts |
| `test_save_creates_the_three_files(saved)` | 24-27 | All three files exist and are not empty |
| `test_config_round_trips(saved)` | 30-32 | Loaded config equals the saved one |
| `test_weights_round_trip_exactly(saved)` | 35-40 | Keys and arrays are exactly equal |
| `test_tokenizer_round_trips(saved)` | 43-52 | Vocabulary, encode and decode match |
| `test_loaded_model_gives_identical_logits(saved)` | 55-61 | Logits are exactly equal after loading |
| `test_non_ascii_characters_survive(tmp_path)` | 64-70 | Non-ASCII text round-trips through a reloaded tokenizer |
| `test_tok_file_is_plain_json(saved)` | 73-76 | Top-level keys are `merges` and `itos`; `itos[0]` is `<unk>` |

### tests/test_data.py

Imports: `numpy`, `pandas`, `pytest`, `data`. Constant: `IDS` (line 7).

| Symbol | Lines | Purpose |
|---|---|---|
| `test_window_table_matches_hand_example()` | 10-16 | Shapes and first and last rows match a hand example |
| `test_targets_are_inputs_shifted_by_one()` | 19-21 | Targets are inputs shifted by one |
| `test_stride_skips_windows()` | 24-26 | Stride 2 skips windows |
| `test_too_few_tokens_raises()` | 29-31 | Too few tokens raises `ValueError` |
| `test_split_is_contiguous_and_disjoint()` | 34-38 | The split is 90/10 and contiguous |
| `test_window_counts_for_n_3000()` | 41-45 | N=3000 gives 2684 train and 284 validation windows |
| `test_batches_cover_every_window_once()` | 48-58 | Every window appears once and pairs stay aligned |
| `test_no_shuffle_keeps_order()` | 61-64 | `shuffle=False` keeps order |
| `test_load_corpus_joins_rows_and_skips_missing(tmp_path)` | 67-70 | Text is stripped and missing rows are dropped |
| `test_load_corpus_respects_max_rows(tmp_path)` | 73-76 | `max_rows` limits rows |
| `test_load_corpus_wrong_column_raises(tmp_path)` | 79-83 | A missing column raises `ValueError` |

### tests/test_generate.py

Imports: `numpy`, `generate`, `transformer`. No constants.

| Symbol | Lines | Purpose |
|---|---|---|
| `class SpyRng` | 7-11 | Test double that records the sampling distribution |
| `SpyRng.choice(self, n, p)` | 9-11 | Stores `p` and returns 0 |
| `probs_used(logits, temperature=1.0, top_k=0, top_p=1.0)` | 14-17 | Helper: returns the distribution `sample_next` used |
| `test_temperature_examples()` | 20-24 | Temperature distributions match hand values |
| `test_temperature_zero_is_greedy()` | 27-30 | Temperature 0 returns the argmax |
| `test_top_k_example()` | 33-35 | `top_k=2` matches hand values |
| `test_top_k_one_is_greedy()` | 38-41 | `top_k=1` always returns the argmax |
| `test_top_p_example()` | 44-47 | `top_p=0.9` keeps the crossing token |
| `test_top_p_adapts_to_confidence()` | 50-54 | Kept token count depends on confidence |
| `tiny()` | 57-60 | Helper: builds a tiny config and parameters |
| `test_generate_length_and_range()` | 63-67 | Output length, seed prefix and id range |
| `test_generate_window_keeps_long_sequences_legal()` | 70-75 | A seed longer than `context_len` still works |
| `test_greedy_generation_is_deterministic()` | 78-82 | Temperature 0 output does not depend on the rng seed |

### tests/test_gradcheck.py

Imports: `numpy`, `gradcheck`, `transformer`. No constants.

| Symbol | Lines | Purpose |
|---|---|---|
| `tiny_setup()` | 6-13 | Helper: tiny model and inputs |
| `test_numerical_grad_on_quadratic()` | 16-19 | Numerical gradient of a quadratic equals `2x` |
| `test_every_tensor_passes()` | 22-30 | Relative error below `1e-4`, skipping names ending in `.bk` |
| `test_key_bias_gradient_is_zero()` | 33-39 | Key-bias gradients are below `1e-12` |

### tests/test_ops.py

Imports: `numpy`, all 12 `ops` functions. No constants.

| Symbol | Lines | Purpose |
|---|---|---|
| `test_linear_worked_example()` | 10-19 | `linear_forward` and `linear_backward` match hand values |
| `test_relu()` | 22-25 | ReLU forward and backward, including gradient 0 at Z=0 |
| `test_softmax_values_and_stability()` | 28-33 | Known values; no NaN for large inputs |
| `test_softmax_backward_rows_sum_to_zero()` | 36-42 | Rows sum to zero; values match |
| `test_cross_entropy_worked_example()` | 45-51 | Loss and gradient match hand values |
| `test_cross_entropy_is_mean_over_positions()` | 54-60 | Uniform logits give ln 5; the gradient has the `1/T` |
| `test_cross_entropy_extreme_logits_no_nan()` | 63-66 | Loss stays finite for large logits |
| `test_layer_norm_worked_example()` | 69-77 | Forward and backward match hand values |
| `test_xavier_bound()` | 80-84 | Shape and bound |
| `test_perplexity_of_uniform_is_vocab_size()` | 87-88 | `perplexity(ln 300)` is about 300 |

### tests/test_optim.py

Imports: `numpy`, `optim`. No constants.

| Symbol | Lines | Purpose |
|---|---|---|
| `test_adam_two_step_hand_trace()` | 6-13 | Two steps match hand values |
| `test_adam_first_step_is_lr_times_sign_of_gradient()` | 16-21 | First update is `-lr * sign(g)` |
| `test_weight_decay_only_touches_matrices()` | 24-29 | Decay changes the 2-D array, not the 1-D array |
| `test_lr_schedule_values()` | 32-35 | Schedule values match |
| `test_clip_rescales_all_tensors_together()` | 38-43 | Global clipping; returns the norm before clipping |
| `test_clip_leaves_small_gradients_alone()` | 46-49 | Small gradients are unchanged |

### tests/test_tokenizer.py

Imports: `pytest`, `tokenizer`. Constant: `CORPUS` (line 7).

| Symbol | Lines | Purpose |
|---|---|---|
| `test_count_words_splits_words_spaces_punctuation()` | 10-11 | Pre-token counts |
| `test_merge_tokens_simple()` | 14-15 | Simple merge |
| `test_merge_tokens_overlap_is_left_to_right()` | 18-20 | Overlaps merge from the left |
| `test_first_three_merges_match_hand_trace()` | 23-26 | First three merges match |
| `tok()` | 29-31 | Fixture: tokenizer with 5 merges |
| `test_unk_is_id_zero(tok)` | 34-35 | `<unk>` is id 0 |
| `test_no_characters_dropped(tok)` | 38-41 | Base characters stay in the vocabulary |
| `test_unseen_character_becomes_unk(tok)` | 44-45 | Unseen character maps to id 0 |
| `test_round_trip(tok, s)` | 48-50 | Decode inverts encode on four strings |

### tests/test_transformer.py

Imports: `numpy`, `optim.Adam`, and 11 names from `transformer`. No constants.

| Symbol | Lines | Purpose |
|---|---|---|
| `tiny_cfg(**kw)` | 9-12 | Helper: tiny `ModelConfig` |
| `test_param_count_matches_formula()` | 15-19 | `count_params` is 45,996 at V=300 |
| `test_initial_loss_is_close_to_ln_vocab()` | 22-28 | Initial loss is within 0.5 of ln 300 |
| `test_embedding_worked_example()` | 31-35 | `embed_forward` matches hand values |
| `test_embedding_backward_sums_repeated_ids()` | 38-44 | Repeated ids accumulate |
| `test_ffn_worked_example()` | 47-56 | Feed-forward matches hand values |
| `test_split_merge_heads_round_trip()` | 59-63 | Split then merge restores the input |
| `test_attention_weights_are_causal_and_normalised()` | 66-74 | Upper triangle is 0; rows sum to 1 |
| `test_future_tokens_do_not_change_earlier_logits()` | 77-86 | Changing a later token leaves earlier logits unchanged |
| `test_grads_have_the_shape_of_params()` | 89-95 | Gradients have the same keys and shapes as parameters |
| `test_overfit_one_window()` | 98-110 | Loss is below 0.05 after 200 Adam steps on one window |

## 4. Import graph

Internal imports only. An arrow means "imports".

```text
data         -> tokenizer
transformer  -> ops
generate     -> ops, transformer
checkpoint   -> tokenizer, transformer
train        -> config, data, checkpoint, generate, ops, optim, tokenizer, transformer
talk         -> config, checkpoint, generate
config, tokenizer, ops, optim, gradcheck -> no internal imports
```

No root module imports `gradcheck`. It is imported only by `tests/test_gradcheck.py`.

Third-party imports: `numpy` everywhere numeric; `pandas` in `data.py` and `tests/test_data.py`; `pytest` in tests. Environment used in this session: Python 3.14.4, NumPy 2.5.2, pandas 3.1.0.

Test imports:

```text
tests/test_checkpoint  -> checkpoint, tokenizer, transformer
tests/test_data        -> data
tests/test_generate    -> generate, transformer
tests/test_gradcheck   -> gradcheck, transformer
tests/test_ops         -> ops
tests/test_optim       -> optim
tests/test_tokenizer   -> tokenizer
tests/test_transformer -> optim, transformer
```

## 5. Defined but never called by the pipeline

Verified with a static call graph from `train.main` and `talk.main`: 53 of 58 root functions, classes and methods are reachable. These 5 are not:

| Symbol | Lines | Who uses it | Note |
|---|---|---|---|
| `ops.linear_forward` | `ops.py` 7-8 | `tests/test_ops.py` only | `transformer.py` does not import it. The same formula is written inline at `transformer.py` lines 75, 76, 77, 85, 113, 116 and 166 |
| `transformer.count_params` | `transformer.py` 44-45 | `tests/test_transformer.py` only | `train.main` does not print a parameter count |
| `gradcheck.numerical_grad` | `gradcheck.py` 3-14 | `tests/test_gradcheck.py` only | |
| `gradcheck.check_params` | `gradcheck.py` 19-33 | `tests/test_gradcheck.py` only | |
| `gradcheck.relative_error` | `gradcheck.py` 16-17 | `check_params` only | Tests reach it indirectly |

`attention_backward` also writes the `linear_backward` formulas inline (`transformer.py` lines 95-97 and 106-108), while `ffn_backward` and `loss_and_grads` call `linear_backward`.

Reachable from `train.main` but not `talk.main` (32 symbols): all backward functions, `init_params`, `xavier_uniform`, cross-entropy, `perplexity`, the optimizer, data loading and tokenizer training.

Reachable from `talk.main` but not `train.main`: only `talk.main` itself and `checkpoint.load_checkpoint`.

Config values never read by any module: `ln_eps` and `head_dim`. See `HYPERPARAMETERS.md`.

Root code that no test exercises: `data.encode_corpus`, `train.batch_loss_and_grads`, `train.validation_loss`, `train.main`, `talk.main`.

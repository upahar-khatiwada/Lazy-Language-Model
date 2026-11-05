# Discrepancies

Pinned commit: `c17aa75`. Reference: `LLM_study_guide_v2_streamlined.pdf`.

The code wins every disagreement. The website shows the discrepancy where it matters and follows the code. The PDF states it was not run against this repository: "Your repository was not used or modified" (p.17). Its result numbers come from "a Linux sandbox and a synthetic corpus" (p.17), so none of them may be presented as output from this project.

## 1. Decisions for the human

Each item is marked **NEEDS HUMAN DECISION**. The default is what later tasks should assume until told otherwise.

| ID | Question | Why it matters | Default |
|---|---|---|---|
| D1 | Which checkpoint and corpus may back Actual Model Output panels? | The existing checkpoint does not match the pinned commit. See section 7 | Show no Actual Model Output from training history. Values read from the checkpoint must state that it is an ignored local file trained on an uncommitted corpus with a different config |
| D2 | May test-only mathematics (`gradcheck.py`, `linear_forward`, `count_params`) be documented? | The rules say to document implemented mathematics, but this code is not part of the pipeline | Yes, in a clearly labelled testing section, and never in the pipeline flow |
| D3 | May PDF worked examples be reused as Illustrative Examples? | They contain hand-picked numbers. Many also appear in tests | Yes, only where the code gives the same result, always labelled Illustrative Example |
| D4 | Should the PDF be available to later tasks? | It is outside the repository, at `C:\Users\upahar\Downloads\`. A fresh session on another machine cannot read it | Later tasks use `PDF_VS_CODE.md` and do not reopen the PDF |
| D5 | Should the site name the repository with or without the trailing hyphen? | `docs-site/CLAUDE.md` says `Lazy-Language-Model-LLM`; the remote is `Lazy-Language-Model-LLM-` | Use the real remote name for links |

## 2. Files and project structure

| Topic | PDF | Code |
|---|---|---|
| Corpus file | `data/corpus.txt` (p.2, 16, 17) | `data/corpus.csv`, set by `config.corpus_path` |
| Corpus format | Plain text | CSV; column named by `config.corpus_column` |
| Test files | "Three test files" (p.2); "3 files, 13 tests" (p.17) | 8 files; 64 functions; 67 cases collected |
| `gradcheck.py` size | "about 45 lines" (p.2) | 33 lines |
| Dependencies | "NumPy only" (p.1) | `data.py` also needs `pandas` |
| `train.py` imports | "everything above" (p.2), which includes `gradcheck` | Does not import `gradcheck` |
| Build order | `train.py` step 8, `checkpoint.py` step 9 (p.3) | `train.py` imports `checkpoint`. The PDF's own listing on p.16 has the same dependency |
| Upstream project | PDF compares with `w3cj/how-llms-work` (p.1, 19) | Nothing in the repository refers to it. Do not document this comparison as a feature |

## 3. Names and signatures

| PDF | Code |
|---|---|
| `count_words(text: str) -> Counter` | `count_words(text)` returns a plain `dict` |
| `merge_tokens(tokens, pair)` | `merge_tokens(tokens, pair, merged)` |
| `train_bpe(text, num_merges) -> list[tuple]` | `train_bpe(word_freqs, max_merges=1000)` returns a list of dicts with `pair`, `merged`, `frequency` |
| Not present | `apply_merges(text, merges)`; `train_bpe_on_text(text, max_merges=1000)` |
| `Tokenizer.__init__(self, merges, base_chars)` | `Tokenizer.__init__(self, merges, stoi, itos)` |
| `@property vocab_size` | Plain attribute set in `__init__` |
| `Tokenizer._encode_pretoken`, `_cache`, `save`, `load` | Not present. `checkpoint.py` handles saving and loading the tokenizer |
| `build_tokenizer(text, num_merges)` | `build_tokenizer(text, max_merges=1000)` |
| `load_corpus(path)` | `load_corpus(path, column, max_rows)` |
| `make_windows` returns NumPy arrays | Returns `pandas` DataFrames |
| `softmax(z, axis=-1)` | `softmax_forward(x)`; last axis only |
| `softmax_backward(p, dp)` | `softmax_backward(y, dY)` |
| `cross_entropy(logits, targets)` | `cross_entropy_forward(logits, targets)` |
| `relu_forward(Z)` | `relu_forward(X)` |
| `layer_norm_forward(X, gamma, beta, eps=1e-5)` | `layer_norm_forward(X, gamma, beta, epsilon=1e-5)` |
| `optim.global_norm(grads)` | Not present; written inline in `clip_grad_norm` |
| `train.evaluate(...) -> tuple[float, float]` | `train.validation_loss(...)` returns one float; `main` calls `perplexity` |
| `train.train(params, cfg, train_data, val_data, tok, rng, log_every=50)` | Not present; the loop is inside `main`, and `50` is a literal |
| Not present | `train.batch_loss_and_grads` |
| `main(corpus_path="data/corpus.txt", prefix="checkpoints/model")` | `main(prefix="checkpoints/model")`; corpus path comes from `config` |
| `va_x[0][:3]` | `va_x.iloc[0][:3]` |
| Block prefix written `bl.` (p.9) | `f"b{l}."`, for example `b0.wq` |
| Assert message "a training window cannot exceed the position table" | "a training window cannot be longer than the position table" |

Parameter names, shapes and initial values match the PDF's table apart from that prefix.

## 4. Formulas and behaviour

| Topic | PDF | Code | Effect |
|---|---|---|---|
| Pre-tokenizer regex | `r" ?\w+\|[^\w\s]\|\s"`; a word carries its leading space (p.4) | `r"\w+\|\s\|[^\w\s]"` (`tokenizer.py` line 4); a space is always its own pre-token | Merges cannot join a space to a word. Tokens differ from the PDF's |
| Case | Text is lower-cased (p.4, 5, 6) | No lower-casing anywhere. Case is preserved | Vocabulary is case-sensitive |
| "Byte-pair" | Called byte-pair encoding (p.2, 4) | `list(word)` splits into Unicode characters, not bytes | Say "character-level BPE" when describing the code |
| Encoding cache | Per-pre-token cache | None; merges are replayed for each pre-token | Performance only |
| `ln_eps` | `ModelConfig(..., config.ln_eps)` (p.16) | `train.py` lines 48-49 do not pass it | `config.ln_eps` has no effect; the value is the same today |
| Overfit test | "300 Adam steps" (p.17) | `test_overfit_one_window` runs 200 steps | |
| Key bias check | "All 38 tensors agree" (p.17) | `test_every_tensor_passes` skips `.bk` names; a separate test asserts their gradient is below `1e-12` | Equivalent intent, different check |
| Gradcheck thresholds | Bands at `1e-6`, `1e-4`, `1e-3` (p.8) | Test asserts `< 1e-4` | Do not present the bands as project rules |
| Default corpus size limit | None | `config.max_rows = 2000` | Present in code only |

These match the code on reading it line by line: Eq. 1, 3, 17-32, 34-36, 38-60. Eq. 2 also matches numerically (63,871 parameters at V=575, from the checkpoint). Only the left side of Eq. 37 is in the code. Eq. 61 and 62 are not computed by the code.

## 5. Defaults

All hyperparameters listed on PDF p.3-4 have the same value in `config.py`. The differences are additions in code: `corpus_path`, `corpus_column`, `max_rows`.

Comments differ throughout. Only the code's comments are quoted in `HYPERPARAMETERS.md`.

## 6. Problems inside the PDF

Reported, not resolved here.

1. (33) is missing from the numbering, with no explanation.
2. p.3 cites a "Two-step Adam table" that is not printed. p.13 has an inline check only.
3. p.4 requires "the regex as a parameter", but the PDF's listing hard-codes it.
4. p.18 says "step-0 loss" while p.17 reports loss "at step 1". The code logs at `step == 1`, on the batch before the first update applies.
5. No test names are given anywhere in the PDF.
6. T5 is named on p.2 and never described.
7. W means window count (Eq. 18), weight matrix (Eq. 19) and warmup steps (Eq. 58). V means vocabulary size and the value matrix. L means block count and loss. Website pages must not reuse a symbol for two meanings on one page.

## 7. Repository state and the checkpoint

**NEEDS HUMAN DECISION (D1).**

| Fact | Evidence |
|---|---|
| The working-tree corpus differs from the commit | `git status`: `M data/corpus.csv`. `git diff --stat`: 1978 insertions, 10479 deletions |
| The checkpoint is ignored by git | `.gitignore` contains `checkpoints/` |
| The checkpoint's vocabulary came from the modified corpus | Actual Model Output: `build_tokenizer` on the working-tree corpus reproduces the checkpoint's `itos` exactly (V=575). The committed corpus gives V=564, which does not match |
| The checkpoint was not trained with the committed config | See below |

Under the committed config, the working-tree corpus gives 225,099 training windows, 14,069 steps per epoch and 281,380 total steps.

The checkpoint itself gives a different count. `pos_emb` rows 16-31 receive only weight decay, so their ratio to their initial values (seed 42) equals the product of `(1 - lr * weight_decay)` over all steps. Measured ratio: 0.9078093812857 in every entry. Solving with the committed `lr_max`, `lr_min`, `warmup_steps` and `weight_decay` gives exactly **17,590** total steps, with the ratio matching to 13 significant digits.

17,590 equals 10 epochs of 1,759 steps. That fits `stride = 8` and `epochs = 10`, which would give 28,138 windows. **This is an inference.** Other settings can give the same total, such as `stride = 4` and `batch_size = 32`. The result also assumes the learning-rate settings were unchanged.

Supporting timestamps: the corpus was last written at 16:31 and the checkpoint at 16:49 on 2026-11-03. `config.py` was written at 16:52 but is identical to the commit, which suggests a temporary edit that was later reverted.

The checkpoint also has no record of its training settings. `model.json` holds only the model shape.

## 8. Possible LLM issues

Reported only. No repository file outside `docs-site/` was changed.

| Location | Issue |
|---|---|
| `train.py` lines 48-49 | `config.ln_eps` is never passed, so editing it has no effect |
| `config.py` line 44 | `head_dim` is never read |
| `talk.py` line 24 | The window comes from the current `config.seq_len`, while the model comes from the checkpoint. Editing config after training can exceed the checkpoint's `context_len` and fail the assertion in `forward` |
| `talk.py` line 30 | More than one command-line argument raises `TypeError` |
| `train.py` line 85 | `main()` takes no command-line prefix, unlike `talk.py` |
| `tests/test_generate.py` line 71 | The comment says 20 tokens; the code makes 21. The assertion is correct for 21 |
| `transformer.py` line 22, `config.py` line 2, `optim.py` line 1 | Misleading comments. See `HYPERPARAMETERS.md`, section 5 |
| `pos_emb` | Half of the table (rows 16-31) is allocated, saved and decayed but never used, at the committed `seq_len` and `context_len`. The PDF notes this on p.10 |
| Test setup | No `conftest.py` or config. Tests run with `python -m pytest tests`; plain `pytest` was not tried in this session |

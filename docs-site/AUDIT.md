# Documentation audit

Task 18. Report only: no page, data file, script or LLM file was changed.

- **Date:** 2026-11-06
- **HEAD:** `dda20ee2908a87007da8090d3412484fbb423cd6`
- **Pinned commit the site documents:** `66da45bea72176fa5d2f8dba2ea1c7e597e30dd3`. `git diff --name-only 66da45b HEAD -- '*.py' tests/` is empty, so the pin still matches the code.
- **Tools:** Node v26.13.0, Bun 1.3.13, Python 3.14.4 from `.venv`.

## 1. Summary

**The mathematics is correct.** Every equation in every concept was compared with the Python by hand, and none is wrong. Every code link resolves to the right line, tensor shapes match a real run, and the ten rows checked by hand all agree with the code. All checks pass.

**The problems are in what the pages say around the mathematics.** The checks test numbers and line numbers, and they do not read sentences.

| Severity | Count | Meaning                                                                                 |
| -------- | ----- | --------------------------------------------------------------------------------------- |
| High     | 2     | A page states something false about the code                                            |
| Medium   | 10    | A rule in `CLAUDE.md` is not met, or a page misleads a reader or overstates its content |
| Low      | 15    | Loose wording, small gaps or housekeeping                                               |

The findings most worth fixing first:

1. **F01:** the Parameter count section says `count_params` is called by `train.main`. Only a test calls it.
2. **F02:** Gradient Checking says the pipeline computes a formula at `gradcheck.py:31`. That file is not in the pipeline.
3. **F03 and F04:** 52 of the 53 concept `notes` never reach a page, so most PDF-versus-code differences appear only on Sources & Discrepancies and not on the page for that topic.
4. **F05:** 23 panels are labelled Illustrative Example but hold values from running the code. This needs your decision.
5. **F07:** Sources & Discrepancies publishes a local path (`C:\Users\upahar\Downloads\`) and internal working notes. This also needs your decision.

No new bug in the LLM was found.

## 2. How this was done

- **Code list:** a subagent that saw only the `.py` files and `tests/` listed 58 definitions and 213 operation rows, plus 48 non-obvious behaviours.
- **Claim list:** a subagent that saw only a snapshot of the built pages listed 1,466 prose claims, 371 of them with absolute words, 98 numeric panels and 34 confusing items.
- **Comparison:** I compared both lists with `contract/`, with `data/concepts/`, `data/links/` and `content/`, and with the Python.

**What was and was not read by hand**

- Read by hand: all six contract files, all 61 concepts (equations, symbols, shapes, summaries, reasons, links), the Python source, seven of the 24 pages in full or in part, and all 371 strong claims, each cut at about 200 characters.
- Not read one by one: the other 1,095 prose claims. They are covered by the subagent's lists of contradictions and by targeted searches.
- Interactive labs were read in their first state only. Their logic is covered by `tests/interactive.test.ts` and the golden tests.
- No browser test ran, under the standing rule in `CLAUDE.md`. GitHub and other external links were not opened, because the working rules allow no requests to other sites.
- The PDF was not reopened (decision D4). Every PDF statement here relies on `contract/PDF_VS_CODE.md` and `contract/DISCREPANCIES.md`.

## 3. Verification results

All commands ran in this session, on the final tree, from `docs-site/` unless noted. Started 2026-11-06T14:39:09+05:45, at HEAD `dda20ee`.

| Command                                                                                              | Exit | Result line                                                                            |
| ---------------------------------------------------------------------------------------------------- | ---- | -------------------------------------------------------------------------------------- |
| `bun run verify:generated`                                                                           | 0    | `OK: 5 generated files match a fresh extraction.`                                      |
| `bun run verify:derived`                                                                             | 0    | `OK: 5 derived file(s) match a fresh run.`                                             |
| `bun run verify:examples`                                                                            | 0    | `OK: 2 example(s) match a fresh run.`                                                  |
| `bun run verify:golden`                                                                              | 0    | `OK: 231 golden cases match a fresh run.`                                              |
| `bun run verify:shapes`                                                                              | 0    | `OK: data/golden/shapes.json matches a fresh run.`                                     |
| `bun run verify:links`                                                                               | 0    | `OK: 349 links, 61 concepts and 1 pipelines match the code.`                           |
| `bun run verify:actual`                                                                              | 0    | `OK: data/generated/actual matches a fresh export from checkpoints/model.`             |
| `bun run verify:content`                                                                             | 0    | `OK: 7 checks passed, 0 with warnings.` `Uncovered functions: none.`                   |
| `bun run lint`                                                                                       | 0    | `All matched files use Prettier code style!`                                           |
| `bun run typecheck`                                                                                  | 0    | no output from `tsc --noEmit`                                                          |
| `bun run test`                                                                                       | 0    | `Test Files  32 passed (32)`, `Tests  2484 passed (2484)`                              |
| `bun run build`                                                                                      | 0    | `(Static) prerendered as static content`                                               |
| `bun run verify:site`                                                                                | 0    | `OK: 50 pages, 5147 internal links, 3082 anchors, 502 GitHub permalinks. No problems.` |
| `python -B -m pytest tests -q -p no:cacheprovider --basetemp=<scratchpad>`, from the repository root | 0    | `67 passed in 2.20s`                                                                   |

**Base-path build.** This had never been run, and `MAINTENANCE.md` needs it for GitHub Pages. With `NEXT_PUBLIC_BASE_PATH=/Lazy-Language-Model-LLM-`, `bun run build` exited 0 and `bun run verify:site` printed `OK: 50 pages, 5147 internal links, 3082 anchors, 502 GitHub permalinks. No problems.` Links in `out/index.html` carried the prefix. I then rebuilt without the variable: both commands exited 0 with the same result line, and `out/index.html` has no prefixed link. No browser was opened on either build.

**Not run:** `bun run test:e2e`, because `CLAUDE.md` forbids it. `tests/e2e/` is stale (see F25).

**Hand checks:** a scratch script that writes each formula out on its own and compares it with the real function printed `PASS` on all 26 of its checks. It trained nothing and wrote no file. See section 6.

## 4. Implementation matrix

One row per implemented operation. M rows come from `contract/MATH_INVENTORY.md` and were re-derived from the code. X rows are operations the code performs that have no row in the inventory.

**Verdicts**

- **Yes:** correct.
- **Yes\*:** correct, with a finding about the wording or label around it. The note names the finding.
- **Partial:** part of it is missing or wrong. The note says what.
- **n/a:** does not apply, such as a scalar with no tensor shape, or an operation with no equation.

**Basis for each column**

- **Equation:** compared by hand with the code lines.
- **Code ref:** `verify:links` resolves every link, and the resolved lines were read by hand.
- **Shapes:** `verify:content` check 5 compares 192 of 202 stages with a real run at two configurations. The other 10 stages have no recorded tensor, and these were read by hand against the code.
- **Example:** the golden and mirror tests compare the numbers with the real function. The ten rows in section 6 were also recomputed by hand.
- **Explanation:** the concept's summary and reason were read by hand, along with strong claims about that operation.

| ID  | Operation                          | Code                                          | Impl. | Documented (concept)                                        | Equation | Code ref | Shapes | Example | Explanation | Note                                                     |
| --- | ---------------------------------- | --------------------------------------------- | ----- | ----------------------------------------------------------- | -------- | -------- | ------ | ------- | ----------- | -------------------------------------------------------- |
| M01 | Pre-token count                    | `tokenizer.py:8-9` `count_words`              | Yes   | Yes, `tokenizer.count_words`                                | Yes      | Yes      | n/a    | Yes     | Yes         | Regex at line 4 is explained on the page                 |
| M02 | Pair frequency                     | `tokenizer.py:32-38` `train_bpe`              | Yes   | Yes, `tokenizer.train_bpe`                                  | Yes      | Yes      | n/a    | Yes     | Yes         |                                                          |
| M03 | Best pair, first maximum wins      | `tokenizer.py:43-48` `train_bpe`              | Yes   | Yes, `tokenizer.train_bpe`                                  | Yes      | Yes      | n/a    | Yes     | Yes\*       | F22                                                      |
| M04 | Merge                              | `tokenizer.py:13-22` `merge_tokens`           | Yes   | Yes, `tokenizer.merge_tokens`                               | Yes      | Yes      | n/a    | Yes     | Yes         |                                                          |
| M05 | Encode                             | `tokenizer.py:87-90` `Tokenizer.encode`       | Yes   | Yes, `tokenizer.Tokenizer.encode`                           | Yes      | Yes      | Yes    | Yes     | Yes         | Replay is in `apply_merges`: X03                         |
| M06 | Vocabulary order                   | `tokenizer.py:102-113` `build_tokenizer`      | Yes   | Yes, `tokenizer.build_tokenizer`                            | Yes      | Yes      | n/a    | Yes     | Yes         |                                                          |
| M07 | Split                              | `data.py:20-21` `split_ids`                   | Yes   | Yes, `data.split_ids`                                       | Yes      | Yes      | Yes    | Yes     | Yes         | Shapes in prose; Reference says none: F08                |
| M08 | Windows                            | `data.py:28-32` `make_windows`                | Yes   | Yes, `data.make_windows`                                    | Yes      | Yes      | Yes    | Yes     | Yes\*       | PDF difference not on page: F04                          |
| M09 | Window count                       | `data.py:28` `make_windows`                   | Yes   | Yes, `data.window_count`                                    | Yes      | Yes      | n/a    | Yes\*   | Yes         | F05                                                      |
| M10 | Batch averaging                    | `train.py:13-22` `batch_loss_and_grads`       | Yes   | Yes, `train.batch_loss_and_grads`                           | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M11 | Validation loss                    | `train.py:27-34` `validation_loss`            | Yes   | Yes, `train.validation_loss`                                | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M12 | Step counts                        | `train.py:54-55` `main`                       | Yes   | Yes, `train.step_counts`                                    | Yes      | Yes      | n/a    | Yes     | Yes         | Also holds `iterate_batches`: X07, F09                   |
| M13 | Xavier uniform                     | `ops.py:4-5` `xavier_uniform`                 | Yes   | Yes, `ops.xavier_uniform`                                   | Yes      | Yes\*    | Yes    | Yes\*   | Yes\*       | F05, F14, F20                                            |
| M14 | Parameter initial values           | `transformer.py:20-40` `init_params`          | Yes   | Yes, `transformer.init_params`                              | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M15 | Parameter count                    | `transformer.py:45` `count_params`            | Yes   | Yes, `transformer.count_params`                             | Yes      | Yes      | Yes    | Yes\*   | Partial     | F01 false caller. Equation total: F16, F05               |
| M16 | Linear                             | `ops.py:8` `linear_forward`                   | Yes   | Yes, `ops.linear_forward`                                   | Yes      | Yes      | Yes    | Yes     | Yes         | Test-only, stated. F24                                   |
| M17 | Linear, inline at 7 sites          | `transformer.py:75-77,85,113,116,166`         | Yes   | Yes, `ops.linear_forward`                                   | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M18 | Linear backward                    | `ops.py:11-13` `linear_backward`              | Yes   | Yes, `ops.linear_backward`                                  | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M19 | ReLU                               | `ops.py:17` `relu_forward`                    | Yes   | Yes, `ops.relu_forward`                                     | Yes      | Yes      | Yes    | Yes\*   | Yes         | F21                                                      |
| M20 | ReLU backward                      | `ops.py:20` `relu_backward`                   | Yes   | Yes, `ops.relu_backward`                                    | Yes      | Yes\*    | Yes    | Yes\*   | Yes         | F20, F21                                                 |
| M21 | Softmax                            | `ops.py:23-24` `softmax_forward`              | Yes   | Yes, `ops.softmax_forward`                                  | Yes      | Yes\*    | Yes    | Yes\*   | Yes         | F05, F19                                                 |
| M22 | Softmax backward                   | `ops.py:27` `softmax_backward`                | Yes   | Yes, `ops.softmax_backward`                                 | Yes      | Yes\*    | Yes    | Yes     | Yes\*       | F14, F20                                                 |
| M23 | Cross-entropy                      | `ops.py:30-34` `cross_entropy_forward`        | Yes   | Yes, `ops.cross_entropy_forward`                            | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M24 | Cross-entropy backward             | `ops.py:37-40` `cross_entropy_backward`       | Yes   | Yes, `ops.cross_entropy_backward`                           | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M25 | Layer norm                         | `ops.py:43-47` `layer_norm_forward`           | Yes   | Yes, `ops.layer_norm_forward`                               | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M26 | Layer norm backward                | `ops.py:50-56` `layer_norm_backward`          | Yes   | Yes, `ops.layer_norm_backward`, derivation                  | Yes      | Yes\*    | Yes    | Yes\*   | Yes\*       | F04, F05, F20. Each derivation step checked              |
| M27 | Perplexity                         | `ops.py:60` `perplexity`                      | Yes   | Yes, `ops.perplexity`                                       | Yes      | Yes      | n/a    | Yes     | Yes\*       | F14                                                      |
| M28 | Embedding                          | `transformer.py:53` `embed_forward`           | Yes   | Yes, three embedding concepts                               | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M29 | Embedding backward                 | `transformer.py:56-57` `embed_backward`       | Yes   | Yes, `transformer.embed_backward`                           | Yes      | Yes      | Yes    | Yes     | Yes\*       | PDF numbering not on page: F04                           |
| M30 | Split, merge heads                 | `transformer.py:61-62,66-67`                  | Yes   | Yes, `attention.split`, `attention.merge`, `notation.heads` | Yes      | Yes\*    | Yes    | Yes\*   | Yes\*       | F02 (`config.py:44`), F05, F18                           |
| M31 | Q, K, V                            | `transformer.py:75-77` `attention_forward`    | Yes   | Yes, `attention.projections`                                | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M32 | Scores, mask, weights, values      | `transformer.py:81-84` `attention_forward`    | Yes   | Yes, five attention concepts                                | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M33 | Output projection                  | `transformer.py:85` `attention_forward`       | Yes   | Yes, `attention.output`                                     | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M34 | Output projection backward         | `transformer.py:95-97` `attention_backward`   | Yes   | Yes, `transformer.attention_backward`                       | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M35 | Values and weights backward        | `transformer.py:98-103` `attention_backward`  | Yes   | Yes, `transformer.attention_backward`                       | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M36 | Feed-forward                       | `transformer.py:113-116` `ffn_forward`        | Yes   | Yes, `transformer.ffn_forward`                              | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M37 | Q, K, V parameters, input backward | `transformer.py:104-108` `attention_backward` | Yes   | Yes, `transformer.attention_backward`                       | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M38 | Key bias gradient is zero          | `transformer.py:107` `attention_backward`     | Yes   | Yes, `transformer.attention_backward`                       | Yes      | Yes      | Yes    | Yes     | Yes         | Gap from PDF's "all 38 agree": F04                       |
| M39 | Feed-forward backward              | `transformer.py:119-126` `ffn_backward`       | Yes   | Yes, `transformer.ffn_backward`                             | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M40 | Pre-LN block                       | `transformer.py:133-139` `block_forward`      | Yes   | Yes, `transformer.block_forward`                            | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M41 | Block backward                     | `transformer.py:143-152` `block_backward`     | Yes   | Yes, `transformer.block_backward`                           | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M42 | Length check                       | `transformer.py:156` `forward`                | Yes   | Yes, `transformer.forward`                                  | Yes      | Yes      | n/a    | n/a     | Yes         |                                                          |
| M43 | Blocks in sequence                 | `transformer.py:161-163` `forward`            | Yes   | Yes, `transformer.forward`                                  | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M44 | Final layer norm                   | `transformer.py:165` `forward`                | Yes   | Yes, `transformer.forward`                                  | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M46 | Head                               | `transformer.py:166` `forward`                | Yes   | Yes, `transformer.forward`                                  | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M47 | Head and final-norm backward       | `transformer.py:176-184` `loss_and_grads`     | Yes   | Yes, `transformer.loss_and_grads`                           | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M48 | Reverse order                      | `transformer.py:186-188` `loss_and_grads`     | Yes   | Yes, `transformer.loss_and_grads`                           | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05                                                      |
| M49 | Moments                            | `optim.py:19-20` `Adam.step`                  | Yes   | Yes, `optim.Adam.step`                                      | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M50 | Bias correction                    | `optim.py:22-23` `Adam.step`                  | Yes   | Yes, `optim.Adam.step`                                      | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M51 | Update with decoupled decay        | `optim.py:25-30` `Adam.step`                  | Yes   | Yes, `optim.Adam.step`                                      | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M52 | Learning-rate schedule             | `optim.py:33-37` `lr_schedule`                | Yes   | Yes, `optim.lr_schedule`                                    | Yes      | Yes      | n/a    | Yes     | Yes         |                                                          |
| M53 | Global-norm clip                   | `optim.py:41-50` `clip_grad_norm`             | Yes   | Yes, `optim.clip_grad_norm`                                 | Yes      | Yes      | Yes    | Yes     | Yes         |                                                          |
| M54 | Greedy guard                       | `generate.py:8-9` `sample_next`               | Yes   | Yes, `sampling.greedy`                                      | Yes      | Yes      | Yes    | Yes     | Yes         | Reference says no example: F08                           |
| M55 | Temperature                        | `generate.py:11` `sample_next`                | Yes   | Yes, `sampling.temperature`                                 | Yes      | Yes      | Yes    | Yes     | Yes         | Reference says no example: F08                           |
| M56 | Top-k                              | `generate.py:13-16` `sample_next`             | Yes   | Yes, `sampling.top_k`                                       | Yes      | Yes      | Yes    | Yes     | Yes\*       | F08, F22                                                 |
| M57 | Top-p                              | `generate.py:18-24` `sample_next`             | Yes   | Yes, `sampling.top_p`                                       | Yes      | Yes      | Yes    | Yes     | Yes\*       | F08, F22                                                 |
| M58 | Sample                             | `generate.py:26` `sample_next`                | Yes   | Yes, `sampling.choice`                                      | Yes      | Yes\*    | Yes    | Yes\*   | Yes         | F05, F08, F20                                            |
| M59 | Generation loop                    | `generate.py:31-33` `generate`                | Yes   | Yes, `generate.generate`                                    | Yes      | Yes\*    | Yes    | Yes\*   | Yes         | F05, F08, F20                                            |
| M60 | Central difference                 | `gradcheck.py:5-12` `numerical_grad`          | Yes   | Yes, `gradcheck.numerical_grad`                             | Yes      | Yes      | n/a    | Yes     | Partial     | F02 false pipeline claim                                 |
| M61 | Relative error                     | `gradcheck.py:17` `relative_error`            | Yes   | Yes, `gradcheck.relative_error`                             | Yes      | Yes      | n/a    | Yes     | Yes\*       | F04, F11                                                 |
| M62 | Per-parameter check                | `gradcheck.py:23-31` `check_params`           | Yes   | Yes, `gradcheck.check_params`                               | Yes      | Yes      | Yes    | Yes\*   | Yes         | F05. Example is in the next section, stated              |
| X01 | Read, clean and join the corpus    | `data.py:6-12` `load_corpus`                  | Yes   | Yes, Data Pipeline prose                                    | n/a      | Yes      | n/a    | Yes     | Yes\*       | No inventory row: F09. F27                               |
| X02 | Text to id array                   | `data.py:16` `encode_corpus`                  | Yes   | Yes, Data Pipeline prose                                    | n/a      | Yes      | Yes    | n/a     | Yes         | No inventory row: F09                                    |
| X03 | Replay merges on pre-tokens        | `tokenizer.py:64-73` `apply_merges`           | Yes   | Yes, `tokenizer.Tokenizer.encode`                           | Partial  | Yes      | n/a    | Yes     | Yes         | Equation covers the lookup only; prose covers the replay |
| X04 | Stop when no pair is left          | `tokenizer.py:40-41,50-51` `train_bpe`        | Yes   | Yes, Tokenization prose                                     | n/a      | Yes      | n/a    | n/a     | Yes         | No inventory row: F09                                    |
| X05 | Count, then train                  | `tokenizer.py:77` `train_bpe_on_text`         | Yes   | Partial, named only                                         | n/a      | Partial  | n/a    | n/a     | Yes         | No line link of its own. F09                             |
| X06 | Decode                             | `tokenizer.py:93-96` `Tokenizer.decode`       | Yes   | Yes, `tokenizer.Tokenizer.encode`                           | Yes      | Yes      | n/a    | Yes     | Yes         | No inventory row: F09                                    |
| X07 | Shuffle and slice into batches     | `data.py:37-43` `iterate_batches`             | Yes   | Yes, `data.iterate_batches`                                 | Yes      | Yes      | Yes    | Yes\*   | Yes\*       | Filed under M12 only: F09. F05, F22                      |
| X08 | Optimizer state and step counter   | `optim.py:8-11,14` `Adam`                     | Yes   | Yes, AdamW page                                             | n/a      | Yes      | Yes    | Yes     | Yes         | No inventory row: F09                                    |
| X09 | Print `ln V`                       | `train.py:52` `main`                          | Yes   | Yes, Logging lines                                          | n/a      | Yes      | n/a    | Yes     | Yes         | No inventory row: F09                                    |
| X10 | Save a checkpoint                  | `checkpoint.py:17-30` `save_checkpoint`       | Yes   | Yes, Checkpoints & Tests                                    | n/a      | Yes      | Yes    | Yes\*   | Yes         | No equation, stated. F05                                 |
| X11 | Load a checkpoint, rebuild `stoi`  | `checkpoint.py:33-46` `load_checkpoint`       | Yes   | Yes, Checkpoints & Tests                                    | n/a      | Yes      | Yes    | n/a     | Yes         | No equation, stated                                      |
| X12 | Config assertions                  | `config.py:42-43` module                      | Yes   | Partial, in the listing only                                | Partial  | Yes      | n/a    | n/a     | Partial     | F27. `d mod H = 0` is on Notation                        |
| X13 | Sample printed each epoch          | `train.py:72-75` `main`                       | Yes   | Yes, Training Loop                                          | n/a      | Yes      | n/a    | Yes     | Yes\*       | F13                                                      |
| X14 | Prompt loop                        | `talk.py:13-27` `main`                        | Yes   | Yes, Checkpoints & Tests                                    | n/a      | Yes      | n/a    | n/a     | Yes         |                                                          |
| X15 | Start of the gradient dictionary   | `transformer.py:174` `loss_and_grads`         | Yes   | Yes, Full Model prose                                       | n/a      | Yes      | Yes    | n/a     | Yes         | No inventory row: F09                                    |

Reading the table: every operation is implemented and every one has a verdict in every column. That is 76 rows, 61 from the inventory and 15 added. Three explanation cells are `Partial` (M15, M60, X12), and no equation in the 61 inventory rows is wrong.

## 5. Findings

Each finding has a severity, its evidence and a proposed fix. No fix was applied.

### High

**F01. Parameter count names the wrong caller.**

- Where: `content/mathematics/notation-and-shapes.mdx:191`, rendered at `/mathematics/notation-and-shapes/`, section Parameter count. Root cause in `components/Trace.tsx`, where `callers` is a `flatMap` over every `source`, and `data/concepts/transformer.count_params.yaml`, whose `source` also lists `transformer.init_params`.
- Evidence: the Used by block says `Called by: train.main`. The same section says "only a test calls it". `data/generated/call-graph.json` gives `transformer.count_params` `production: []`. The Tested by list has the same fault, since it lists tests that use `init_params`.
- Fix: drop `init_params` from that concept's `source`, or give `Trace` a way to read callers from only the primary symbol. Keep the `init_params` line links.

**F02. A generated note says the pipeline computes something it does not.**

- Where: `lib/equation-links.ts:102-103`, fed by `data/links/gradcheck.numerical_grad.yaml` and `data/links/notation.heads.yaml`.
- Evidence, Gradient Checking, section Central differences: "Inline equivalent: the pipeline does not call gradcheck.numerical_grad. It computes the same formula in place at gradcheck.py:31." That line is in `check_params`, which only `tests/test_gradcheck.py` calls. The page's own opening says `gradcheck.py` is not part of the pipeline.
- Evidence, Notation & Shapes, section Heads: "the pipeline calls transformer.attention_forward, and also computes the same formula in place, with no call, at transformer.py:94, config.py:44." `config.py:44` is `head_dim`, and the same site says nothing reads it.
- Fix: mark these two links with a role other than `inline`, or have `inlineEquivalentNote` skip sites that `train.main` and `talk.main` cannot reach.

### Medium

**F03. Concept `notes` are never shown.**

- Where: `lib/concepts.ts:125` requires the field, and no component reads it. Found with a scratch script over all 53 notes: `53 notes in total, 52 not found verbatim in any built page`.
- Effect: anyone writing a note will believe a reader sees it. Many of these notes are PDF-versus-code differences, so F04 follows from this.
- Fix: render `notes` in `ConceptCard` or `Trace`, then remove the ones that only record how the PDF reference was found. Or delete the field and move what matters into page prose.

**F04. PDF and code disagree, and the topic's page does not say so.** Each row below is recorded on Sources & Discrepancies, so the difference is disclosed once. It is missing from the page where a reader meets the topic. A search of the prose of each topic page for "PDF" found mentions only on Data Pipeline, Full Model, Tokenization, Training Loop and Sampling. The Used by block on other pages gives a PDF section number, and says nothing about a difference.

| Difference (from `contract/DISCREPANCIES.md`)                | Page where it matters                                                                                               | Unrendered note                                                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Corpus is `data/corpus.csv`, PDF says `data/corpus.txt`      | `content/mathematics/data-pipeline.mdx` Text to ids                                                                 | none                                                                                              |
| `max_rows` limit exists only in code                         | `content/mathematics/data-pipeline.mdx` Text to ids                                                                 | none                                                                                              |
| `make_windows` returns DataFrames, PDF says NumPy arrays     | `content/mathematics/data-pipeline.mdx` Windows                                                                     | `data.make_windows`                                                                               |
| PDF's Eq. 18 has a different but equal form                  | `content/mathematics/data-pipeline.mdx` Window count                                                                | `data.window_count`                                                                               |
| Positions number from 0, PDF's Eq. 40 from 1                 | `content/mathematics/embeddings.mdx` Embedding backward                                                             | `transformer.embed_backward`                                                                      |
| PDF's layer-norm property `sum dx_j xhat_j = 0` is not exact | `content/mathematics/layernorm.mdx` backward                                                                        | none. Measured here: about `7.0e-07` and `8.2e-04` on two rows, while `sum dx_j` is about `1e-16` |
| PDF's error bands are not in the repository                  | `content/testing/gradient-checking.mdx` Relative error                                                              | `gradcheck.relative_error`                                                                        |
| PDF says all 38 tensors agree; the test skips `.bk`          | `content/testing/gradient-checking.mdx`                                                                             | none                                                                                              |
| Overfit test is 200 steps, PDF says 300                      | `content/code/checkpoints-tests.mdx` test list                                                                      | none                                                                                              |
| Only the left side of Eq. 37 is in the code                  | `content/testing/gradient-checking.mdx`                                                                             | `gradcheck.numerical_grad`                                                                        |
| PDF also reaches the loss gradient through softmax backward  | `content/mathematics/softmax-cross-entropy.mdx`                                                                     | `ops.cross_entropy_backward`                                                                      |
| PDF function and argument names differ                       | Softmax, LayerNorm, ReLU, Tokenization pages                                                                        | `ops.*`, `tokenizer.*`                                                                            |
| Case and regex                                               | `content/mathematics/tokenization.mdx:26` says "its treatment of case differs" without saying how the PDF treats it | `tokenizer.count_words`                                                                           |

- Fix: rendering the notes (F03) covers most rows. Add a short note for the rows marked none.

**F05. Illustrative Example is defined one way and used another.**

- Rule: `CLAUDE.md` line 21 and `content/getting-started/introduction.mdx:68-70`: Illustrative means hand-picked numbers, and Actual Model Output means produced by running the repository.
- Evidence: the page subagent found 22 panels labelled Illustrative whose own caption says values came from running real code. I read those captions, and added one (Logging lines, whose explanation is in the sentence above it). They include `What init_params returned`, `Two epochs of iterate_batches`, `The residual stream`, `The whole forward pass`, `A batch of 3 windows`, `1000 draws from one distribution`, `Worst relative error per name`, and `Forward, with the real config`. In the tiny-model panels the parameters are random draws from `init_params`, so they are not hand-picked either.
- Mitigation: every panel carries a caption saying it is not the trained model, so no reader is told these are learned values. Task 04 chose this reading on purpose: hand-picked input means Illustrative.
- Fix, your decision: either keep the current practice and change the definition to "hand-picked inputs, including small models with random starting values, computed by the real code or its tested mirror", or add a third label. The first is smaller and only needs one sentence on Introduction and one in `CLAUDE.md`.

**F06. Run values with no label on Sources & Discrepancies.**

- Where: `content/reference/sources-discrepancies.mdx`, rendered from `data/derived/discrepancies.json`, sections Formulas and behaviour, and Repository state and the checkpoint.
- Evidence: `63,871 parameters at V=575`, `about 3.27e-6`, `225,099 training windows, 14,069 steps per epoch and 281,380 total steps`, `0.9078093812857`, `17,590 total steps`, `V=564`. One row in that section carries the label, and these do not.
- Fix: label them Actual Model Output in `contract/DISCREPANCIES.md`, then run `bun run sync:derived`.

**F07. A public page publishes internal notes and a local path.**

- Where: `content/reference/sources-discrepancies.mdx:16-50`, from the git-tracked `data/derived/discrepancies.json`, a snapshot of the git-ignored `contract/DISCREPANCIES.md`.
- Evidence: `It is outside the repository, at C:\Users\upahar\Downloads\`; `The human decided all five on 2026-11-05`; `Found in Task 11; the PDF was not reopened (D4)`; `plain pytest was not tried in this session`; `Do not document this comparison as a feature`; `Website pages must not reuse a symbol for two meanings on one page`. It also cites `See section 7` and `See HYPERPARAMETERS.md, section 5`, while the page's headings have no numbers and that file is not on the site.
- Fix, your decision: write a reader-facing version of this page and keep the working notes in `contract/`. At the least, remove the local path.

**F08. The Reference table says six concepts have no example, and they do.**

- Where: `lib/reference.ts:87-90` looks for a tag ending in `Example` that has a `concept=` attribute. `components/MathReference.tsx:85` prints the result. `content/generation/sampling.mdx:44,88,142,199,241` and `content/generation/autoregressive-generation.mdx:38` use example tags with no `concept=`.
- Evidence: `/reference/mathematics/` shows "No numerical example" for Greedy guard, Temperature, Top-k, Top-p, Random choice and Autoregressive loop. Each has a labelled Tiny numerical example section.
- Fix: add `concept="..."` to those six tags.

**F09. "61 of 61 operations documented" overstates the coverage.**

- Where: `content/reference/mathematics.mdx:7` says the inventory "lists every operation the code implements". `contract/MATH_INVENTORY.md` gives the operations X01 to X15 above no row of their own. `data/concepts/data.iterate_batches.yaml` declares `inventory: [M12]`, which is the step count in `train.py`, because batching has no row of its own.
- Effect: nothing is undocumented, since those operations are on pages. But the number measures the inventory, and the sentence claims more.
- Fix: add rows for the operations with arithmetic or a data transformation (at least `load_corpus`, `encode_corpus`, `apply_merges`, `Tokenizer.decode`, `iterate_batches`), run `sync:derived`, and repoint `iterate_batches`. `contract/` is yours to change; this task did not change it.

**F10. Home says "Every step the code runs", and steps are missing.**

- Where: `app/(docs)/page.tsx:133`, with steps in `data/pipeline.ts`.
- Evidence: the 19 steps have nothing for `split_ids` (`train.py:43`), shuffling and batching (`train.py:60`), validation (`train.py:70`) or saving (`train.py:78`). Step 15 is called "Softmax and cross-entropy", while `cross_entropy_forward` does not call `softmax_forward`.
- Fix: say "The main steps", or add the four steps. Rename step 15 to Cross-entropy.

**F11. "Used by, in the pipeline" lists test-only code.**

- Where: `components/CodeExplorer.tsx:401,527`, reading `production` from `scripts/extract_repo.py:477-480`, where `production` means any caller that is not a test.
- Evidence: `/code/gradcheck/` shows `relative_error` with "Used by, in the pipeline: gradcheck.check_params" directly beside "Reached from: Neither train.main nor talk.main".
- Fix: filter that list by reachability, or rename the label "Used by, outside tests".

**F12. Math ↔ Code does not exist, and Home points at it.**

- Where: `data/nav.ts:81`, `data/home.ts:55`. No `content/code/math-code.mdx`.
- Evidence: the sidebar item is disabled on every page, and Home shows "Where is it implemented? (page not written yet)". This is one of the eight questions on Home. The 166 JSON files under `out/math-code/` are in use, by the equation navigator.
- Fix, your decision: write the page, or point that question at Code Explorer and remove the nav item. Already logged as open in PROGRESS.md.

### Low

**F13. Two statements about runs read as contradictory.** `content/getting-started/how-this-llm-works.mdx:109` says "no run of the pinned commit and its corpus is available", while Actual Model Output panels begin "Commit 66da45b. Checkpoint model.npz ..." (`lib/actual.ts:203`). Both are true and the panels' disclosure follows. The leading "Commit 66da45b" can still read as "trained at that commit". Fix: start with "Exported at commit 66da45b".

**F14. Absolute words that ignore tests.** `content/mathematics/linear-relu-initialisation.mdx:252` ("only `train.main` calls `init_params`", while tests also call it), `content/mathematics/softmax-cross-entropy.mdx:102` ("Only attention calls it", while `test_softmax_backward_rows_sum_to_zero` does too), `data/concepts/ops.perplexity.yaml:8` ("nothing else reads it"). Fix: add "in the pipeline", as `content/code/checkpoints-tests.mdx:75` already does.

**F15. "The code uses NumPy only for arrays."** `content/getting-started/how-this-llm-works.mdx:9`. Line 45 of the same page says windows are `pandas` DataFrames, and `data.py:31-32` confirms it. This repeats the PDF's "NumPy only" claim, which `contract/DISCREPANCIES.md` section 2 records as wrong. Fix: "NumPy for arrays and `pandas` for the corpus and windows".

**F16. Introduction promises more than the site does.** All in `content/getting-started/introduction.mdx`:

- Line 41, "Each one has its equation, a small worked example, its tensor shapes and its code": scalar operations such as perplexity have no tensor shapes.
- Line 45, "collects every equation": the total `N = 65 V + 26,496` from `components/ArchitectureTables.tsx` is not in that table's records.
- Line 60, "Every page also has previous and next links": Home has none, and the first and last pages have one.
- Line 64, "Each equation has a Show code button": that same total has none. The line numbers on Sources & Discrepancies are also not generated, as that page itself says.

Fix: soften these to "most".

**F17. Stale sentences.** `components/PipelineDiagram.tsx:115`, "Plain names have no page yet", when every step name is now a link. `content/mathematics/attention.mdx:22`, "when an export exists", when the export is on that page. Fix: remove or reword.

**F18. Duplicated names in Used by.** `components/Trace.tsx` does not remove duplicates. `/mathematics/notation-and-shapes/` section Heads shows "transformer.attention_backward, transformer.attention_forward, transformer.attention_backward, transformer.attention_forward". Fix: wrap in `new Set`.

**F19. A test missing from the math links.** `data/links/ops.softmax_forward.yaml` has no `role: test` link, although `test_softmax_values_and_stability` (`tests/test_ops.py:28-33`) tests the function. So the Mathematics used table on `/code/test_ops/` lists "Softmax, backward" and not "Softmax". Fix: add the link.

**F20. Link roles that do not fit.** The line numbers are right, and the role wording is loose:

- `role: implements` on steps the code does not compute: `softmax.jacobian` (`ops.py:27`), `relu.derivative` (`ops.py:20`), `xavier.variance` (`ops.py:4`), and the `ln.derive.*` steps. The pages do say the code never forms these, so the page and the "Computes it" table disagree slightly.
- `role: call` on lines that are not a call of the concept: `sampling.choice` on `train.py:37` and `talk.py:9`, which create the generator; `generate.window` on the assert at `transformer.py:156`.

Fix: add a role such as `derives` and one such as `context`.

**F21. Example attributed to a test, in another shape.** ReLU and ReLU backward on `/mathematics/linear-relu-initialisation/` say "Hand-picked numbers from test_relu" and show `[[-1, 2, 0, 3]]`, which is 2-D. `tests/test_ops.py:23` uses a 1-D array. The values are the same. Fix: "The values of test_relu, as one row".

**F22. General reasoning without "In general".** `CLAUDE.md` line 16 requires the label. Candidates in concept `why` fields: `tokenizer.train_bpe` ("common words take fewer tokens"), `sampling.top_k` ("stops the choice from landing on unlikely tokens"), `sampling.top_p` ("removes the unlikely tail"), `data.iterate_batches` ("would be slow ... would be noisy", which `train.batch_loss_and_grads` correctly attributes to a comment in `data.py:35`), `optim.Adam.step` ("its size varies a lot"), `attention.mask` ("it could read the answer"). None is wrong, but none traces to a measurement. See section 7.4.

**F23. "This is the only page that describes them."** `content/reference/not-in-this-model.mdx:9`. Other pages state absences on purpose: "no chat format, no message history and no stopping token" (`how-this-llm-works.mdx:121`), "not tied to `tok_emb`" (`full-model.mdx:173`), and the "What the model is and is not" section. The check only looks for the words it lists. Fix: "the only page that lists them".

**F24. Test-only mathematics on pipeline pages.** Decision D2 says "in a clearly labelled testing section, and never in the pipeline flow". Parameter count is on Notation & Shapes, and "Checking this backward numerically" panels are on pipeline pages. All are labelled "A test-time idea". Fix: either move them, or relax D2 to match. The labels already do what D2 was for.

**F25. What the checks do not cover.**

- `verify:content` check 4 looks at 3 `NumericExample` tags. About 70 other example components are not checked for a label by that check. They were found labelled in this audit (section 7.3).
- Check 5 skips 10 shape stages with no recorded tensor.
- `tests/e2e/` is stale, and a known spec expects old headings. This is a standing decision, not a defect.
- GitHub permalinks were checked for form, not opened.
- Fix: extend check 4 to every tag ending in `Example`. Delete `tests/e2e/` if it will not be run again.

**F26. Stale project notes.** `CLAUDE.md` line 6 says the repository was last inspected at `c17aa75`, and the pin is `66da45b`. `contract/MATH_INVENTORY.md` M05 cites `tokenizer.py:87-90`, and the replay it describes is in `apply_merges`, lines 64-73. Fix: update both.

**F27. Small behaviours not documented.** All from the code-only list, none wrong on a page:

- Config assertions `config.py:42-43` are only in the listing, while `contract/PDF_VS_CODE.md` E001 says to document them.
- A negative temperature is also greedy (`generate.py:8`).
- Text such as `NA` or `null` in the corpus is dropped, because `pandas` parses it as missing (`data.py:6,11`).
- Intermediate merged tokens can stay in the vocabulary without occurring in the ids.

Fix: add one line each to the relevant page, if wanted.

## 6. Ten rows checked by hand

For each row I compared the concept's equation with the code line, then recomputed the rendered example or a fresh case. The script is `spot_check.py` in the session scratchpad, not kept in the repository.

| Row | Code read              | Rendered value on the page                                                   | Hand or script result                                                                                                  | Verdict |
| --- | ---------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| M09 | `data.py:28`           | `N=10, T=4, k=3` gives 2, and `k=6` gives 1                                  | `ceil(6/3)=2`, `ceil(6/6)=1`. Formula and PDF form match `make_windows` over a grid of `N`, `T`, `k`: `mismatches 0`   | Correct |
| M21 | `ops.py:23-24`         | `[2,1,0]` gives `[0.6652, 0.2447, 0.0900]`                                   | `softmax [2,1,0] [0.6652 0.2447 0.09]`, and `-inf` gives exactly 0                                                     | Correct |
| M24 | `ops.py:37-40`         | `(0.6652 - 1) / 2 = -0.1674`                                                 | `(P - E_y) / T` matches on fresh logits                                                                                | Correct |
| M26 | `ops.py:50-56`         | `dX = [0.4391, -0.2482, -0.4009, 0.2100]`                                    | By hand: `0.5345 * (1 - 0.75 + 1.069 * 0.5345) = 0.4391`. Matches central differences                                  | Correct |
| M29 | `transformer.py:56-57` | `tok_emb[2] = [1,0] + [2,2] = [3,2]`, position rows equal `dX0`              | Repeated id sums, rows at `T` and beyond are 0                                                                         | Correct |
| M32 | `transformer.py:81-84` | Row "cat": scores `0.95, -0.78` give `0.849, 0.151`; first row gives `1.000` | `e^1.73 = 5.64`, `5.64 / 6.64 = 0.849`. Written out by hand on a fresh model: equal, shape `(H,T,T)`, upper triangle 0 | Correct |
| M38 | `transformer.py:107`   | Largest entry `2.9e-17`, "round-off around 0"                                | `b0.bk` largest entry `1.73e-17` on a different tiny model, the same order                                             | Correct |
| M52 | `optim.py:33-37`       | Step 1 gives `0.00002`                                                       | `1e-3 * 1 / 50 = 2e-5`. Step 50 is exactly `lr_max`, step 51 is lower, last step is `lr_min`                           | Correct |
| M53 | `optim.py:41-50`       | Tensors of length 3 and 4, norm 5, factor 0.2, norm after 1                  | Returns 5.0 before clipping, scales both by one factor, leaves a small gradient alone                                  | Correct |
| M57 | `generate.py:18-24`    | `0.5, 0.3, 0.15` kept, `0.5263, 0.3158, 0.1579`. Exact hit at 0.5 keeps 2    | Same values. Crossing token kept. `cum` exactly at `top_p` stops at that token                                         | Correct |

The same script also passed on M03, M04, M13, M15 (`63871`, `38` tensors), M22, M23, M25, M28, M30, M40, M51, M54, M56, M60 and M61.

## 7. Requested lists

### 7.1 Undocumented functions

None are wholly undocumented. All 58 root definitions are named on a page, and `verify:content` reports `Uncovered functions: none.`

Thin coverage:

- `tokenizer.train_bpe_on_text`: named, with no line link of its own (X05).
- `tokenizer.Tokenizer.__init__`: listed as skipped with a reason, "There is nothing to compute".
- `optim.Adam` and `tokenizer.Tokenizer`, as classes: documented through their methods.

### 7.2 Stale or ambiguous links

- **Broken:** none. `verify:site` reports 5147 internal links and 3082 anchors with no problems. The page subagent separately found 0 missing targets and 0 link text mismatches.
- **Commit:** every pinned GitHub link uses `66da45b`, which still matches the code.
- **Unpinned:** the "GitHub" header link goes to the repository root. This looks intended.
- **Not opened:** all GitHub links, plus the two external links on Introduction (`w3cj/how-llms-work`, the YouTube video).
- **Going nowhere:** Math ↔ Code and the Home question (F12).
- **Ambiguous:** the link roles in F20, and the misleading caller list in F01 and F11.
- **Not linked anywhere:** `/design/`, `/design/components/` and `/design/interactive/`. These are internal sample pages, noindex by design.

### 7.3 Examples missing a label

- **Panels:** the page subagent found 98 panels. It flagged one as having no label: Checking every parameter on Gradient Checking. I checked this, and it is **not a defect**. The section says "The example is in Reading a failure", and `<CheckParamsExample>` (`content/testing/gradient-checking.mdx:200`) is labelled.
- **Plain numbers:** the unlabelled run values are in F06.
- **Mixed labels:** see F05.
- **Other plain values:** test expectations such as "N=3000 gives 2684 train" on Checkpoints & Tests, and sentences that follow a labelled panel on the same topic, such as "The function returns 1.94512596". These rely on the panel or the test, and I did not count them as defects.

### 7.4 Claims without code support

No claim contradicts the code except F01, F02, F10, F11, F14 and F15.

These have no code behind them:

- General reasoning listed in F22.
- Credits on `content/getting-started/introduction.mdx:27`: nothing in the repository refers to `how-llms-work` or the video. Progress notes record that you approved this content, so this is information only.
- Attention's "In general, these three names are explained with a question", which is correctly labelled.

### 7.5 PDF and code disagree, and the page does not say so

See the table in F04.

### 7.6 Content about techniques the code does not use

- **On the intended page:** `content/reference/not-in-this-model.mdx` lists 24 techniques.
- **Everywhere else:** a separate search over `content/` and `data/concepts/` found only statements of absence, each correct: no chat format, not tied to `tok_emb`, no autograd. No page presents an absent technique as something this project does. `verify:content` check 7 agrees: `48 words for 24 contract rows, over 313 site files and 19 repository files`.
- **General background:** Eq. 61 on Data Pipeline is labelled "In general", and says the code never computes it. Eq. 62 on Training Loop says the same.
- **Discrepancies page:** Sources & Discrepancies quotes techniques from the PDF, which is what it is for.

## 8. Decisions needed from you

1. **F05:** keep Illustrative Example for hand-picked inputs computed by real code, and change the definition? Recommended.
2. **F07:** rewrite Sources & Discrepancies for readers, or only remove the local path? I recommend a rewrite, with the path removed either way.
3. **F12:** write Math ↔ Code, or remove it and repoint the Home question? I recommend repointing, since each equation already has Show code.
4. **Still open from Task 15:** `data/generated/actual/` holds corpus excerpts and is tracked in git. Should they stay public?
5. **F24:** relax D2 to match the current panels, or move them? I recommend relaxing it.

## 9. Assumptions

- "Operation" means a distinct mathematical step or data transformation, as in `contract/MATH_INVENTORY.md`. Plain control flow from the code-only list is not given its own row.
- Rows M34, M35, M37 and M38 share one concept, so their example verdicts come from one panel set.
- Severity is my judgement, using the meanings in section 1.
- Running `bun run build` rewrites the git-ignored `out/`, `.next/` and `public/search-index.json`. I treated this as allowed by "re-run every verification script". `git status --short` was empty afterwards, before this file existed.

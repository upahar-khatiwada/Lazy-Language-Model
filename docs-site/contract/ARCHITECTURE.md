# Architecture

Pinned commit: `c17aa75`. Derived from `transformer.forward`, `transformer.loss_and_grads`, `train.main`, `generate.generate` and `talk.main`. Line numbers were read from the pinned code; website pages must generate theirs.

Symbols: `T` sequence length, `d` = `emb_dim`, `H` = `num_heads`, `dh` = `d // H`, `F` = `ff_dim`, `V` = vocabulary size, `L` = `num_layers`, `Tmax` = `context_len`.

## 1. Model summary

A decoder-only transformer with pre-layer-norm blocks. It uses NumPy with hand-written backward functions.

**The model has no batch dimension.** `forward` and `loss_and_grads` process one sequence at a time, with shapes `(T,)` and `(T,d)`. Batching is a Python loop in `train.batch_loss_and_grads`.

Values from `config.py`: `d=32`, `H=2`, `dh=16`, `F=128`, `L=2`, `Tmax=32`, training and generation window `T=16`.

## 2. Parameters

Names are exactly as `init_params` creates them (`transformer.py` lines 17-42), in insertion order. `{l}` is the block index, starting at 0.

| Key | Shape | Initial value | AdamW decay |
|---|---|---|---|
| `tok_emb` | `(V, d)` | Xavier uniform | yes |
| `pos_emb` | `(Tmax, d)` | Xavier uniform | yes |
| `b{l}.ln1_g` | `(d,)` | ones | no |
| `b{l}.ln1_b` | `(d,)` | zeros | no |
| `b{l}.wq` | `(d, d)` | Xavier uniform | yes |
| `b{l}.bq` | `(d,)` | zeros | no |
| `b{l}.wk` | `(d, d)` | Xavier uniform | yes |
| `b{l}.bk` | `(d,)` | zeros | no |
| `b{l}.wv` | `(d, d)` | Xavier uniform | yes |
| `b{l}.bv` | `(d,)` | zeros | no |
| `b{l}.wo` | `(d, d)` | Xavier uniform | yes |
| `b{l}.bo` | `(d,)` | zeros | no |
| `b{l}.ln2_g` | `(d,)` | ones | no |
| `b{l}.ln2_b` | `(d,)` | zeros | no |
| `b{l}.ff1_w` | `(d, F)` | Xavier uniform | yes |
| `b{l}.ff1_b` | `(F,)` | zeros | no |
| `b{l}.ff2_w` | `(F, d)` | Xavier uniform | yes |
| `b{l}.ff2_b` | `(d,)` | zeros | no |
| `lnf_g` | `(d,)` | ones | no |
| `lnf_b` | `(d,)` | zeros | no |
| `head_w` | `(d, V)` | Xavier uniform | yes |
| `head_b` | `(V,)` | zeros | no |

`head_w` is a separate matrix. It is not tied to `tok_emb`.

**Actual Model Output** (read from `checkpoints/model.npz`; see section 8 for its provenance): `V = 575`, 38 tensors, 63,871 scalar parameters. Every shape matches the table. Decay applies to 15 tensors (62,400 scalars) and not to 23 tensors (1,471 scalars).

## 3. Forward pass, in code order

`transformer.forward` (lines 155-168):

| Step | Code | Output shape | Line |
|---|---|---|---:|
| 0. Check length | `assert len(ids) <= cfg.context_len` | n/a | 156 |
| 1. Embeddings | `X = embed_forward(params, ids)` | `(T, d)` | 158 |
| 2. Blocks | `X, c = block_forward(X, params, l, cfg)` for `l` in `range(num_layers)` | `(T, d)` | 161-163 |
| 3. Final layer norm | `Xf, c_lnf = layer_norm_forward(X, lnf_g, lnf_b, cfg.ln_eps)` | `(T, d)` | 165 |
| 4. Head | `logits = Xf @ head_w + head_b` | `(T, V)` | 166 |

### Embeddings

`embed_forward` (line 53): `tok_emb[ids] + pos_emb[:len(ids)]`.

### One block

`block_forward` (lines 131-139) is pre-LN: each residual branch reads a normalized copy, and the residual adds to the unnormalized value.

| Step | Code | Line |
|---|---|---:|
| 1 | `Xn = layer_norm_forward(X, ln1_g, ln1_b, ln_eps)` | 133 |
| 2 | `att = attention_forward(Xn, ...)` | 134 |
| 3 | `X1 = X + att` | 135 |
| 4 | `Xn2 = layer_norm_forward(X1, ln2_g, ln2_b, ln_eps)` | 136 |
| 5 | `ff = ffn_forward(Xn2, ...)` | 137 |
| 6 | return `X1 + ff` | 139 |

### Attention

`attention_forward` (lines 70-86):

| Step | Code | Shape | Line |
|---|---|---|---:|
| Query | `Q = Xn @ wq + bq` | `(T, d)` | 75 |
| Key | `K = Xn @ wk + bk` | `(T, d)` | 76 |
| Value | `V = Xn @ wv + bv` | `(T, d)` | 77 |
| Split | `split_heads` on each | `(H, T, dh)` | 79 |
| Scores | `S = Qh @ Kh^T / sqrt(dh)` | `(H, T, T)` | 81 |
| Causal mask | `-inf` where `np.triu(..., 1)` is true | `(H, T, T)` | 82 |
| Weights | `A = softmax_forward(S)` | `(H, T, T)` | 83 |
| Combine | `O = merge_heads(A @ Vh)` | `(T, d)` | 84 |
| Output | `out = O @ wo + bo` | `(T, d)` | 85 |

Note: inside this function `V` is the value matrix, not the vocabulary size.

### Feed-forward

`ffn_forward` (lines 112-116): `Z = Xn @ ff1_w + ff1_b` with shape `(T,F)`, then `A = relu_forward(Z)`, then `A @ ff2_w + ff2_b` with shape `(T,d)`. Here `A` is the ReLU output, not attention weights.

## 4. Backward pass, in code order

`transformer.loss_and_grads` (lines 171-190):

| Step | Code | Line |
|---|---|---:|
| 1 | `logits, cache = forward(params, ids, cfg)` | 172 |
| 2 | `loss, probs = cross_entropy_forward(logits, targets)` | 173 |
| 3 | `grads` = zeros with each parameter's shape | 174 |
| 4 | `dZ = cross_entropy_backward(probs, targets)` | 175 |
| 5 | `dXf, dW, db = linear_backward(cache["Xf"], head_w, dZ)`; fills `head_w`, `head_b` | 176-179 |
| 6 | `dX, dg, db = layer_norm_backward(dXf, cache["lnf"], lnf_g)`; fills `lnf_g`, `lnf_b` | 181-184 |
| 7 | `dX = block_backward(...)` for `l` in `reversed(range(num_layers))` | 186-187 |
| 8 | `embed_backward(dX, ids, grads)` | 188 |

`block_backward` (lines 141-152), reverse of the block:

| Step | Code | Line |
|---|---|---:|
| 1 | `dXn2 = ffn_backward(dY, ...)` | 143 |
| 2 | `dX1_branch, dg, db = layer_norm_backward(dXn2, ln2 cache, ln2_g)` | 144-146 |
| 3 | `dX1 = dY + dX1_branch` | 147 |
| 4 | `dXn = attention_backward(dX1, ...)` | 148 |
| 5 | `dX_branch, dg, db = layer_norm_backward(dXn, ln1 cache, ln1_g)` | 149-151 |
| 6 | return `dX1 + dX_branch` | 152 |

`attention_backward` (lines 92-109) and `embed_backward` (lines 55-57) are itemized in `MATH_INVENTORY.md`.

## 5. Training pipeline

`train.main` (lines 36-81):

1. `rng = np.random.default_rng(config.seed)` (37)
2. `load_corpus` (40)
3. `build_tokenizer` (41)
4. `encode_corpus` (42)
5. `split_ids` (43), before windowing, so no window crosses the split
6. `make_windows` for train (44) and validation (45), as `pandas` DataFrames
7. `ModelConfig(...)` (48-49); `ln_eps` is not passed, so the dataclass default applies
8. `init_params` (50)
9. `Adam(...)` (51)
10. Prints `V` and `ln V` (52)
11. `steps_per_epoch = ceil(len(tr_x) / batch_size)`; `total_steps = epochs * steps_per_epoch` (54-55)
12. For each epoch (59), for each batch from `iterate_batches` (60):
    1. `batch_loss_and_grads` (61): calls `loss_and_grads` per window, divides loss and gradients by `B`
    2. `clip_grad_norm` (62)
    3. `step += 1` (63), so the schedule sees steps starting at 1
    4. `lr_schedule` (64-65)
    5. `opt.step` (66)
    6. Logs when `step == 1 or step % 50 == 0` (67-68)
13. After each epoch: `validation_loss` (70), prints loss and `perplexity` (71), generates a sample from the first 3 tokens of the first validation window, with 30 new tokens (72-75)
14. `save_checkpoint` (78)
15. Returns `params, cfg, tok` (81)

The clip runs before the learning rate is computed. Weight decay does not go through clipping because it is applied inside `Adam.step`.

## 6. Generation

`generate.generate` (lines 28-35), for each new token:

1. `forward(params, np.array(ids[-window:]), cfg)` (32)
2. `sample_next(logits[-1], temperature, top_k, top_p, rng)` (33)

`sample_next` (lines 7-26), in order:

1. If `temperature < 1e-6`, return `argmax(logits)` (8-9)
2. `p = softmax_forward(logits / temperature)` (11)
3. If `0 < top_k < len(p)`, keep `p >= k-th largest`, renormalize (13-16). Off by default, because `config.top_k = 0`
4. If `top_p < 1.0`, keep the smallest high-probability prefix reaching `top_p`, including the crossing token, and renormalize (18-24). On by default, at `0.9`
5. `rng.choice(len(p), p=p)` (26)

With the defaults, the temperature branch at step 1 is not taken (`temperature = 0.8`) and step 3 is skipped.

## 7. Interactive inference

`talk.main` (lines 7-27):

1. `load_checkpoint(prefix)` (8); the default prefix is `checkpoints/model`, and `python talk.py <prefix>` overrides it
2. `rng = np.random.default_rng(config.seed)` (9), so a new run repeats the same random sequence
3. Loop: read a prompt (15); end of input or a blank line quits (16-20)
4. `tok.encode(prompt)` (22)
5. `generate(params, cfg, ids, config.max_new_tokens, config.seq_len, config.temperature, config.top_k, config.top_p, rng)` (24-25)
6. Prints only the new tokens (27)

Model shape and tokenizer come from the checkpoint. The window and sampling settings come from the current `config.py`.

There is no chat format, message history or stopping token. Each prompt starts fresh, and output stops after `max_new_tokens`.

## 8. Checkpoint

`save_checkpoint` writes three files sharing a prefix: `.npz` (weights under the `params` keys), `.json` (`ModelConfig` fields), `.tok.json` (`merges` and `itos`). `stoi` is rebuilt on load. `checkpoints/` is git-ignored, so the files are not tied to a commit.

The existing checkpoint is not a plain output of the pinned code and data. See `DISCREPANCIES.md`, section 7, and decision D1 there, before using it for any Actual Model Output panel.

## 9. Facts that follow from the code

**Position rows beyond the window are never trained.** Training windows and the generation window both use `seq_len = 16`, so `pos_emb` rows 16 to 31 receive a gradient of zero (`embed_backward`, line 57). They still shrink through weight decay, because `Adam.step` applies decay regardless of gradient.

Actual Model Output: in `checkpoints/model.npz`, every entry of rows 16-31, divided by its value from `init_params` with seed 42, gives the same ratio, 0.90780938. Rows 0-15 do not show this.

**Key biases have zero gradient.** `b{l}.bk` adds the same amount to every score in a row of `S`, and softmax ignores a constant added to a row. `tests/test_gradcheck.py::test_key_bias_gradient_is_zero` asserts this.

**`ln_eps` is fixed by the dataclass.** `train.main` does not pass `config.ln_eps`, so `ModelConfig.ln_eps` uses its default (`transformer.py` line 15). Both values are `1e-5` at this commit, but editing `config.ln_eps` has no effect.

# Math inventory

Pinned commit: `c17aa75`. One row per implemented operation. Website pages may only document mathematics that traces to a row here.

**Columns**

- **Dir:** F = forward, B = backward. "n/a" marks operations outside differentiation (tokenizing, data, optimizer, sampling, metrics).
- **File, Function:** where the operation is written.
- **Lines:** lines within the named function at the pinned commit.
- **Called from:** the direct caller.
- **Reach:** `train`, `talk`, `both`, or `tests` (not reachable from either entry point).
- **PDF:** equation number in `LLM_study_guide_v2_streamlined.pdf`, or a page when the PDF only shows the idea without a numbered equation. `None` if absent.
- **Status:** `Implemented` means the PDF states the same mathematics. `Derived from implementation` means the code does it and the PDF does not state it as an equation.

**Notation:** `@` is matrix multiplication, `*` is element-wise. Shapes are for one sequence; the model has no batch dimension. `dM` means the gradient of the scalar loss with respect to `M`.

## Tokenizer (`tokenizer.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M01 | Pre-token count | `n_w` = occurrences of pre-token `w`; pattern `\w+`, or one whitespace, or one other character | `tokenizer.py` | `count_words` | 8-9 | n/a | string to dict | `train_bpe_on_text` | train | Eq. 3 (`n_w`) | Implemented |
| M02 | Pair frequency | `f(a,b) = sum_w n_w * count(adjacent a,b in w)` | `tokenizer.py` | `train_bpe` | 32-38 | n/a | dict | `train_bpe_on_text` | train | Eq. 3 | Implemented |
| M03 | Best pair | first pair with the largest `f`; strict `>` means ties go to the pair seen first | `tokenizer.py` | `train_bpe` | 43-48 | n/a | scalar | `train_bpe_on_text` | train | p.4, unnumbered | Implemented |
| M04 | Merge | replace adjacent `(a,b)` with `a+b`, scanning left to right | `tokenizer.py` | `merge_tokens` | 13-22 | n/a | list to list | `train_bpe`, `apply_merges` | both | p.4, code | Implemented |
| M05 | Encode | replay all merges on each pre-token; unknown token maps to `stoi["<unk>"]` | `tokenizer.py` | `Tokenizer.encode` | 87-90 | n/a | string to list of ids | `encode_corpus`, `talk.main` | both | p.4, code | Implemented |
| M06 | Vocabulary order | `itos = [<unk>] + characters in first-seen order + new merged tokens in merge order` | `tokenizer.py` | `build_tokenizer` | 102-113 | n/a | list | `train.main` | train | p.4, prose | Implemented |

## Data (`data.py`, `train.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M07 | Split | `cut = int((1 - val_fraction) * N)`; train `ids[:cut]`, validation `ids[cut:]` | `data.py` | `split_ids` | 20-21 | n/a | `(N,)` to two arrays | `train.main` | train | p.6, unnumbered | Implemented |
| M08 | Windows | `starts = 0, stride, ... < N - T`; input `ids[s+t]`, target `ids[s+t+1]`, `t = 0..T-1` | `data.py` | `make_windows` | 28-32 | n/a | `(N,)` to two `(W,T)` DataFrames | `train.main` | train | Eq. 17 | Implemented |
| M09 | Window count | `W = ceil((N - T) / stride)`, the length of `starts`. Not computed as a formula; it follows from `np.arange` | `data.py` | `make_windows` | 28 | n/a | scalar | `train.main` | train | Eq. 18 (equivalent form) | Derived from implementation |
| M10 | Batch averaging | `L = sum_b L_b / B`, `g = sum_b g_b / B`, `B = len(xb)`; a final short batch uses its own `B` | `train.py` | `batch_loss_and_grads` | 13-22 | n/a | dict of arrays | `train.main` | train | p.14, prose | Derived from implementation |
| M11 | Validation loss | mean over all windows of each window's mean cross-entropy | `train.py` | `validation_loss` | 27-34 | n/a | scalar | `train.main` | train | p.15-16, code | Derived from implementation |
| M12 | Step counts | `steps_per_epoch = ceil(W / batch_size)`, `total = epochs * steps_per_epoch` | `train.py` | `main` | 54-55 | n/a | scalar | entry point | train | p.16, code | Derived from implementation |

## Initialization (`ops.py`, `transformer.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M13 | Xavier uniform | `a = sqrt(6 / (fan_in + fan_out))`, `W ~ U[-a, a]` | `ops.py` | `xavier_uniform` | 4-5 | n/a | `(fan_in, fan_out)` | `init_params` | train | Eq. 35, 36 | Implemented |
| M14 | Parameter initial values | matrices Xavier; biases and layer-norm `beta` zeros; layer-norm `gamma` ones | `transformer.py` | `init_params` | 20-40 | n/a | see `ARCHITECTURE.md` | `train.main` | train | p.9-10, table | Implemented |
| M15 | Parameter count | `sum(v.size)` over all parameters; matches `(2d+1)V + (Tmax+2)d + L(4d^2 + 2dF + F + 9d)` | `transformer.py` | `count_params` | 45 | n/a | scalar | tests only | tests | Eq. 2 | Implemented |

## Basic operations (`ops.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M16 | Linear | `Y = X @ W + b` | `ops.py` | `linear_forward` | 8 | F | `(T,in)` to `(T,out)` | tests only | tests | Eq. 19 | Implemented |
| M17 | Linear (inline) | same formula, written inline at 7 sites: Q, K, V, attention output, two feed-forward layers, head | `transformer.py` | `attention_forward`, `ffn_forward`, `forward` | 75-77, 85, 113, 116, 166 | F | as above | `block_forward`, and the callers of `forward` | both | Eq. 19 | Implemented |
| M18 | Linear backward | `dW = X^T @ dY`, `db = sum_rows dY`, `dX = dY @ W^T` | `ops.py` | `linear_backward` | 11-13 | B | `dY (T,out)`; `dW (in,out)`; `db (out,)`; `dX (T,in)` | `ffn_backward`, `loss_and_grads` | train | Eq. 20-22 | Implemented |
| M19 | ReLU | `max(0, X)` | `ops.py` | `relu_forward` | 17 | F | shape kept | `ffn_forward` | both | Eq. 29 | Implemented |
| M20 | ReLU backward | `dZ = dA * (Z > 0)`; gradient at `Z = 0` is 0 | `ops.py` | `relu_backward` | 20 | B | shape kept | `ffn_backward` | train | Eq. 29 | Implemented |
| M21 | Softmax | `e = exp(x - max(x))`, `p = e / sum(e)`, on the last axis | `ops.py` | `softmax_forward` | 23-24 | F | shape kept | `attention_forward`, `sample_next` | both | Eq. 23, p.7 shift | Implemented |
| M22 | Softmax backward | `dx = y * (dY - sum(y * dY))`, on the last axis | `ops.py` | `softmax_backward` | 27 | B | shape kept | `attention_backward` | train | Eq. 25 | Implemented |
| M23 | Cross-entropy | `z = logits - max(logits)`, `logp = z - log(sum exp z)`, `L = -mean_t logp[t, y_t]`, `P = exp(logp)` | `ops.py` | `cross_entropy_forward` | 30-34 | F | `(T,V)` to scalar and `(T,V)` | `loss_and_grads`, `validation_loss` | train | Eq. 26, 28 | Implemented |
| M24 | Cross-entropy backward | `dZ = (P - onehot(y)) / T` | `ops.py` | `cross_entropy_backward` | 37-40 | B | `(T,V)` | `loss_and_grads` | train | Eq. 27, 28 | Implemented |
| M25 | Layer norm | `mu = mean(X)`, `var = mean((X-mu)^2)`, `s = 1/sqrt(var+eps)`, `Xhat = (X-mu)*s`, `Y = gamma*Xhat + beta`; statistics per row, across `d` | `ops.py` | `layer_norm_forward` | 43-47 | F | `(T,d)`; `gamma`, `beta` `(d,)` | `block_forward`, `forward` | both | Eq. 30, 31 | Implemented |
| M26 | Layer norm backward | `dXhat = dY*gamma`, `dX = s*(dXhat - mean(dXhat) - Xhat*mean(dXhat*Xhat))`, `dgamma = sum_rows(dY*Xhat)`, `dbeta = sum_rows dY` | `ops.py` | `layer_norm_backward` | 50-56 | B | `dX (T,d)`; `dgamma`, `dbeta` `(d,)` | `block_backward`, `loss_and_grads` | train | Eq. 32, 34 | Implemented |
| M27 | Perplexity | `exp(loss)` | `ops.py` | `perplexity` | 60 | n/a | scalar | `train.main` (validation loss only) | train | p.7, p.15, unnumbered | Implemented |

## Embeddings (`transformer.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M28 | Embedding | `X0[t] = tok_emb[ids[t]] + pos_emb[t]` | `transformer.py` | `embed_forward` | 53 | F | `(T,)` to `(T,d)` | `forward` | both | Eq. 39 | Implemented |
| M29 | Embedding backward | `dtok_emb[v] += sum over t with ids[t]=v of dX0[t]`; `dpos_emb[:T] += dX0`; rows at `T` and beyond get 0 | `transformer.py` | `embed_backward` | 56-57 | B | `dX0 (T,d)` | `loss_and_grads` | train | Eq. 40 | Implemented |

## Attention (`transformer.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M30 | Split, merge heads | `(T,d)` to `(H,T,dh)` by reshape and transpose, and the inverse | `transformer.py` | `split_heads`, `merge_heads` | 61-62, 66-67 | F and B | as stated | `attention_forward`, `attention_backward` | both | p.10, code comments p.12 | Implemented |
| M31 | Q, K, V | `Q = Xn@wq + bq`, `K = Xn@wk + bk`, `V = Xn@wv + bv` | `transformer.py` | `attention_forward` | 75-77 | F | `(T,d)` | `block_forward` | both | Eq. 41 | Implemented |
| M32 | Scores, mask, weights, values | `S = Qh @ Kh^T / sqrt(dh)`; `S = -inf` where `j > i`; `A = softmax(S)`; `O = merge(A @ Vh)` | `transformer.py` | `attention_forward` | 81-84 | F | `S`, `A` `(H,T,T)`; `O (T,d)` | `block_forward` | both | Eq. 42, 43, 44, 45 | Implemented |
| M33 | Output projection | `out = O@wo + bo` | `transformer.py` | `attention_forward` | 85 | F | `(T,d)` | `block_forward` | both | Eq. 45 | Implemented |
| M34 | Output projection backward | `dwo += O^T @ dOut`, `dbo += sum_rows dOut`, `dOh = split(dOut @ wo^T)` | `transformer.py` | `attention_backward` | 95-97 | B | `dOh (H,T,dh)` | `block_backward` | train | Eq. 47 | Implemented |
| M35 | Values and weights backward | `dVh = A^T @ dOh`, `dA = dOh @ Vh^T`, `dS = softmax_backward(A, dA)`, `dQh = dS@Kh/sqrt(dh)`, `dKh = dS^T@Qh/sqrt(dh)`, then `dQ, dK, dV = merge(...)` | `transformer.py` | `attention_backward` | 98-103 | B | per head `(H,T,dh)`; merged `(T,d)` | `block_backward` | train | p.10 "(step 1-2)", Eq. 46 | Implemented |
| M37 | Q, K, V parameters and input backward | for `n` in q,k,v: `dwn += Xn^T @ dM`, `dbn += sum_rows dM`, `dXn += dM @ wn^T` | `transformer.py` | `attention_backward` | 104-108 | B | `dXn (T,d)` | `block_backward` | train | Eq. 48, 49, 50 | Implemented |
| M38 | Key bias gradient is zero | `dbk` is 0 for each layer, because `bk` shifts every score in a row equally and softmax ignores this. Follows from M35, M37; asserted by `test_key_bias_gradient_is_zero` | `transformer.py` | `attention_backward` | 107 | B | `(d,)` | `block_backward` | train | p.9, prose | Implemented |

## Feed-forward and block (`transformer.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M36 | Feed-forward | `Z = Xn@ff1_w + ff1_b`, `A = ReLU(Z)`, `out = A@ff2_w + ff2_b` | `transformer.py` | `ffn_forward` | 113-116 | F | `Z`, `A` `(T,F)`; `out (T,d)` | `block_forward` | both | Eq. 51 | Implemented |
| M39 | Feed-forward backward | `dA, dff2_w, dff2_b = linear_backward(A, ff2_w, dOut)`; `dZ = relu_backward(Z, dA)`; `dXn, dff1_w, dff1_b = linear_backward(Xn, ff1_w, dZ)` | `transformer.py` | `ffn_backward` | 119-126 | B | `dXn (T,d)` | `block_backward` | train | Eq. 52 | Implemented |
| M40 | Pre-LN block | `Xn = LN1(X)`; `X1 = X + Att(Xn)`; `Xn2 = LN2(X1)`; `Y = X1 + FF(Xn2)` | `transformer.py` | `block_forward` | 133-139 | F | `(T,d)` | `forward` | both | Eq. 53 | Implemented |
| M41 | Block backward | `dXn2 = FFback(dY)`; `dX1 = dY + LN2back(dXn2)`; `dXn = Attback(dX1)`; `dX = dX1 + LN1back(dXn)` | `transformer.py` | `block_backward` | 143-152 | B | `(T,d)` | `loss_and_grads` | train | p.11, unnumbered | Implemented |

## Whole model (`transformer.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M42 | Length check | `len(ids) <= context_len` | `transformer.py` | `forward` | 156 | F | n/a | `generate`, `validation_loss`, `loss_and_grads` | both | p.12, code | Implemented |
| M43 | Blocks in sequence | `X(l+1) = Block_l(X(l))` | `transformer.py` | `forward` | 161-163 | F | `(T,d)` | as M42 | both | Eq. 54 | Implemented |
| M44 | Final layer norm | `Xf = LN_f(X(L))` | `transformer.py` | `forward` | 165 | F | `(T,d)` | as M42 | both | Eq. 54 | Implemented |
| M46 | Head | `logits = Xf@head_w + head_b` | `transformer.py` | `forward` | 166 | F | `(T,V)` | as M42 | both | Eq. 54 | Implemented |
| M47 | Head and final-norm backward | `dXf, dhead_w, dhead_b = linear_backward(Xf, head_w, dZ)`; `dX, dlnf_g, dlnf_b = layer_norm_backward(dXf, ...)` | `transformer.py` | `loss_and_grads` | 176-184 | B | `(T,d)`; `(d,V)`; `(V,)`; `(d,)` | `batch_loss_and_grads` | train | p.11, unnumbered | Implemented |
| M48 | Reverse order | blocks in `reversed(range(L))`, then embeddings | `transformer.py` | `loss_and_grads` | 186-188 | B | `(T,d)` | `batch_loss_and_grads` | train | p.11, unnumbered | Implemented |

## Optimizer (`optim.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M49 | Moments | `m = b1*m + (1-b1)*g`, `v = b2*v + (1-b2)*g*g` | `optim.py` | `Adam.step` | 19-20 | n/a | per tensor | `train.main` | train | Eq. 55 | Implemented |
| M50 | Bias correction | `mhat = m/(1-b1^t)`, `vhat = v/(1-b2^t)`; `t` starts at 1 | `optim.py` | `Adam.step` | 22-23 | n/a | per tensor | `train.main` | train | Eq. 56 | Implemented |
| M51 | Update with decoupled decay | `u = mhat/(sqrt(vhat)+eps)`; if `ndim >= 2`, `u += wd*theta`; `theta -= lr*u` | `optim.py` | `Adam.step` | 25-30 | n/a | per tensor | `train.main` | train | Eq. 56, 57 | Implemented |
| M52 | Learning-rate schedule | `step <= warmup`: `lr_max*step/warmup`. Otherwise `frac = (step-warmup)/(total-warmup)`, `lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*frac))` | `optim.py` | `lr_schedule` | 33-37 | n/a | scalar | `train.main` | train | Eq. 58 | Implemented |
| M53 | Global-norm clip | `norm = sqrt(sum of all g_i^2)`; if `norm > max_norm`, every tensor `*= max_norm/norm`; returns `norm` before clipping | `optim.py` | `clip_grad_norm` | 41-50 | n/a | per tensor | `train.main` | train | Eq. 59 | Implemented |

## Sampling (`generate.py`)

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M54 | Greedy guard | `temperature < 1e-6`: `argmax(logits)`. Not taken at the default | `generate.py` | `sample_next` | 8-9 | n/a | `(V,)` to int | `generate` | both | p.14, unnumbered | Implemented |
| M55 | Temperature | `p = softmax(logits / temperature)` | `generate.py` | `sample_next` | 11 | n/a | `(V,)` | `generate` | both | Eq. 60 | Implemented |
| M56 | Top-k | if `0 < k < V`: keep `p_i >= k-th largest`, renormalize; ties are all kept. Skipped at the default `top_k = 0` | `generate.py` | `sample_next` | 13-16 | n/a | `(V,)` | `generate` | both | p.14, worked example, no equation | Derived from implementation |
| M57 | Top-p | if `top_p < 1`: sort descending, `cum = cumsum`, keep the first `searchsorted(cum, top_p) + 1` tokens, renormalize | `generate.py` | `sample_next` | 18-24 | n/a | `(V,)` | `generate` | both | p.15, worked example, no equation | Derived from implementation |
| M58 | Sample | `rng.choice(V, p=p)` | `generate.py` | `sample_next` | 26 | n/a | `(V,)` to int | `generate` | both | p.15, code | Implemented |
| M59 | Generation loop | repeat: `logits = forward(ids[-window:])`, append `sample_next(logits[-1])` | `generate.py` | `generate` | 31-33 | n/a | `(min(n,window),V)`; last row `(V,)` | `train.main`, `talk.main` | both | p.15, code | Implemented |

## Gradient checking (`gradcheck.py`). Test-time only

| ID | Name | Equation as implemented | File | Function | Lines | Dir | Shapes | Called from | Reach | PDF | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M60 | Central difference | `g[i] = (f(x + h e_i) - f(x - h e_i)) / (2h)` | `gradcheck.py` | `numerical_grad` | 5-12 | n/a | shape of `x` | `tests/test_gradcheck.py` | tests | Eq. 37 (left side) | Implemented |
| M61 | Relative error | `abs(a-b) / max(abs(a), abs(b), eps)` | `gradcheck.py` | `relative_error` | 17 | n/a | scalar | `check_params` | tests | Eq. 38 | Implemented |
| M62 | Per-parameter check | at sampled indices, worst `relative_error(analytic, central difference)` per parameter | `gradcheck.py` | `check_params` | 23-31 | n/a | dict | `tests/test_gradcheck.py` | tests | p.8-9, prose | Implemented |

IDs only identify a row. They are not strictly in order, because rows are grouped by topic, and M45 is not used.

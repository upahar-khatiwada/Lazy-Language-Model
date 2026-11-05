# PDF versus code

Reference: `LLM_study_guide_v2_streamlined.pdf`, 19 pages. Pinned commit: `c17aa75`.

This file is the filter for what the website may document.

**Columns**

- **In PDF:** always yes in section 2, because each row comes from the PDF. Section 3 includes techniques the PDF never mentions.
- **In code:** `yes`, `partly`, `test` (only in tests or test-only code), or `no`.
- **Used:** reachable from `train.main` or `talk.main`, not merely defined or tested. `yes (T)` means training only; `yes (I)` means inference only; `yes` means both. `off` means the code is reachable but the branch does not run with the committed config.
- **Document:** `yes` or `no`. A `no` row may not appear as something this project does. Where noted, it may appear as a clearly labelled "In general, ..." explanation of an implemented operation.

**Reading method and limits:** the PDF was read as extracted text with `pdftotext`, not as rendered pages. Superscripts are flattened by extraction and were reconstructed from context. IDs (E001...) were assigned during extraction; they are not printed in the PDF. Numbers printed in the PDF run (1)-(3), (17)-(32), (34)-(62). The PDF explains that (4)-(16) were removed with earlier stages. (33) is absent with no explanation.

## 1. Topic verdicts

| Topic | Verdict | Evidence |
|---|---|---|
| Xavier initialisation | **Confirmed.** Uniform variant, applied to every matrix including both embeddings | `ops.py` `xavier_uniform`; `transformer.py` `init_params` |
| ReLU | **Confirmed** | `ops.py` `relu_forward`, `relu_backward`; called in `ffn_forward`, `ffn_backward` |
| LayerNorm | **Confirmed.** Statistics across `d` per row; learned `gamma`, `beta` per feature | `ops.py` `layer_norm_forward`, `layer_norm_backward` |
| Pre-LN residual blocks | **Confirmed.** Branches read normalized copies; residuals add to unnormalized values | `transformer.py` `block_forward`, `block_backward` |
| Learned positional embeddings | **Confirmed.** `pos_emb` has `context_len` rows; only the first 16 ever get a gradient | `transformer.py` `init_params`, `embed_forward`, `embed_backward` |
| Embedding backward | **Confirmed.** `np.add.at` sums repeated ids | `transformer.py` `embed_backward` |
| Multi-head causal attention | **Confirmed.** `H = 2`; mask is `-inf` above the diagonal | `transformer.py` `split_heads`, `merge_heads`, `attention_forward`, `attention_backward` |
| Perplexity | **Confirmed, narrow use.** Applied only to validation loss, for a printed line during training | `ops.py` `perplexity`; `train.py` `main` |
| AdamW weight decay | **Confirmed.** The class is named `Adam`; decay is decoupled. Applied where `ndim >= 2` | `optim.py` `Adam.step`; `train.py` `main` passes `config.weight_decay` |
| Warmup | **Confirmed.** Linear, 50 steps | `optim.py` `lr_schedule` |
| Cosine decay | **Confirmed.** From `lr_max` to `lr_min` | `optim.py` `lr_schedule` |
| Gradient clipping | **Confirmed.** Global norm, `max_norm = 1.0` | `optim.py` `clip_grad_norm`; `train.py` `main` |
| Top-k | **Confirmed in code, off by default.** `config.top_k = 0` skips the branch | `generate.py` `sample_next` |
| Top-p | **Confirmed and active.** `config.top_p = 0.9` | `generate.py` `sample_next` |
| Temperature | **Confirmed and active.** `config.temperature = 0.8` | `generate.py` `sample_next` |
| Checkpointing | **Confirmed.** Three files; saved once, at the end of training | `checkpoint.py` `save_checkpoint`, `load_checkpoint`; `train.py` `main` |
| Interactive inference | **Confirmed.** A completion prompt, not a chat | `talk.py` `main` |

**Parameters that receive weight decay** (`ndim >= 2`): `tok_emb`, `pos_emb`, `b{l}.wq`, `b{l}.wk`, `b{l}.wv`, `b{l}.wo`, `b{l}.ff1_w`, `b{l}.ff2_w`, `head_w`.

**No decay:** `b{l}.bq`, `bk`, `bv`, `bo`, `ln1_g`, `ln1_b`, `ln2_g`, `ln2_b`, `ff1_b`, `ff2_b`, and `lnf_g`, `lnf_b`, `head_b`.

## 2. Every equation in the PDF

| ID | Eq. | Sec. | Page | Formula (from extraction) | In PDF | In code | Used | Document | Code location or reason |
|---|---|---|---:|---|---|---|---|---|---|
| E001 | none | 2 | 3 | d divisible by H, T <= Tmax | yes | yes | yes | yes | `config.py` asserts, on import |
| E002 | none | 2 | 3 | decode(encode(x)) = x | yes | test | no | no | Only `test_round_trip`; not true for unseen characters |
| E003 | none | 2 | 3 | targets[:, :-1] == inputs[:, 1:] | yes | yes | yes (T) | yes | Result of `make_windows`; asserted by a test |
| E004 | none | 3 | 3 | Y = XW + 1b^T | yes | yes | yes | yes | Inline in `transformer.py` |
| E005 | none | 3 | 3 | dY = partial L / partial Y | yes | yes | yes (T) | yes | Naming convention across backward functions |
| E006 | (1) | 3 | 3 | d_h = d / H, d mod H = 0 | yes | yes | yes | yes | `attention_forward`; `config.py` assert |
| E007 | (2) | 3 | 3 | #params = (2d+1)V + (Tmax+2)d + L(4d^2+2dF+F+9d) | yes | partly | no | yes | Not computed as a formula. Holds for `init_params` shapes; label Derived from implementation |
| E008 | none | 3 | 3 | one block = 12,704 | yes | partly | no | yes | Matches checkpoint shapes |
| E009 | none | 3 | 3 | 26,496 + 65V; V=300 gives 45,996 | yes | test | no | no | V=300 is a test value. The checkpoint has V=575 and 63,871 |
| E010 | none | 3 | 3 | p = 1/V; first loss near ln V; ln 300 = 5.704 | yes | partly | yes (T) | yes | `train.main` prints `ln V`; `test_initial_loss_is_close_to_ln_vocab`. Do not use the V=300 value as the project's |
| E011 | (3) | 4.1 | 4 | f(a,b) = sum_w n_w #{i: s_i=a, s_{i+1}=b} | yes | yes | yes (T) | yes | `train_bpe` |
| E012 | none | 4.1 | 4 | (a*,b*) = argmax f | yes | yes | yes (T) | yes | `train_bpe`; ties to first seen |
| E013 | none | 4.1 | 5 | f(a,b) = sum_w n_w * #(a,b in w) | yes | yes | yes (T) | no | Duplicate of E011 |
| E014 | (17) | 4.2 | 6 | input (x_s..x_{s+T-1}), target (x_{s+1}..x_{s+T}) | yes | yes | yes (T) | yes | `make_windows` |
| E015 | (18) | 4.2 | 6 | W = floor((N-T-1)/k) + 1 | yes | partly | yes (T) | yes | Equals the length of `np.arange(0, N-T, k)`; label Derived from implementation |
| E016 | none | 4.2 | 6 | cut at floor((1-rho)N) | yes | yes | yes (T) | yes | `split_ids` |
| E017 | none | 4.2 | 6 | N=10, W=6 | yes | test | no | yes | Illustrative Example; `test_window_table_matches_hand_example` |
| E018 | none | 4.2 | 6 | inputs[w,t] = ids[s+t], targets[w,t] = ids[s+t+1] | yes | yes | yes (T) | no | Duplicate of E014 |
| E019 | (19) | 4.3 | 6 | Y = XW + 1b^T with shapes | yes | yes | yes | yes | Inline; `ops.linear_forward` itself is not called |
| E020 | (20-22) | 4.3 | 6 | dW = X^T dY, db = 1^T dY, dX = dY W^T | yes | yes | yes (T) | yes | `linear_backward`; inline in `attention_backward` |
| E021 | (23) | 4.3 | 7 | p_i = exp(z_i) / sum exp(z_k) | yes | yes | yes | yes | `softmax_forward` |
| E022 | none | 4.3 | 7 | softmax(z - c) = softmax(z); c = max z | yes | yes | yes | yes | `softmax_forward`, `cross_entropy_forward` |
| E023 | (24) | 4.3 | 7 | dp_i/dz_j = p_i(delta_ij - p_j) | yes | partly | no | yes | The code never forms this Jacobian. Allowed as the derivation of E024, stated as such |
| E024 | (25) | 4.3 | 7 | dz_j = p_j(dp_j - sum p_i dp_i) | yes | yes | yes (T) | yes | `softmax_backward` |
| E025 | (26) | 4.3 | 7 | l = -log p_y = -z_y + log sum exp(z_k) | yes | yes | yes (T) | yes | `cross_entropy_forward` |
| E026 | (27) | 4.3 | 7 | dl/dz_j = p_j - 1[j=y] | yes | yes | yes (T) | yes | `cross_entropy_backward` |
| E027 | (28) | 4.3 | 7 | L = (1/T) sum l_t, dZ = (1/T)(P - E_y) | yes | yes | yes (T) | yes | `cross_entropy_forward`, `cross_entropy_backward` |
| E028 | none | 4.3 | 7 | Same gradient through softmax backward | yes | no | no | no | The code does not take this route |
| E029 | none | 4.3 | 7 | Perplexity = e^L; uniform model gives V | yes | yes | yes (T) | yes | `perplexity`, on validation loss |
| E030 | (29) | 4.3 | 7 | ReLU(z) = max(0,z), dZ = dA * 1[Z>0] | yes | yes | yes | yes | `relu_forward`, `relu_backward` |
| E031 | (30) | 4.3 | 7 | mu, sigma^2, s = (sigma^2+eps)^(-1/2) | yes | yes | yes | yes | `layer_norm_forward` |
| E032 | (31) | 4.3 | 7 | xhat_i = (x_i-mu)s, y_i = gamma_i xhat_i + beta_i | yes | yes | yes | yes | `layer_norm_forward` |
| E033 | (32) | 4.3 | 7 | dbeta = sum dy, dgamma = sum dy*xhat, dxhat = dy*gamma | yes | yes | yes (T) | yes | `layer_norm_backward` |
| E034 | (34) | 4.3 | 7 | dx_j = s(dxhat_j - mean(dxhat) - xhat_j mean(dxhat*xhat)) | yes | yes | yes (T) | yes | `layer_norm_backward` |
| E035 | none | 4.3 | 7 | sum dx_j = 0; sum dx_j xhat_j = 0 | yes | test | no | no | First part is asserted in `test_layer_norm_worked_example`; second part is not in the repository |
| E036 | none | 4.3 | 7 | Var(y) = n Var(x) Var(w); 1/n_in; 1/n_out | yes | no | no | no | Rationale only. "In general, ..." wording allowed next to Xavier |
| E037 | (35,36) | 4.3 | 7 | Var(W) = 2/(n_in+n_out), W ~ U[-a,a], a = sqrt(6/(n_in+n_out)) | yes | yes | yes (T) | yes | `xavier_uniform`. The code sets `a`; the variance follows from it |
| E038 | none | 4.3 | 7 | a = 0.3062 (32x32), 0.1936 (feed-forward) | yes | partly | yes (T) | yes | Values follow from `config.py` and `xavier_uniform` |
| E039 | none | 4.3 | 7 | head logit variance near d*2/(d+V), 0.19 at V=300 | yes | no | no | no | Not in code; V=300 is not this model |
| E040 | none | 4.4 | 8 | central error O(h^2), one-sided O(h) | yes | no | no | no | Rationale only |
| E041 | (37) | 4.4 | 8 | [f(x+h)-f(x-h)]/2h = f'(x) + (h^2/6)f'''(x) + ... | yes | partly | no | no | Left side is E043; the expansion is not in code |
| E042 | (38) | 4.4 | 8 | rel(a,n) = abs(a-n) / max(abs(a),abs(n),eps0) | yes | test | no | yes | `relative_error`; label test-only. Decision D2 |
| E043 | none | 4.4 | 9 | (f(x+h e_i) - f(x-h e_i)) / 2h | yes | test | no | yes | `numerical_grad`, `check_params`; label test-only. Decision D2 |
| E044 | none | 4.5 | 9 | X0 = E[ids] + P[:T] | yes | yes | yes | yes | `embed_forward` |
| E045 | none | 4.5 | 9 | Xn = LN1(X) | yes | yes | yes | yes | `block_forward` |
| E046 | none | 4.5 | 9 | X1 = X + MultiHeadCausalAttention(Xn) | yes | yes | yes | yes | `block_forward` |
| E047 | none | 4.5 | 9 | Xn2 = LN2(X1) | yes | yes | yes | yes | `block_forward` |
| E048 | none | 4.5 | 9 | Xout = X1 + (ReLU(Xn2 W1 + b1) W2 + b2) | yes | yes | yes | yes | `block_forward`, `ffn_forward` |
| E049 | none | 4.5 | 9 | Xf = LN_f(X_L) | yes | yes | yes | yes | `forward` |
| E050 | none | 4.5 | 9 | logits = Xf W_head + b_head | yes | yes | yes | yes | `forward` |
| E051 | none | 4.5 | 9 | loss = mean_t(-log softmax(logits)[t,target_t]) | yes | yes | yes (T) | yes | `cross_entropy_forward` |
| E052 | (39) | 4.5 | 10 | X(0)_t = E_{i_t} + P_t | yes | yes | yes | yes | `embed_forward` |
| E053 | (40) | 4.5 | 10 | dE_v = sum dX_t, dP_t = dX_t (t<=T), 0 otherwise | yes | yes | yes (T) | yes | `embed_backward` |
| E054 | (41) | 4.5 | 10 | Q, K, V projections | yes | yes | yes | yes | `attention_forward` |
| E055 | (42) | 4.5 | 10 | S = Q K^T / sqrt(d_h) | yes | yes | yes | yes | `attention_forward` |
| E056 | (43) | 4.5 | 10 | S_tilde = S if j<=i else -inf; A = softmax_rows | yes | yes | yes | yes | `attention_forward` |
| E057 | (44) | 4.5 | 10 | O = A V | yes | yes | yes | yes | `attention_forward` |
| E058 | (45) | 4.5 | 10 | Att = [O_1...O_H] W_O + 1b_O^T | yes | yes | yes | yes | `attention_forward` |
| E059 | none | 4.5 | 10 | q.k has variance d_h | yes | no | no | no | Rationale only. "In general, ..." wording allowed next to E055 |
| E060 | none | 4.5 | 10 | d_h=64, gap 8, largest weight 0.9997 | yes | no | no | no | This model has `dh = 16`; not in code |
| E061 | none | 4.5 | 10 | exp(-inf) = 0 | yes | yes | yes | yes | Effect of the mask in `softmax_forward`; `test_attention_weights_are_causal_and_normalised` |
| E062 | none | 4.5 | 10 | head h owns a column block; count 4(d^2+d) for any H | yes | yes | yes | yes | `split_heads`; count from `init_params` |
| E063 | (47) | 4.5 | 10 | dO = dAtt W_O^T, dW_O = O^T dAtt, db_O = 1^T dAtt | yes | yes | yes (T) | yes | `attention_backward` |
| E064 | "(step 1-2)" | 4.5 | 10 | dV = A^T dO, dA = dO V^T, dS = A*(dA - rowsum(A*dA)) | yes | yes | yes (T) | yes | `attention_backward`, `softmax_backward` |
| E065 | (46) | 4.5 | 10 | dQ = c dS K, dK = c dS^T Q, c = 1/sqrt(d_h) | yes | yes | yes (T) | yes | `attention_backward` |
| E066 | (48,49) | 4.5 | 10 | dW_Q = X_n^T dQ, db_Q = 1^T dQ | yes | yes | yes (T) | yes | `attention_backward` |
| E067 | (50) | 4.5 | 10 | dX_n = dQ W_Q^T + dK W_K^T + dV W_V^T | yes | yes | yes (T) | yes | `attention_backward` |
| E068 | (51) | 4.5 | 11 | Z, A = ReLU(Z), Ff = A W_2 + 1b_2^T | yes | yes | yes | yes | `ffn_forward` |
| E069 | (52) | 4.5 | 11 | Feed-forward backward through two linear_bwd calls | yes | yes | yes (T) | yes | `ffn_backward` |
| E070 | (53) | 4.5 | 11 | X_1 = X + Att(LN_1(X)), X' = X_1 + Ff(LN_2(X_1)) | yes | yes | yes | yes | `block_forward` |
| E071 | none | 4.5 | 11 | dX_1 = dX' + branch; dX = dX_1 + branch | yes | yes | yes (T) | yes | `block_backward` |
| E072 | none | 4.5 | 11 | dX(L)/dX(0) = product (I + J_l) | yes | no | no | no | Rationale only. "In general, ..." wording allowed next to residuals |
| E073 | none | 4.5 | 11 | y = x + wx, dx = 1.5 dy; 0.5^20 | yes | no | no | no | Generic example, not this code |
| E074 | (54) | 4.5 | 11 | X(l+1) = Block_l, X_f = LN_f, Z = X_f W_head + b, L = CE | yes | yes | yes | yes | `forward`, `loss_and_grads` |
| E075 | none | 4.5 | 11 | dZ = (1/T)(P - E_y) | yes | yes | yes (T) | yes | `cross_entropy_backward` |
| E076 | none | 4.6 | 13 | L(theta+Delta) ~ L + g.Delta; Delta = -eta g | yes | no | no | no | The code does not use plain gradient descent. "In general, ..." wording allowed |
| E077 | (55) | 4.6 | 13 | m_t, v_t moving averages | yes | yes | yes (T) | yes | `Adam.step` |
| E078 | (56) | 4.6 | 13 | mhat, vhat, theta update without decay | yes | yes | yes (T) | yes | `Adam.step`; exact form for parameters with `ndim < 2` |
| E079 | none | 4.6 | 13 | underestimate by 1-beta_1^t; first update eta sign(g) | yes | yes | yes (T) | yes | Comment in `Adam.step`; `test_adam_first_step_is_lr_times_sign_of_gradient`. The sign form holds without decay |
| E080 | (57) | 4.6 | 13 | AdamW update with lambda theta | yes | yes | yes (T) | yes | `Adam.step`; only where `ndim >= 2` |
| E081 | (58) | 4.6 | 14 | warmup and cosine schedule | yes | yes | yes (T) | yes | `lr_schedule` |
| E082 | (59) | 4.6 | 14 | global norm; g <- g min(1, c/norm) | yes | yes | yes (T) | yes | `clip_grad_norm` |
| E083 | (60) | 4.7 | 14 | p_i(tau) = exp(z_i/tau) / sum exp(z_k/tau) | yes | yes | yes | yes | `sample_next` |
| E084 | none | 4.7 | 14 | argmax when tau < 10^-6 | yes | yes | off | yes | `sample_next`; not taken at `temperature = 0.8` |
| E085 | (61) | 4.8 | 15 | p(x_1..x_N) = product p(x_t given x_<t) | yes | no | no | no | Not computed. "In general, ..." wording allowed |
| E086 | (62) | 4.8 | 15 | L = -(1/N) sum log p(x_{t+1} given x_<=t) | yes | no | no | no | The code computes a mean over windows of at most `seq_len` tokens (M11), not this corpus-wide quantity with full history |
| E087 | none | 4.8 | 15 | e^L as equally likely choices per token | yes | partly | yes (T) | no | Duplicate of E029. The interpretation may use "In general, ..." wording |
| E088 | none | 7 | 18 | N(0, 0.02^2) | yes | no | no | no | PDF contrast only; superscript was reconstructed |
| E089 | none | 7 | 19 | -log(p + 10^-10) | yes | no | no | no | PDF contrast only; the code uses log-softmax without an epsilon |

Count: 89 rows, E001 to E089.

## 3. Every technique in the PDF, and six it does not mention

| Technique | In PDF | In code | Used | Document | Note |
|---|---|---|---|---|---|
| Decoder-only transformer | yes | yes | yes | yes | |
| Pre-LN | yes | yes | yes | yes | |
| Final layer norm | yes | yes | yes | yes | |
| Post-LN | reference title only | no | no | no | |
| Layer norm with learned gamma, beta | yes | yes | yes | yes | |
| BatchNorm | no | no | no | no | |
| Dropout | no | no | no | no | Do not imply it exists |
| Label smoothing | no | no | no | no | |
| Multi-head attention | yes | yes | yes | yes | |
| Causal mask | yes | yes | yes | yes | |
| Scaled dot-product | yes | yes | yes | yes | |
| Residual connections | yes | yes | yes | yes | |
| Feed-forward block | yes | yes | yes | yes | |
| Learned token embedding | yes | yes | yes | yes | |
| Learned position table | yes | yes | yes | yes | Rows 16-31 get no gradient |
| Sinusoidal positions | no | no | no | no | |
| Weight tying | yes, as not used | no | no | no | May state that `head_w` is separate from `tok_emb` |
| ReLU | yes | yes | yes | yes | |
| GELU | yes, as future upgrade | no | no | no | |
| Softmax stability shift | yes | yes | yes | yes | |
| Cross-entropy by log-softmax | yes | yes | yes (T) | yes | |
| Upstream epsilon loss | yes, contrast | no | no | no | |
| Xavier uniform | yes | yes | yes (T) | yes | |
| N(0, 0.02^2) embedding init | yes, contrast | no | no | no | |
| Zero bias, zero beta, gamma = 1 | yes | yes | yes (T) | yes | |
| Hand-written backward, no autograd | yes | yes | yes (T) | yes | |
| Backward cache | yes | yes | yes | yes | Forward builds it, including during inference, where it goes unused |
| `+=` accumulation in backward | yes | yes | yes (T) | yes | |
| `np.add.at` scatter-add | yes | yes | yes (T) | yes | |
| Flat dict parameters | yes | yes | yes | yes | |
| BPE | yes | yes | yes | yes | Works on Unicode characters, not bytes |
| Regex pre-tokenization | yes | partly | yes | yes | Document the code's pattern, not the PDF's. See `DISCREPANCIES.md` |
| Lower-casing | yes | no | no | no | The code preserves case |
| `<unk>` policy | yes | yes | yes | yes | |
| Complete vocabulary | yes | yes | yes (T) | yes | |
| Tokenizer save and load | yes | partly | yes | yes | In `checkpoint.py`; `Tokenizer` has no such methods |
| Pre-token encoding cache | yes | no | no | no | |
| Pad, BOS, EOS tokens | no | no | no | no | |
| Windowing with stride | yes | yes | yes (T) | yes | |
| Contiguous split before windowing | yes | yes | yes (T) | yes | |
| Mini-batches, mean gradient, shuffled | yes | yes | yes (T) | yes | |
| Adam with bias correction | yes | yes | yes (T) | yes | |
| AdamW | yes | yes | yes (T) | yes | |
| L2 penalty in the loss | yes, contrast | no | no | no | |
| Plain Adam at constant LR | yes, contrast | partly | no | no | `weight_decay=0.0` is the class default, but the pipeline passes `0.01` and a schedule |
| Linear warmup | yes | yes | yes (T) | yes | |
| Cosine decay | yes | yes | yes (T) | yes | |
| Global-norm clipping | yes | yes | yes (T) | yes | |
| Plain gradient descent | yes, motivation | no | no | no | |
| Autoregressive generation | yes | yes | yes | yes | |
| Temperature with guard | yes | yes | yes | yes | |
| Greedy decoding | yes | yes | off | yes | Label as off at default settings |
| Top-k | yes | yes | off | yes | Label as off at default settings |
| Top-p | yes | yes | yes | yes | |
| Sampling order | yes | yes | yes | yes | |
| Window limited to T | yes | yes | yes | yes | |
| Beam search, repetition penalty | no | no | no | no | |
| Perplexity | yes | yes | yes (T) | yes | |
| Gradient checking | yes | test | no | yes | Label test-only. Decision D2 |
| Relative-error thresholds | yes | partly | no | no | Test asserts `< 1e-4`; the PDF's three bands are not in the repository |
| Validation during training | yes | yes | yes (T) | yes | |
| Overfit one batch | yes | test | no | no | Only `test_overfit_one_window`; may be listed as a test |
| Six-rung ladder of checks | yes | no | no | no | |
| Checkpointing | yes | yes | yes | yes | |
| Interactive prompt | yes | yes | yes (I) | yes | |
| Stop when validation rises | yes, advice | no | no | no | No early stopping in code |
| ELIZA chat (`chat.py`) | yes, removed | no | no | no | |
| XOR network (`xor_net.py`) | yes, removed | no | no | no | |
| Skip-gram (`skipgram.py`) | yes, removed | no | no | no | |
| Instruction following | yes, out of scope | no | no | no | |

Count: 70 rows.

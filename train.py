import math
import numpy as np
import config
from data import load_corpus, encode_corpus, split_ids, make_windows, iterate_batches
from generate import generate
from ops import cross_entropy_forward, perplexity
from optim import Adam, clip_grad_norm, lr_schedule
from tokenizer import build_tokenizer
from transformer import ModelConfig, init_params, forward, loss_and_grads

CORPUS_PATH, COLUMN, MAX_ROWS = "data/corpus.csv", "text", 2000

def batch_loss_and_grads(params, xb, yb, cfg: ModelConfig):
    B = len(xb)
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    total_loss = 0.0

    for x, y in zip(xb, yb):
        loss, g = loss_and_grads(params, x, y, cfg)
        total_loss += loss / B

        for k in grads:
            grads[k] += g[k] / B

    return total_loss, grads

def validation_loss(params, cfg, inputs, targets):
    losses = []

    for x, y in zip(inputs.to_numpy(), targets.to_numpy()):
        logits, _ = forward(params, x, cfg)
        loss, _ = cross_entropy_forward(logits, y)
        losses.append(loss)

    return float(np.mean(losses))

def main():
    rng = np.random.default_rng(config.seed)

    # 1. text -> ids -> train/val windows (split BEFORE windowing)
    text = load_corpus(CORPUS_PATH, COLUMN, MAX_ROWS)
    tok = build_tokenizer(text, config.num_merges)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = split_ids(ids, config.val_fraction)
    tr_x, tr_y = make_windows(train_ids, config.seq_len, config.stride)
    va_x, va_y = make_windows(val_ids, config.seq_len, config.stride)

    # 2. model + optimiser
    cfg = ModelConfig(tok.vocab_size, config.context_len, config.emb_dim,
                      config.num_heads, config.ff_dim, config.num_layers)
    params = init_params(cfg, rng)
    opt = Adam(params, config.beta1, config.beta2, config.adam_eps, config.weight_decay)
    print(f"V={cfg.vocab_size}  ln V={math.log(cfg.vocab_size):.3f}")

    steps_per_epoch = math.ceil(len(tr_x) / config.batch_size)
    total_steps = config.epochs * steps_per_epoch

    # 3. the loop: gradient -> clip -> learning rate -> Adam
    step = 0
    for epoch in range(config.epochs):
        for xb, yb in iterate_batches(tr_x, tr_y, config.batch_size, rng):
            loss, grads = batch_loss_and_grads(params, xb, yb, cfg)
            norm = clip_grad_norm(grads, config.grad_clip)
            step += 1
            lr = lr_schedule(step, config.warmup_steps, total_steps,
                             config.lr_max, config.lr_min)
            opt.step(params, grads, lr)
            if step == 1 or step % 50 == 0:
                print(f"step {step:5d}  loss {loss:.3f}  |g| {norm:.2f}  lr {lr:.5f}")

        v = validation_loss(params, cfg, va_x, va_y)
        print(f"epoch {epoch + 1}: val loss {v:.3f}  ppl {perplexity(v):.1f}")
        seed_ids = [int(i) for i in va_x.iloc[0][:3]]
        out = generate(params, cfg, seed_ids, 30, config.seq_len,
                       config.temperature, config.top_k, config.top_p, rng)
        print("  sample:", repr(tok.decode(out)))

    return params, cfg, tok


if __name__ == "__main__":
    main()
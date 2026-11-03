import sys
import numpy as np
import config
from checkpoint import load_checkpoint
from generate import generate

def main(prefix="checkpoints/model"):
    params, cfg, tok = load_checkpoint(prefix)
    rng = np.random.default_rng(config.seed)
    print(f"loaded {prefix}  (V={cfg.vocab_size})")
    print("Type the start of a text and press Enter. Empty line quits.")

    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break

        if prompt.strip() == "":
            break

        ids = tok.encode(prompt)

        out = generate(params, cfg, ids, config.max_new_tokens, config.seq_len,
                       config.temperature, config.top_k, config.top_p, rng)

        print(tok.decode(out[len(ids):])) # only the new tokens, the prompt is already on screen

if __name__ == "__main__":
    main(*sys.argv[1:])
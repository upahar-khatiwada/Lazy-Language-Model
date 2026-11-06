# Lazy Language Model

A small LLM built from scratch in plain Python and NumPy, to understand how LLMs work from first principles.

## Setup

Developed with Python 3.14. `requirements.txt` pins the exact package versions.
```sh
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

Training reads the `text` column of `data/corpus.csv`, which is included in the repo.
```sh
python train.py
```
This builds the tokenizer, trains the model and prints the loss and a text sample as it goes. It saves `model.npz`, `model.json` and `model.tok.json` to `checkpoints/`. The settings in `config.py` take a few hours; lower `max_rows` or `epochs` for a quicker run.

## Run the model

`talk.py` needs a trained checkpoint, and `checkpoints/` is git-ignored, so train first. Type the start of a text, and press Enter on an empty line to quit.
```sh
python talk.py
```
```text
> once upon a time
 to see how she had eat him and stood to make the scirar, and he had been afraid that the L
```

## Run the tests

```sh
python -m pytest tests
```

## Docs and learning the math

The docs are a website in `docs-site/`. They need [Bun](https://bun.sh) and [Node.js](https://nodejs.org). Open the address the last command prints, usually `http://localhost:3000`.
```sh
cd docs-site
bun install
bun run dev
```
Read the stages in this order:

1. Notation & Shapes: symbols, matrix multiplication, parameter count
2. Tokenization: byte pair encoding (BPE) merges
3. Data Pipeline: train/validation split, sliding windows, batches
4. Linear / ReLU / Initialisation: linear layers, ReLU, Xavier initialisation
5. Softmax & Cross-Entropy: softmax, cross-entropy loss, perplexity
6. LayerNorm: layer normalisation
7. Embeddings: token and position embeddings
8. Attention: scaled dot-product attention, causal mask, multiple heads
9. Transformer Block: feed-forward network, pre-norm block, residuals
10. Full Model: forward pass, backward pass (backprop), gradient flow
11. How the Model Learns: one learning step, from gradient to weight update
12. AdamW / Learning Rate / Clipping: AdamW, learning rate schedule, gradient clipping
13. Training Loop: mini-batch gradients, validation loss
14. Sampling: temperature, top-k, top-p
15. Autoregressive Generation: feeding each new token back in
16. Gradient Checking: numerical gradients by central differences

Each stage has a small worked example with hand-picked numbers. `tests/` checks hand-worked values for tokenization, windows, linear layers, softmax, cross-entropy, LayerNorm, embeddings, the feed-forward step, Adam and sampling. Attention and the full model are checked by properties and numerical gradient checks instead, and the training loop has no test of its own.

## Project structure

```text
config.py                 model, training and generation settings
tokenizer.py, data.py     BPE tokenizer, corpus, windows and batches
ops.py, transformer.py    math operations and the model
optim.py, train.py        optimizer and training script
generate.py, talk.py      text generation and interactive prompt
checkpoint.py             save and load a trained model
gradcheck.py              numerical gradient checks
requirements.txt          pinned Python packages
tests/                    pytest tests
docs-site/                documentation website
data/                     training corpus
checkpoints/              trained model (git-ignored)
```

## Credits and tutorials used

- https://github.com/w3cj/how-llms-work (the TypeScript project this is ported from)
- https://youtu.be/YmLp8qe87A0 (video tutorial followed)

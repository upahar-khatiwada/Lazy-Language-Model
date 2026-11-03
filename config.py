# ---------- model shape ----------
context_len = 32 # suppose 5 tokens then theres 5x32 rows, the model can look 32 texts at once
emb_dim     = 32 # size of the vector that flows through the entire model with same width at each layer
num_heads   = 2 # emb_dim must be divisible by this, model splits each vector into 2 halves of 16 numbers and runs attention separately
ff_dim      = 128 # in the feed forward step, vector is stretched from 32 to 128 temporarily to allow model compute richer things
num_layers  = 2 # number of layers of the LLM
seq_len     = 16 # each training example is a 16 rows table, we cut the input corpus into pieces of 16 tokens
ln_eps      = 1e-5 # tiny number added inside layer norm so we never divide by zero

# ---------- data ----------
corpus_path   = "data/corpus.csv" # file the text is read from
corpus_column = "text" # which column of the csv holds the text
max_rows      = 2000 # only read this many rows, keeps training fast
num_merges    = 500 # how many times BPE glues the most common pair into a new token, more merges = bigger vocab
val_fraction  = 0.1 # last 10% of the token stream is kept aside to test the model on text it never trained on
stride        = 1 # how far the window slides each time, 1 = every possible window (neighbours overlap by 15 tokens)

# ---------- training ----------
batch_size   = 16 # how many windows we average the gradient over before one update
epochs       = 20 # how many full passes over the training windows
grad_clip    = 1.0 # if the whole gradient is longer than this, shrink it to this length so one bad batch cant blow up the model

# ---------- optimizer (AdamW) ----------
lr_max       = 1e-3 # biggest learning rate, reached right after warmup
lr_min       = 1e-4 # smallest learning rate, where the cosine decay ends
warmup_steps = 50 # number of steps spent ramping the learning rate up from near 0, a fresh model is fragile
beta1        = 0.9 # how much of the old gradient direction Adam keeps (momentum)
beta2        = 0.999 # how much of the old squared gradient Adam keeps (typical gradient size)
adam_eps     = 1e-8 # tiny number so Adam never divides by zero
weight_decay = 0.01 # each step shrinks the weight matrices slightly, only matrices, not biases or layer norm

# ---------- generation ----------
temperature    = 0.8 # below 1 = safer and more repetitive, above 1 = more random, 0 = always pick the top token
top_k          = 0 # only sample from the k most likely tokens, 0 = turned off
top_p          = 0.9 # only sample from the smallest group of tokens whose probabilities add up to 0.9
max_new_tokens = 50 # how many tokens talk.py writes after your prompt

# ---------- misc ----------
seed = 42 # fixes the random numbers so a run can be repeated exactly

# ---------- sanity checks ----------
assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
assert seq_len <= context_len, "a training window cannot be longer than the position table"
head_dim = emb_dim // num_heads # width of one head, 32 / 2 = 16
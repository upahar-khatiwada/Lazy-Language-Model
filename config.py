context_len = 32 # suppose 5 tokens then theres 5x32 rows, the model can look 32 texts at once
emb_dim     = 32 # size of the vector that flows through the entire model with same width at each layer
num_heads   = 2 # emb_dim must be divisible by this, model splits each vector into 2 halves of 16 numbers and runs attention separately
ff_dim      = 128 # in the feed forward step, vector is stretched from 32 to 128 temporarily to allow model compute richer things
num_layers  = 2 # number of layers of the LLM
seq_len     = 16 # each training example is a 16 rows table, we cut the input corpus into pieces of 16 tokens
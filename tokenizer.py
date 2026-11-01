import re

UNK = "<unk>"
PRE_TOKEN_RE = re.compile(r"\w+|\s|[^\w\s]")

def count_words(text):
    counts = {}
    for word in PRE_TOKEN_RE.findall(text):
        counts[word] = counts.get(word, 0) + 1
    return counts

def merge_tokens(tokens, pair, merged):
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def train_bpe(word_freqs, max_merges=1000):
    word_splits = {}
    for word in word_freqs:
        word_splits[word] = list(word)

    merges = []

    for step in range(max_merges):
        pair_freq = {}
        for word in word_splits:
            tokens = word_splits[word]
            weight = word_freqs[word]
            for i in range(len(tokens) - 1):
                key = (tokens[i], tokens[i + 1])
                pair_freq[key] = pair_freq.get(key, 0) + weight

        if len(pair_freq) == 0:
            break

        best_pair = None
        best_count = 0
        for pair in pair_freq:
            if pair_freq[pair] > best_count:
                best_count = pair_freq[pair]
                best_pair = pair

        if best_pair is None:
            break

        a, b = best_pair
        merged = a + b

        for word in word_splits:
            word_splits[word] = merge_tokens(word_splits[word], best_pair, merged)

        merges.append({"pair": best_pair, "merged": merged, "frequency": best_count})

    return merges

def apply_merges(text, merges):
    result = []

    for pre_token in PRE_TOKEN_RE.findall(text):
        tokens = list(pre_token)

        for m in merges:
            tokens = merge_tokens(tokens, m["pair"], m["merged"])
        result.extend(tokens)

    return result


def train_bpe_on_text(text, max_merges=1000):
    return train_bpe(count_words(text), max_merges)

class Tokenizer:
    def __init__(self, merges, stoi, itos):
        self.merges = merges
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(itos)

    def encode(self, text):
        ids = []
        for token in apply_merges(text, self.merges):
            ids.append(self.stoi.get(token, self.stoi[UNK]))
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.itos[i])
        return "".join(tokens)


def build_tokenizer(text, max_merges=1000):
    merges = train_bpe_on_text(text, max_merges)

    itos = [UNK]
    seen = {UNK}

    for ch in text:
        if ch not in seen:
            seen.add(ch)
            itos.append(ch)

    for m in merges:
        if m["merged"] not in seen:
            seen.add(m["merged"])
            itos.append(m["merged"])
    
    stoi = {}
    for i in range(len(itos)):
        stoi[itos[i]] = i

    return Tokenizer(merges, stoi, itos)
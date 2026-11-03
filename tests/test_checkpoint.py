import json

import numpy as np
import pytest

from checkpoint import save_checkpoint, load_checkpoint
from tokenizer import build_tokenizer
from transformer import ModelConfig, init_params, forward

CORPUS = "low low low lower newest newest\nthe low newest tower\n"


@pytest.fixture
def saved(tmp_path):
    tok = build_tokenizer(CORPUS, max_merges=8)
    cfg = ModelConfig(tok.vocab_size, context_len=8, emb_dim=4,
                      num_heads=2, ff_dim=8, num_layers=2)
    params = init_params(cfg, np.random.default_rng(0))
    prefix = str(tmp_path / "ckpt" / "model")        # "ckpt" does not exist yet
    save_checkpoint(prefix, params, cfg, tok)
    return prefix, params, cfg, tok


def test_save_creates_the_three_files(saved):
    prefix = saved[0]
    for ext in (".npz", ".json", ".tok.json"):
        assert open(prefix + ext, "rb").read()       # exists and is not empty


def test_config_round_trips(saved):
    prefix, _, cfg, _ = saved
    assert load_checkpoint(prefix)[1] == cfg


def test_weights_round_trip_exactly(saved):
    prefix, params, _, _ = saved
    loaded = load_checkpoint(prefix)[0]
    assert loaded.keys() == params.keys()
    for k in params:
        assert np.array_equal(loaded[k], params[k]), k


def test_tokenizer_round_trips(saved):
    prefix, _, _, tok = saved
    t2 = load_checkpoint(prefix)[2]
    assert t2.itos == tok.itos
    assert t2.stoi == tok.stoi
    assert t2.vocab_size == tok.vocab_size
    for s in ["low newest", "lower tower", "  low  ", "low!"]:   # "!" was never seen
        assert t2.encode(s) == tok.encode(s), s
        assert t2.decode(t2.encode(s)) == tok.decode(tok.encode(s)), s
    assert t2.encode("low!")[-1] == 0                # unseen character is still <unk>


def test_loaded_model_gives_identical_logits(saved):
    prefix, params, cfg, tok = saved
    p2, c2, t2 = load_checkpoint(prefix)
    ids = np.array(tok.encode("low newest tower"))
    a, _ = forward(params, ids, cfg)
    b, _ = forward(p2, ids, c2)
    assert np.array_equal(a, b)


def test_non_ascii_characters_survive(tmp_path):
    tok = build_tokenizer("café café über\n", max_merges=3)
    cfg = ModelConfig(tok.vocab_size, 8, 4, 2, 8, 1)
    prefix = str(tmp_path / "m")
    save_checkpoint(prefix, init_params(cfg, np.random.default_rng(0)), cfg, tok)
    t2 = load_checkpoint(prefix)[2]
    assert t2.decode(t2.encode("café über")) == "café über"


def test_tok_file_is_plain_json(saved):
    d = json.load(open(saved[0] + ".tok.json", encoding="utf-8"))
    assert set(d) == {"merges", "itos"}
    assert d["itos"][0] == "<unk>"
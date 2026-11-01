import pytest

from tokenizer import (
    UNK, count_words, merge_tokens, train_bpe_on_text, build_tokenizer,
)

CORPUS = "low low low lower newest newest"


def test_count_words_splits_words_spaces_punctuation():
    assert count_words("a a b!") == {"a": 2, " ": 2, "b": 1, "!": 1}


def test_merge_tokens_simple():
    assert merge_tokens(["a", "b", "a", "b"], ("a", "b"), "ab") == ["ab", "ab"]


def test_merge_tokens_overlap_is_left_to_right():
    # a a a  ->  (aa) a, not a (aa)
    assert merge_tokens(["a", "a", "a"], ("a", "a"), "aa") == ["aa", "a"]


def test_first_three_merges_match_hand_trace():
    merges = train_bpe_on_text(CORPUS, max_merges=3)
    got = [(m["pair"], m["frequency"]) for m in merges]
    assert got == [(("l", "o"), 4), (("lo", "w"), 4), (("n", "e"), 2)]


@pytest.fixture
def tok():
    return build_tokenizer(CORPUS, max_merges=5)


def test_unk_is_id_zero(tok):
    assert tok.itos[0] == UNK


def test_no_characters_dropped(tok):
    ids = tok.encode("lot")          # 'lo' + 't'
    assert len(ids) == 2
    assert 0 not in ids


def test_unseen_character_becomes_unk(tok):
    assert tok.encode("low!")[-1] == 0


@pytest.mark.parametrize("s", ["low newest", "lower lower", "lot", "  low  "])
def test_round_trip(tok, s):
    assert tok.decode(tok.encode(s)) == s
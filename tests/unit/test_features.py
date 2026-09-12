# test_features.py
# Purpose: unit tests for the feature layer — normaliser, vectoriser, vocabulary, encoder.
# Author: Colile Sibanda

from __future__ import annotations

from phishlens.config.loader import load_config
from phishlens.features.encoder import encode
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise
from phishlens.features.vectoriser import fit_tfidf, transform
from phishlens.features.vocabulary import PAD_INDEX, UNK_INDEX, build_vocabulary

CONFIG = load_config()


def test_normalise_replaces_url_email_and_number() -> None:
    text = "Visit http://evil.example/login or mail admin@example.com, ref 483920."
    result = normalise_text(text, CONFIG)
    assert "<url>" in result
    assert "<email>" in result
    assert "<num>" in result
    assert "http://" not in result
    assert "483920" not in result


def test_normalise_is_idempotent() -> None:
    text = "Click http://evil.example NOW, call 0123456789 or email a@b.com!!"
    once = normalise_text(text, CONFIG)
    twice = normalise_text(normalise_text(text, CONFIG), CONFIG)
    assert once == twice


def test_normalise_empty_and_non_string() -> None:
    assert normalise_text("", CONFIG) == ""
    assert normalise_text(None, CONFIG) == ""  # type: ignore[arg-type]


def test_tokenise_splits_words_and_keeps_placeholders() -> None:
    tokens = tokenise("click <url> now, call <num> today")
    assert tokens == ["click", "<url>", "now", "call", "<num>", "today"]


def test_tokenise_empty() -> None:
    assert tokenise("") == []


def test_vectoriser_fitted_on_train_only() -> None:
    train_texts = [
        "free money now click here",
        "free money now click here again",
        "meeting agenda attached for review",
        "meeting agenda attached for the team",
    ]
    vectoriser = fit_tfidf(train_texts, CONFIG)

    # A token that appears only in "validation" text, never in train, must
    # not appear in the fitted vocabulary — the signature makes misuse
    # awkward, but this proves the fit itself never saw held-out text.
    assert "quarterly" not in vectoriser.vocabulary_

    other_texts = ["quarterly report review", "free free free money"]
    matrix = transform(vectoriser, other_texts, CONFIG)
    assert matrix.shape[0] == len(other_texts)
    assert matrix.shape[1] == len(vectoriser.vocabulary_)


def test_vectoriser_transform_unseen_tokens_are_ignored_not_errored() -> None:
    vectoriser = fit_tfidf(["hello world", "hello again world"], CONFIG)
    matrix = transform(vectoriser, ["completely unseen vocabulary here"], CONFIG)
    assert matrix.shape[0] == 1


def test_build_vocabulary_has_pad_and_unk_at_fixed_indices() -> None:
    vocabulary = build_vocabulary(["free money now", "money money money"], CONFIG)
    assert vocabulary.token_to_index[vocabulary.pad_token] == PAD_INDEX
    assert vocabulary.token_to_index[vocabulary.unk_token] == UNK_INDEX


def test_build_vocabulary_respects_max_size() -> None:
    small_config = _config_with_vocab(max_size=5, min_frequency=1)
    texts = ["alpha beta gamma delta epsilon zeta eta theta"]
    vocabulary = build_vocabulary(texts, small_config)
    assert vocabulary.size() <= 5


def test_build_vocabulary_respects_min_frequency() -> None:
    config = _config_with_vocab(max_size=100, min_frequency=2)
    texts = ["common common common rare"]
    vocabulary = build_vocabulary(texts, config)
    assert "common" in vocabulary.token_to_index
    assert "rare" not in vocabulary.token_to_index


def test_build_vocabulary_out_of_vocabulary_maps_to_unk() -> None:
    vocabulary = build_vocabulary(["seen tokens only"], CONFIG)
    assert vocabulary.encode_token("never_seen_token_xyz") == UNK_INDEX


def test_vocabulary_save_and_load_round_trip(tmp_path) -> None:
    vocabulary = build_vocabulary(["free money now", "meeting notes"], CONFIG)
    path = tmp_path / "vocab.json"
    vocabulary.save(path)
    reloaded = vocabulary.load(path)
    assert reloaded.token_to_index == vocabulary.token_to_index
    assert reloaded.pad_token == vocabulary.pad_token
    assert reloaded.unk_token == vocabulary.unk_token


def test_encode_round_trips_a_known_sentence() -> None:
    vocabulary = build_vocabulary(["free money now available today"], CONFIG)
    sequences = encode(["free money now"], vocabulary, CONFIG)
    max_length = CONFIG.features.max_sequence_length

    assert sequences.shape == (1, max_length)
    tokens = tokenise(normalise_text("free money now", CONFIG))
    for position, token in enumerate(tokens):
        assert sequences[0, position] == vocabulary.encode_token(token)


def test_encode_pads_short_sequences() -> None:
    vocabulary = build_vocabulary(["hello"], CONFIG)
    sequences = encode(["hello"], vocabulary, CONFIG)
    assert sequences[0, 1] == PAD_INDEX


def test_encode_truncates_long_sequences() -> None:
    config = _config_with_max_length(3)
    vocabulary = build_vocabulary(["one two three four five"], config)
    sequences = encode(["one two three four five"], vocabulary, config)
    assert sequences.shape == (1, 3)


def test_encode_unknown_token_maps_to_unk_index() -> None:
    vocabulary = build_vocabulary(["only these words appear"], CONFIG)
    sequences = encode(["totally_unfamiliar_word"], vocabulary, CONFIG)
    assert sequences[0, 0] == UNK_INDEX


def _config_with_vocab(max_size: int, min_frequency: int):
    from dataclasses import replace

    features = replace(
        CONFIG.features,
        vocabulary=replace(
            CONFIG.features.vocabulary, max_size=max_size, min_frequency=min_frequency
        ),
    )
    return replace(CONFIG, features=features)


def _config_with_max_length(max_length: int):
    from dataclasses import replace

    features = replace(CONFIG.features, max_sequence_length=max_length)
    return replace(CONFIG, features=features)

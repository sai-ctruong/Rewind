"""Vietnamese/English query normalization.

The rule under test: the original query is never replaced. CLIP must keep seeing what the
user typed, while lexical channels get folded and expanded views. Negation and temporal
words survive, because dropping either changes what the query means.
"""
from __future__ import annotations

import unicodedata

import pytest

from aic2026.query_normalization import (
    ENGLISH_VIETNAMESE_TERMS,
    NEGATION_TERMS,
    QUERY_VOCABULARY_VERSION,
    TEMPORAL_TERMS,
    VIETNAMESE_ENGLISH_TERMS,
    fold_accents,
    label_tokens,
    normalize_label,
    normalize_query,
    tokenize_query,
)


# ------------------------------------------------------------ original preserved


def test_the_original_query_is_never_replaced() -> None:
    text = "một người đang đi bộ"
    representation = normalize_query(text)
    assert representation.original == text
    # CLIP receives the user's own words, unfolded and unexpanded.
    assert representation.dense_query == text
    assert "nguoi" not in representation.dense_query
    assert representation.accent_folded == "mot nguoi dang di bo"


def test_whitespace_normalization_does_not_alter_words() -> None:
    representation = normalize_query("  xe   máy  trên đường \n")
    assert representation.normalized_whitespace == "xe máy trên đường"
    assert representation.dense_query == "xe máy trên đường"


# ------------------------------------------------------------------- folding


@pytest.mark.parametrize(
    "text,expected",
    [
        ("người", "nguoi"),
        ("xe máy", "xe may"),
        ("đường", "duong"),
        ("Đường", "Duong"),
        ("thành phố Hồ Chí Minh", "thanh pho Ho Chi Minh"),
        ("chiếc ô tô màu đỏ", "chiec o to mau do"),
        ("ĐI BỘ", "DI BO"),
        ("", ""),
    ],
)
def test_accent_folding(text: str, expected: str) -> None:
    assert fold_accents(text) == expected


def test_folding_is_stable_across_nfc_and_nfd() -> None:
    nfc = unicodedata.normalize("NFC", "người đi đường")
    nfd = unicodedata.normalize("NFD", "người đi đường")
    assert nfc != nfd, "the fixture must genuinely differ in normal form"
    assert fold_accents(nfc) == fold_accents(nfd) == "nguoi di duong"
    assert normalize_query(nfc).accent_folded == normalize_query(nfd).accent_folded
    assert normalize_query(nfc).tokens_folded == normalize_query(nfd).tokens_folded


def test_d_with_stroke_is_handled_because_it_does_not_decompose() -> None:
    # đ carries no combining mark, so NFD alone would leave it untouched.
    assert "đ" not in fold_accents("đường đi")
    assert fold_accents("đường đi") == "duong di"
    assert fold_accents("Đà Nẵng") == "Da Nang"


def test_ascii_text_is_unchanged() -> None:
    assert fold_accents("a person riding a motorcycle") == "a person riding a motorcycle"


# ------------------------------------------------------------------ negation


def test_negation_tokens_are_preserved_not_stripped() -> None:
    representation = normalize_query("không có xe máy")
    assert "không" in representation.tokens_original
    assert "khong" in representation.tokens_folded
    assert representation.negated_tokens == ("không",)


@pytest.mark.parametrize("text", ["không có xe", "no car", "a street without cars"])
def test_negated_terms_are_not_offered_as_positive_object_terms(text: str) -> None:
    representation = normalize_query(text)
    assert representation.negated_tokens
    # The thing being negated must not become a positive retrieval term.
    positives = set(representation.object_terms)
    negatives = set(representation.negated_terms)
    assert negatives, f"nothing was marked negated in {text!r}"
    assert not (positives & negatives)


def test_negation_scopes_over_the_following_words_only() -> None:
    representation = normalize_query("không có xe nhưng có người")
    negated = set(representation.negated_terms)
    positives = set(representation.object_terms)
    # "xe" follows the negation; "nguoi" is far enough away to stay positive.
    assert "xe" in negated
    assert "car" in negated or "vehicle" in negated
    assert "nguoi" in positives


def test_a_query_without_negation_marks_nothing() -> None:
    representation = normalize_query("xe máy trên đường")
    assert representation.negated_tokens == ()
    assert representation.negated_terms == ()
    assert "motorcycle" in representation.object_terms


# ------------------------------------------------------------------ temporal


@pytest.mark.parametrize(
    "text,expected",
    [
        ("người đi trước rồi dừng lại", {"trước", "rồi"}),
        ("the person sits down then stands up", {"then"}),
        ("a car stops before the intersection", {"before"}),
        ("sau đó xe chạy tiếp", {"sau", "đó", "tiếp"} & TEMPORAL_TERMS | {"sau", "tiếp"}),
    ],
)
def test_temporal_markers_are_exposed_not_removed(text: str, expected: set[str]) -> None:
    representation = normalize_query(text)
    markers = set(representation.temporal_markers)
    assert markers & expected, f"{text!r} -> {markers}"
    # They also survive in the token views for downstream consumers.
    for marker in markers:
        assert marker in representation.tokens_original


# ---------------------------------------------------------------- expansion


def test_vietnamese_terms_expand_to_detector_labels() -> None:
    representation = normalize_query("một người và một chiếc xe máy")
    terms = set(representation.object_terms)
    assert "person" in terms
    assert "motorcycle" in terms
    # The Vietnamese tokens are kept too, so Vietnamese lexical fields still match.
    assert "nguoi" in terms and "xe" in terms


def test_the_longest_phrase_wins_over_its_prefix() -> None:
    motorcycle = set(normalize_query("xe máy").object_terms)
    assert "motorcycle" in motorcycle
    # A bare "xe" would have expanded to car/vehicle; the phrase must take precedence.
    assert "car" not in motorcycle
    assert "car" in set(normalize_query("chiếc xe").object_terms)


def test_english_queries_reach_vietnamese_lexical_fields() -> None:
    terms = set(normalize_query("a person riding a motorcycle").object_terms)
    assert "person" in terms
    assert "nguoi" in terms
    assert "xe may" in terms


def test_expansion_is_deterministic_and_carries_provenance() -> None:
    first = normalize_query("người đi xe đạp")
    second = normalize_query("người đi xe đạp")
    assert [item.to_dict() for item in first.expanded_terms] == [
        item.to_dict() for item in second.expanded_terms
    ]
    sources = {item.source for item in first.expanded_terms}
    assert sources <= {"query_token", "vi_en_vocabulary", "en_vi_vocabulary"}
    bicycle = next(item for item in first.expanded_terms if item.term == "bicycle")
    assert bicycle.origin == "xe dap"
    assert bicycle.source == "vi_en_vocabulary"


def test_unknown_words_pass_through_unchanged() -> None:
    representation = normalize_query("một cái xyzzy kỳ lạ")
    assert "xyzzy" in representation.object_terms
    # No fabricated translation for a word that is not in the vocabulary.
    assert not any(
        item.origin == "xyzzy" and item.source != "query_token"
        for item in representation.expanded_terms
    )


def test_expansion_can_be_switched_off() -> None:
    representation = normalize_query("xe máy", expand=False)
    assert representation.expanded_terms == ()
    assert representation.object_terms == ()
    assert representation.accent_folded == "xe may"


def test_the_vocabulary_is_versioned_and_bidirectional() -> None:
    assert QUERY_VOCABULARY_VERSION >= 1
    assert normalize_query("x").vocabulary_version == QUERY_VOCABULARY_VERSION
    assert VIETNAMESE_ENGLISH_TERMS["nguoi"][0] == "person"
    assert "nguoi" in ENGLISH_VIETNAMESE_TERMS["person"]
    # Every vocabulary key is already folded, or lookups would silently miss.
    for key in VIETNAMESE_ENGLISH_TERMS:
        assert fold_accents(key) == key, f"vocabulary key {key!r} is not accent-folded"


# ---------------------------------------------------------------- label side


def test_labels_are_normalized_the_same_way_as_queries() -> None:
    assert normalize_label("Motorcycles") == "motorcycle"
    assert normalize_label("Traffic_Light") == "traffic light"
    assert normalize_label("Người") == "nguoi"
    assert label_tokens("Mobile Phone") == ("mobile", "phone")


def test_tokenizer_drops_punctuation_but_keeps_words() -> None:
    assert tokenize_query("một người, đang đi bộ!") == ("một", "người", "đang", "đi", "bộ")
    assert tokenize_query("") == ()


def test_number_terms_are_exposed() -> None:
    assert normalize_query("3 người").number_terms == ("3",)
    assert normalize_query("người").number_terms == ()


def test_negation_vocabulary_covers_both_languages() -> None:
    assert {"khong", "không", "no", "not", "without"} <= NEGATION_TERMS

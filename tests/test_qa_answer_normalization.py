"""Answer types and Vietnamese/English normalization, end to end.

The type matters most for one word: Vietnamese `không` means "no" for a boolean question
and "zero" for a counting one. Without a declared type it is left alone, because
guessing wrong silently changes the submitted answer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.engine import AICCompetitionEngine
from aic2026.qa import (
    ANSWER_TYPE_AUTO,
    ANSWER_TYPE_BOOLEAN,
    ANSWER_TYPE_COLOR,
    ANSWER_TYPE_NUMBER,
    ANSWER_TYPE_SHORT_TEXT,
    answer_matches_type,
    build_answer_prompt,
    canonical_answer_type,
    is_unknown_answer,
    normalize_answer,
)
from aic2026.text_encoder import HashingTextEncoder
from tests.qa_support import ScriptedQAAnswerer, make_qa_config, make_qa_root


# ---------------------------------------------------------------- type mapping


def test_ui_answer_type_spellings_are_accepted() -> None:
    assert canonical_answer_type(None) == ANSWER_TYPE_AUTO
    assert canonical_answer_type("") == ANSWER_TYPE_AUTO
    assert canonical_answer_type("yes/no") == ANSWER_TYPE_BOOLEAN
    assert canonical_answer_type("text") == ANSWER_TYPE_SHORT_TEXT
    assert canonical_answer_type("Number") == ANSWER_TYPE_NUMBER
    assert canonical_answer_type("colour") == ANSWER_TYPE_COLOR
    with pytest.raises(ValueError, match="Unsupported expected_answer_type"):
        canonical_answer_type("haiku")


# -------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("three people", "3"),
        ("There are 4 people", "4"),
        ("ten", "10"),
        ("zero", "0"),
        ("một", "1"),
        ("Bốn", "4"),
        ("mười", "10"),
        ("hai người", "2"),
        ("có 7 chiếc xe", "7"),
    ],
)
def test_number_normalization_english_and_vietnamese(raw: str, expected: str) -> None:
    assert normalize_answer(raw, expected_type="number") == expected


# ------------------------------------------------------------------- booleans


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("yes, he does", "yes"),
        ("Yes", "yes"),
        ("true", "yes"),
        ("no", "no"),
        ("nope", "no"),
        ("có", "yes"),
        ("Đúng", "yes"),
        ("không", "no"),
        ("sai", "no"),
    ],
)
def test_boolean_normalization_english_and_vietnamese(raw: str, expected: str) -> None:
    assert normalize_answer(raw, expected_type="yes/no") == expected


def test_ambiguous_khong_respects_the_expected_type() -> None:
    # The whole reason `normalize_answer` takes a type.
    assert normalize_answer("không", expected_type="number") == "0"
    assert normalize_answer("không", expected_type="boolean") == "no"
    assert normalize_answer("khong", expected_type="number") == "0"
    assert normalize_answer("khong", expected_type="boolean") == "no"
    # With no type declared it is NOT guessed at.
    assert normalize_answer("không") == "không"
    assert normalize_answer("khong") == "khong"


# --------------------------------------------------------------------- colors


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Red", "red"),
        ("the car is red", "red"),
        ("xanh dương", "blue"),
        ("màu đỏ", "red"),
        ("trắng", "white"),
        ("Đen.", "black"),
    ],
)
def test_color_normalization(raw: str, expected: str) -> None:
    assert normalize_answer(raw, expected_type="color") == expected


def test_color_normalization_never_invents_a_colour() -> None:
    # A colour was asked for and none was said: the text survives, unchanged.
    assert normalize_answer("i cannot tell", expected_type="color") == "i cannot tell"
    assert answer_matches_type("i cannot tell", "color") is False


def test_longer_colour_phrases_win_over_their_prefix() -> None:
    assert normalize_answer("chiếc xe màu xanh dương", expected_type="color") == "blue"


# ----------------------------------------------------------------- short text


def test_short_text_is_only_minimally_normalized() -> None:
    assert normalize_answer("A red motorcycle!", expected_type="text") == "a red motorcycle"
    # No number or boolean substitution is applied to free text.
    assert normalize_answer("one way street", expected_type="text") == "one way street"
    assert normalize_answer("không", expected_type="short_text") == "không"


def test_auto_keeps_the_historical_behaviour() -> None:
    assert normalize_answer("  Bốn! ") == "4"
    assert normalize_answer("CÓ") == "yes"
    assert normalize_answer("xanh dương") == "blue"


# ------------------------------------------------------------------ validation


def test_answer_matches_type_checks_the_shape() -> None:
    assert answer_matches_type("3", "number") is True
    assert answer_matches_type("three", "number") is False
    assert answer_matches_type("yes", "boolean") is True
    assert answer_matches_type("maybe", "boolean") is False
    assert answer_matches_type("blue", "color") is True
    assert answer_matches_type("", "auto") is False
    assert answer_matches_type("unknown", "auto") is False
    assert answer_matches_type("a red car", "short_text") is True


@pytest.mark.parametrize("expected_type", ["number", "yes/no", "color", "text", None])
def test_an_unknown_answer_never_becomes_a_confident_one(expected_type) -> None:
    # "khong co mo ta" starts with "khong", so a naive boolean pass would report "no"
    # and a naive number pass would report "0". A refusal must survive as a refusal.
    for refusal in ("không có mô tả", "không xác định", "unknown", ""):
        result = normalize_answer(refusal, expected_type=expected_type)
        assert is_unknown_answer(result), f"{refusal!r} -> {result!r} under {expected_type}"
    assert normalize_answer("không có mô tả", expected_type="yes/no") != "no"
    assert normalize_answer("không xác định", expected_type="number") != "0"


def test_unknown_forms_are_recognized_in_both_languages() -> None:
    assert is_unknown_answer("") is True
    assert is_unknown_answer("unknown") is True
    assert is_unknown_answer("không xác định") is True
    assert is_unknown_answer("không có mô tả") is True
    assert is_unknown_answer("red") is False


def test_prompts_ask_for_a_submission_shaped_answer() -> None:
    assert "single integer" in build_answer_prompt("How many?", "number")
    assert "yes or no" in build_answer_prompt("Is it?", "yes/no")
    assert "colour name" in build_answer_prompt("What colour?", "color")
    assert "noun phrase" in build_answer_prompt("What is it?", "text")
    # Every prompt confines the answer to the one video whose frames were supplied.
    assert "one video" in build_answer_prompt("What?", None)


# ----------------------------------------------------------------- end to end


def build(tmp_path: Path, answers: dict[str, str], **qa):
    root = make_qa_root(tmp_path / "data")
    config = make_qa_config(root, tmp_path / "cache", tmp_path / "frames", **qa)
    engine, _ = AICCompetitionEngine.from_data_root(
        root,
        cache_dir=tmp_path / "cache",
        app_config=config,
        text_encoder=HashingTextEncoder(2),
        qa_answerer=ScriptedQAAnswerer(answers),
    )
    return engine


def test_expected_answer_type_reaches_normalization(tmp_path: Path) -> None:
    engine = build(tmp_path, {"L21_V001": "three people", "L21_V002": "hai"})
    predictions, info = engine.answer_qa(
        "people", "How many people?", top_k=20, expected_answer_type="number"
    )
    assert info["expected_answer_type"] == "number"
    answers = {p.video_id: p.answer for p in predictions}
    assert answers["L21_V001"] == "3"
    assert answers["L21_V002"] == "2"
    # The raw answer is kept beside the normalized one.
    raw = {p.video_id: p.qa["raw_answer"] for p in predictions}
    assert raw["L21_V001"] == "three people"


def test_boolean_type_reaches_normalization(tmp_path: Path) -> None:
    engine = build(tmp_path, {"L21_V001": "yes, he is", "L21_V002": "không"})
    predictions, _ = engine.answer_qa(
        "a person", "Is the person standing?", top_k=20, expected_answer_type="yes/no"
    )
    answers = {p.video_id: p.answer for p in predictions}
    assert answers["L21_V001"] == "yes"
    assert answers["L21_V002"] == "no"


def test_number_type_disambiguates_khong_end_to_end(tmp_path: Path) -> None:
    engine = build(tmp_path, {"L21_V001": "không", "L21_V002": "không"})
    predictions, _ = engine.answer_qa(
        "people", "How many people?", top_k=20, expected_answer_type="number"
    )
    assert {p.answer for p in predictions} == {"0"}


def test_default_answer_type_from_config_is_applied(tmp_path: Path) -> None:
    engine = build(tmp_path, {"L21_V001": "ba", "L21_V002": "ba"}, default_answer_type="number")
    predictions, info = engine.answer_qa("people", "How many?", top_k=20)
    assert info["expected_answer_type"] == "number"
    assert {p.answer for p in predictions} == {"3"}


def test_an_invalid_answer_type_is_rejected_rather_than_ignored(tmp_path: Path) -> None:
    engine = build(tmp_path, {"L21_V001": "red"})
    with pytest.raises(ValueError, match="Unsupported expected_answer_type"):
        engine.answer_qa("x", "y", top_k=5, expected_answer_type="sonnet")


def test_matching_the_requested_type_raises_reliability(tmp_path: Path) -> None:
    typed = build(tmp_path / "typed", {"L21_V001": "3", "L21_V002": "3"})
    untyped = build(tmp_path / "untyped", {"L21_V001": "3", "L21_V002": "3"})
    _, with_type = typed.answer_qa("x", "How many?", top_k=5, expected_answer_type="number")
    _, without = untyped.answer_qa("x", "How many?", top_k=5)
    assert with_type["answer_reliability_score"] > without["answer_reliability_score"]

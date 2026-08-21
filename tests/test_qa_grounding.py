import numpy as np

from aic2026.qa import (
    QAInput,
    build_evidence_retrieval_query,
    build_retrieval_query,
    normalize_answer,
    select_diverse_evidence,
)
from ingestion.schemas import KeyframeRecord


def record(i):
    return KeyframeRecord(str(i), "V", float(i), np.ones(2), objects=["car"] if i == 5 else [])


def test_retrieval_query_modes_do_not_blindly_mix_question() -> None:
    data = QAInput("person enters room", "what color is the shirt?")
    assert build_retrieval_query(data) == "person enters room"
    assert "what color" in build_retrieval_query(data, "event_plus_question")


def test_evidence_retrieval_query_adds_visual_question_cues() -> None:
    data = QAInput("nguoi dan ong dung truoc quay", "anh ta dang cam gi?")
    query = build_evidence_retrieval_query(data)
    assert "nguoi dan ong dung truoc quay" in query
    assert "person" in query and "man" in query
    assert "person holding object" in query


def test_retrieval_query_mode_evidence_is_available() -> None:
    data = QAInput("a woman on the street", "what color is the umbrella?")
    query = build_retrieval_query(data, "evidence")
    assert "a woman on the street" in query
    assert "umbrella" in query
    assert "color" in query


def test_diverse_evidence_is_ordered_and_bounded() -> None:
    selected = select_diverse_evidence([record(i) for i in range(12)], 5.0, count=5, query="car")
    assert len(selected) == 5
    assert [item.record.timestamp for item in selected] == sorted(item.record.timestamp for item in selected)
    assert any(item.record.id == "5" for item in selected)


def test_answer_normalization_numbers_yes_no_and_colors() -> None:
    assert normalize_answer("  Bốn! ") == "4"
    assert normalize_answer("CÓ") == "yes"
    assert normalize_answer("xanh dương") == "blue"

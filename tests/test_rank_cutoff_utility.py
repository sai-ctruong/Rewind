"""R1: the rank-utility function must match the official cutoffs exactly.

This is the one component in R1 that is not a heuristic — it is arithmetic about the
metric — so it is pinned against the scorer's own cutoff list rather than a copy.
"""
from __future__ import annotations

import pytest

from aic2026.metrics import TOP_KS, final_score_from_r_scores
from aic2026.rank_utility import (
    MAX_OFFICIAL_RANK,
    OFFICIAL_CUTOFFS,
    bucket_capacity,
    bucket_labels,
    bucket_of,
    marginal_utility,
    rank_cutoff_utility,
    utility_by_rank,
)


def test_cutoffs_come_from_the_scorer_not_a_copy() -> None:
    assert OFFICIAL_CUTOFFS == tuple(sorted(TOP_KS))
    assert MAX_OFFICIAL_RANK == 100


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (1, 1.0),
        (2, 0.8),
        (5, 0.8),
        (6, 0.6),
        (20, 0.6),
        (21, 0.4),
        (50, 0.4),
        (51, 0.2),
        (100, 0.2),
        (101, 0.0),
        (10_000, 0.0),
    ],
)
def test_utility_matches_the_official_cutoff_grid(rank, expected) -> None:
    assert rank_cutoff_utility(rank) == pytest.approx(expected)


def test_utility_equals_what_the_scorer_would_award() -> None:
    """Ground the function in the real scorer: one correct row at rank r."""
    for rank in (1, 3, 12, 40, 80):
        r_scores = [0.0] * (rank - 1) + [1.0]
        expected = final_score_from_r_scores(r_scores)["Final Score"]
        assert rank_cutoff_utility(rank) == pytest.approx(expected)


def test_utility_is_non_increasing() -> None:
    values = utility_by_rank(120)
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_rank_zero_is_rejected() -> None:
    with pytest.raises(ValueError):
        rank_cutoff_utility(0)


def test_it_is_a_step_function_not_a_decay() -> None:
    """Inside a bucket nothing changes; that is the property a smooth prior would miss."""
    assert rank_cutoff_utility(6) == rank_cutoff_utility(20)
    assert rank_cutoff_utility(21) == rank_cutoff_utility(50)
    assert rank_cutoff_utility(20) > rank_cutoff_utility(21)


def test_marginal_utility_is_zero_inside_a_bucket() -> None:
    assert marginal_utility(20, 6) == 0.0
    assert marginal_utility(50, 21) == 0.0
    # Crossing a cutoff is what pays.
    assert marginal_utility(21, 20) == pytest.approx(0.2)
    assert marginal_utility(6, 1) == pytest.approx(0.4)


def test_marginal_utility_of_a_demotion_is_zero_not_negative() -> None:
    assert marginal_utility(1, 50) == 0.0


@pytest.mark.parametrize(
    ("rank", "label"),
    [(1, "1"), (3, "2-5"), (7, "6-20"), (33, "21-50"), (99, "51-100"), (500, ">100")],
)
def test_bucket_labels_are_stable(rank, label) -> None:
    assert bucket_of(rank) == label


def test_bucket_capacity_covers_every_legal_slot() -> None:
    capacity = bucket_capacity()
    assert sum(capacity.values()) == MAX_OFFICIAL_RANK
    assert capacity["1"] == 1 and capacity["2-5"] == 4
    assert capacity["51-100"] == 50
    assert tuple(capacity) == bucket_labels()


def test_low_buckets_hold_most_of_the_slots_and_least_of_the_value() -> None:
    """Why the tail is worth diversifying: many slots, each worth a fifth of rank 1."""
    capacity = bucket_capacity()
    assert capacity["51-100"] > capacity["1"] * 10
    assert rank_cutoff_utility(51) == pytest.approx(rank_cutoff_utility(1) / 5)

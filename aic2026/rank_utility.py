"""The official metric's rank geometry, as a pure function.

AIC 2026 scores each query at R@1, R@5, R@20, R@50 and R@100, and the Final Score is
their mean. That makes rank value a step function, not a smooth decay: for a binary-
correct task, a first correct row at rank 2 and at rank 5 are worth exactly the same,
and a row at rank 21 is worth twice a row at rank 51.

Nothing here predicts whether a row IS correct. `rank_cutoff_utility(r)` answers a
different question — "if the first correct row landed at rank r, what fraction of the
Final Score could this query still earn?" — which is arithmetic about the metric and
holds with no ground truth at all. R1 uses it as an ALLOCATION PRIOR: where an
improvement would be worth the most if one happened.
"""
from __future__ import annotations

from typing import Sequence

from .metrics import TOP_KS

# The official cutoffs, ascending. Imported rather than redeclared so this file cannot
# drift from the scorer.
OFFICIAL_CUTOFFS: tuple[int, ...] = tuple(sorted(int(k) for k in TOP_KS))
MAX_OFFICIAL_RANK: int = OFFICIAL_CUTOFFS[-1]

# Bucket boundaries implied by the cutoffs: ranks inside one bucket are worth the same.
CUTOFF_BUCKETS: tuple[tuple[int, int], ...] = tuple(
    (low, high)
    for low, high in zip((1,) + tuple(k + 1 for k in OFFICIAL_CUTOFFS[:-1]), OFFICIAL_CUTOFFS)
)


def rank_cutoff_utility(rank: int, cutoffs: Sequence[int] = OFFICIAL_CUTOFFS) -> float:
    """Fraction of the Final Score still reachable if the first correct row is at `rank`.

    Not a probability and not a predicted correctness: it is the metric's own shape.
    Rank 1 -> 1.0, ranks 2-5 -> 0.8, 6-20 -> 0.6, 21-50 -> 0.4, 51-100 -> 0.2, beyond the
    last cutoff -> 0.0.
    """
    ordered = sorted(int(k) for k in cutoffs)
    if not ordered:
        return 0.0
    position = int(rank)
    if position < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    satisfied = sum(1 for cutoff in ordered if position <= cutoff)
    return satisfied / len(ordered)


def bucket_of(rank: int, cutoffs: Sequence[int] = OFFICIAL_CUTOFFS) -> str:
    """Label of the cutoff bucket a rank falls in, e.g. `6-20`."""
    ordered = sorted(int(k) for k in cutoffs)
    low = 1
    for cutoff in ordered:
        if int(rank) <= cutoff:
            return f"{low}" if low == cutoff else f"{low}-{cutoff}"
        low = cutoff + 1
    return f">{ordered[-1]}"


def bucket_labels(cutoffs: Sequence[int] = OFFICIAL_CUTOFFS) -> tuple[str, ...]:
    ordered = sorted(int(k) for k in cutoffs)
    low, labels = 1, []
    for cutoff in ordered:
        labels.append(f"{low}" if low == cutoff else f"{low}-{cutoff}")
        low = cutoff + 1
    return tuple(labels)


def marginal_utility(from_rank: int, to_rank: int, cutoffs: Sequence[int] = OFFICIAL_CUTOFFS) -> float:
    """Final-Score opportunity gained by moving a row from one rank to a better one.

    Zero inside a bucket: promoting a row from rank 20 to rank 6 crosses no cutoff and
    changes no score. That is the property a metric-aware allocator needs and a smooth
    rank-decay heuristic would miss.
    """
    return max(
        0.0,
        rank_cutoff_utility(to_rank, cutoffs) - rank_cutoff_utility(from_rank, cutoffs),
    )


def bucket_capacity(cutoffs: Sequence[int] = OFFICIAL_CUTOFFS) -> dict[str, int]:
    """How many of the 100 legal slots each bucket owns."""
    ordered = sorted(int(k) for k in cutoffs)
    low, out = 1, {}
    for cutoff in ordered:
        out[f"{low}" if low == cutoff else f"{low}-{cutoff}"] = cutoff - low + 1
        low = cutoff + 1
    return out


def utility_by_rank(count: int = MAX_OFFICIAL_RANK) -> tuple[float, ...]:
    """Utility for ranks 1..count, for plotting and for tests."""
    return tuple(rank_cutoff_utility(rank) for rank in range(1, int(count) + 1))


__all__ = [
    "CUTOFF_BUCKETS",
    "MAX_OFFICIAL_RANK",
    "OFFICIAL_CUTOFFS",
    "bucket_capacity",
    "bucket_labels",
    "bucket_of",
    "marginal_utility",
    "rank_cutoff_utility",
    "utility_by_rank",
]

"""Training-free uncertainty signals and a transparent compute budget controller.

Two deliberate constraints shape everything here.

**Nothing is learned.** Every signal is an arithmetic function of scores the pipeline
already computed. There is no router to train, no threshold fitted to outcomes, and no
component whose value was chosen by looking at results — there are no results to look at,
because no ground truth exists.

**Nothing is called a probability.** `uncertainty` is a bounded score in [0, 1] built
from margins and disagreement. It is a *ranking* of which queries look less settled, not
an estimate of being wrong. `expected_gain_proxy` is likewise a proxy: it says an action
plausibly matters more here than there, never that it will improve an answer.

The controller spends a hard budget. It can stop early, it can refuse an action, and it
can run out — what it cannot do is exceed `max_cost_units`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .rank_utility import rank_cutoff_utility

# --- action names -------------------------------------------------------------------
# Each is an expensive, optional piece of work the controller may choose to buy.
ACTION_DEEPEN_CHANNEL = "DEEPEN_CHANNEL"
ACTION_OFFICIAL_GRID_REFINE = "OFFICIAL_GRID_REFINE"
ACTION_SPARSE_VIDEO_SAMPLE = "SPARSE_VIDEO_SAMPLE"
ACTION_DENSE_TEMPORAL_ZOOM = "DENSE_TEMPORAL_ZOOM"
ACTION_QA_VLM_CALL = "QA_VLM_CALL"

ACTIONS = (
    ACTION_DEEPEN_CHANNEL,
    ACTION_OFFICIAL_GRID_REFINE,
    ACTION_SPARSE_VIDEO_SAMPLE,
    ACTION_DENSE_TEMPORAL_ZOOM,
    ACTION_QA_VLM_CALL,
)

# Order-of-magnitude cost of one unit of each action on a CPU-only machine, in the same
# arbitrary units as `QueryCost.cost_proxy`. Reading an already-indexed CLIP vector is
# nearly free; decoding a frame is not; a VLM call dwarfs both. These are NOT tuned.
ACTION_UNIT_COST = {
    ACTION_DEEPEN_CHANNEL: 0.5,
    ACTION_OFFICIAL_GRID_REFINE: 0.05,
    ACTION_SPARSE_VIDEO_SAMPLE: 4.0,
    ACTION_DENSE_TEMPORAL_ZOOM: 4.0,
    ACTION_QA_VLM_CALL: 200.0,
}

EPSILON = 1e-9


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, float(value)))


# ------------------------------------------------------------------ uncertainty


@dataclass(frozen=True)
class UncertaintySignals:
    """Transparent, training-free views of how settled a result looks.

    Every component is logged so a reader can see which one drove an allocation. None of
    them is validated against correctness; that is exactly what cannot be checked here.
    """

    score_margin: float = 0.0
    channel_disagreement: float = 0.0
    support_concentration: float = 0.0
    temporal_ambiguity: float = 0.0
    enabled: tuple[str, ...] = (
        "score_margin",
        "channel_disagreement",
        "support_concentration",
        "temporal_ambiguity",
    )
    components: dict[str, Any] = field(default_factory=dict)

    @property
    def uncertainty(self) -> float:
        """Mean of the ENABLED components, each already in [0, 1]. Not a probability.

        A disabled component is excluded from the mean rather than counted as zero:
        counting it as zero would quietly report every query as more settled.
        """
        parts = [float(getattr(self, name)) for name in self.enabled]
        if not parts:
            return 0.0
        return round(sum(parts) / len(parts), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_margin": round(self.score_margin, 6),
            "channel_disagreement": round(self.channel_disagreement, 6),
            "support_concentration": round(self.support_concentration, 6),
            "temporal_ambiguity": round(self.temporal_ambiguity, 6),
            "enabled_components": list(self.enabled),
            "uncertainty": self.uncertainty,
            "components": self.components,
            "note": (
                "Training-free structural signals in [0, 1]. Higher means less settled, "
                "NOT more likely to be wrong: no ground truth exists to calibrate that."
            ),
        }


def margin_uncertainty(scores: Sequence[float]) -> float:
    """1 - relative gap between the two best scores. One candidate alone is settled."""
    values = [float(v) for v in scores if math.isfinite(float(v))]
    if len(values) < 2:
        return 0.0
    values.sort(reverse=True)
    best, second = values[0], values[1]
    scale = max(abs(best), EPSILON)
    return _clamp(1.0 - (best - second) / scale)


def channel_disagreement(rank_lists: Iterable[Sequence[str]], *, top_n: int = 10) -> float:
    """How little the active channels agree on their heads.

    Full agreement -> 0. Disjoint heads -> 1. With fewer than two channels there is
    nothing to disagree about, so the signal is 0 rather than an invented value.
    """
    heads = [tuple(items[:top_n]) for items in rank_lists if items]
    if len(heads) < 2:
        return 0.0
    sets = [set(head) for head in heads]
    union = set().union(*sets)
    if not union:
        return 0.0
    intersection = set(sets[0]).intersection(*sets[1:])
    return _clamp(1.0 - len(intersection) / len(union))


def support_concentration(video_ids: Sequence[str]) -> float:
    """How spread the head is across videos.

    All from one video -> 0 (a concentrated hypothesis). Every row a different video -> 1.
    """
    items = [str(v) for v in video_ids if v]
    if len(items) < 2:
        return 0.0
    return _clamp((len(set(items)) - 1) / (len(items) - 1))


def temporal_ambiguity(timestamps: Sequence[float], *, window_s: float = 2.0) -> float:
    """How many separated temporal regions compete in the head.

    One tight cluster -> 0. Every candidate its own region -> 1.
    """
    values = sorted(float(t) for t in timestamps if math.isfinite(float(t)))
    if len(values) < 2:
        return 0.0
    regions = 1
    for previous, current in zip(values, values[1:]):
        if current - previous > float(window_s):
            regions += 1
    return _clamp((regions - 1) / (len(values) - 1))


def kis_uncertainty(
    candidates: Sequence[Any],
    *,
    channel_heads: Optional[Iterable[Sequence[str]]] = None,
    head_size: int = 10,
    window_s: float = 2.0,
    use_margin: bool = True,
    use_channel_disagreement: bool = True,
    use_temporal_ambiguity: bool = True,
) -> UncertaintySignals:
    """Uncertainty for one KIS result, from the candidate head only.

    Each component can be switched off by configuration. Support concentration is always
    on: it costs one `set()` over the head and is the only signal that survives when a
    single channel is active.
    """
    head = list(candidates[:head_size])
    scores = [float(getattr(item, "score", 0.0)) for item in head]
    videos = [str(getattr(item, "video_id", "")) for item in head]
    stamps = [float(getattr(item, "timestamp", 0.0) or 0.0) for item in head]
    enabled = ["support_concentration"]
    if use_margin:
        enabled.append("score_margin")
    if use_channel_disagreement:
        enabled.append("channel_disagreement")
    if use_temporal_ambiguity:
        enabled.append("temporal_ambiguity")
    return UncertaintySignals(
        score_margin=margin_uncertainty(scores) if use_margin else 0.0,
        channel_disagreement=(
            channel_disagreement(channel_heads or (), top_n=head_size)
            if use_channel_disagreement
            else 0.0
        ),
        support_concentration=support_concentration(videos),
        temporal_ambiguity=(
            temporal_ambiguity(stamps, window_s=window_s) if use_temporal_ambiguity else 0.0
        ),
        enabled=tuple(sorted(enabled)),
        components={
            "head_size": len(head),
            "distinct_videos": len(set(videos)),
            "best_score": round(max(scores), 6) if scores else 0.0,
        },
    )


@dataclass(frozen=True)
class EventUncertainty:
    """Per-event structural uncertainty for TRAKE. Higher means weaker support."""

    event_index: int
    video_coverage: int = 0
    candidate_count: int = 0
    score_margin: float = 0.0
    feasible_candidates: int = 0
    required_expansion: bool = False

    @property
    def uncertainty(self) -> float:
        """Combine coverage scarcity, margin and feasibility. Never a probability.

        A single candidate contributes the MAXIMUM margin term, not zero. For a KIS head
        one candidate means "nothing else is competing", but for a TRAKE event it means
        there is no alternative at all if that one frame is wrong — the thinnest possible
        support, and exactly the event a budget should be spent on.
        """
        scarcity = 1.0 / (1.0 + max(0, int(self.video_coverage)))
        feasibility = 1.0 / (1.0 + max(0, int(self.feasible_candidates)))
        expansion = 0.25 if self.required_expansion else 0.0
        margin = 1.0 if int(self.candidate_count) < 2 else float(self.score_margin)
        return round(
            _clamp(0.4 * scarcity + 0.35 * margin + 0.25 * feasibility + expansion), 6
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "video_coverage": self.video_coverage,
            "candidate_count": self.candidate_count,
            "score_margin": round(self.score_margin, 6),
            "feasible_candidates": self.feasible_candidates,
            "required_expansion": self.required_expansion,
            "uncertainty": self.uncertainty,
        }


def trake_event_uncertainty(
    by_event: dict[int, Sequence[Any]], *, expanded: Sequence[int] = ()
) -> list[EventUncertainty]:
    """Score every event's structural weakness from candidates already retrieved."""
    out: list[EventUncertainty] = []
    for index in sorted(by_event):
        rows = list(by_event[index])
        scores = [float(getattr(row, "score", 0.0)) for row in rows]
        out.append(
            EventUncertainty(
                event_index=int(index),
                video_coverage=len({getattr(row, "video_id", "") for row in rows}),
                candidate_count=len(rows),
                score_margin=margin_uncertainty(scores),
                feasible_candidates=len(rows),
                required_expansion=int(index) in set(expanded),
            )
        )
    return out


# ------------------------------------------------------------------- budget actions


@dataclass(frozen=True)
class BudgetAction:
    """One optional purchase the controller may make."""

    name: str
    target: str = ""
    units: int = 1
    rank: int = 1
    uncertainty: float = 0.0
    expected_gain_proxy: float = 1.0
    cost_proxy: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        if self.cost_proxy:
            return float(self.cost_proxy)
        return ACTION_UNIT_COST.get(self.name, 1.0) * max(1, int(self.units))

    @property
    def priority(self) -> float:
        """utility x uncertainty x gain-proxy / cost.

        `expected_gain_proxy` is NOT expected accuracy. It is a caller-supplied hint
        about how much an action could plausibly matter, and it stays a proxy until it is
        calibrated against held-out ground truth — which does not exist here.
        """
        return round(
            rank_cutoff_utility(max(1, int(self.rank)))
            * max(0.0, float(self.uncertainty))
            * max(0.0, float(self.expected_gain_proxy))
            / max(self.cost, EPSILON),
            9,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "units": self.units,
            "rank": self.rank,
            "rank_cutoff_utility": round(rank_cutoff_utility(max(1, int(self.rank))), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "expected_gain_proxy": round(float(self.expected_gain_proxy), 6),
            "cost": round(self.cost, 6),
            "priority": self.priority,
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass
class BudgetLedger:
    """Spend record for one query. The cap is hard.

    `try_spend` is the only way to buy anything, and it refuses rather than overshooting.
    A refusal is recorded, so "the controller wanted to do more" stays visible instead of
    looking like a decision not to.
    """

    max_cost_units: float
    spent: float = 0.0
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def remaining(self) -> float:
        return max(0.0, float(self.max_cost_units) - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= EPSILON

    def affordable(self, action: BudgetAction) -> bool:
        return action.cost <= self.remaining + EPSILON

    def try_spend(self, action: BudgetAction) -> bool:
        if not self.affordable(action):
            self.rejected.append({**action.to_dict(), "reason": "over_budget"})
            if not self.stop_reason:
                self.stop_reason = "budget_exhausted"
            return False
        self.spent = round(self.spent + action.cost, 6)
        self.accepted.append(action.to_dict())
        return True

    def refuse(self, action: BudgetAction, reason: str) -> None:
        """Record an action the controller chose not to take, and why."""
        self.rejected.append({**action.to_dict(), "reason": reason})

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cost_units": round(float(self.max_cost_units), 6),
            "spent": round(self.spent, 6),
            "remaining": round(self.remaining, 6),
            "actions_accepted": len(self.accepted),
            "actions_rejected": len(self.rejected),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "stop_reason": self.stop_reason or "completed",
        }


def prioritize(actions: Sequence[BudgetAction]) -> list[BudgetAction]:
    """Highest priority first; ties broken deterministically by name and target."""
    return sorted(actions, key=lambda item: (-item.priority, item.name, item.target, item.rank))


def allocate(
    actions: Sequence[BudgetAction], ledger: BudgetLedger
) -> list[BudgetAction]:
    """Buy actions in priority order until the budget cannot afford the next one.

    Deliberately greedy and deliberately transparent: a knapsack solver would spend the
    budget better against a cost model whose weights are guesses, which buys precision
    the inputs do not have.
    """
    taken: list[BudgetAction] = []
    for action in prioritize(actions):
        if ledger.exhausted:
            ledger.refuse(action, "budget_exhausted")
            continue
        if ledger.try_spend(action):
            taken.append(action)
    return taken


def split_budget_by_uncertainty(
    weights: dict[int, float], total_units: int, *, minimum: int = 0, maximum: Optional[int] = None
) -> dict[int, int]:
    """Divide an integer budget across keys in proportion to weight.

    Used for TRAKE: the weakest event should get the most, but no single event may take
    everything, and the parts must sum to exactly `total_units` so the cap is real.
    """
    keys = sorted(weights)
    total = max(0, int(total_units))
    if not keys or total == 0:
        return {key: 0 for key in keys}
    ceiling = total if maximum is None else max(0, int(maximum))
    # The per-key cap can make the requested total unreachable. Clamping here keeps the
    # postcondition honest: the parts sum to what could actually be allocated, and never
    # to a number the caps forbid.
    total = min(total, ceiling * len(keys))
    floor = max(0, int(minimum))
    if floor * len(keys) > total:
        floor = total // len(keys)

    out = {key: floor for key in keys}
    remaining = total - floor * len(keys)
    mass = sum(max(0.0, float(weights[key])) for key in keys)
    def hand_out(leftover: int, order: Sequence[int]) -> int:
        """Give out one unit at a time in `order`, skipping keys at their ceiling.

        Loops until a whole pass grants nothing, so a binding cap on one key cannot
        strand budget that another key could still take.
        """
        while leftover > 0:
            granted = 0
            for key in order:
                if leftover <= 0:
                    break
                if out[key] < ceiling:
                    out[key] += 1
                    leftover -= 1
                    granted += 1
            if granted == 0:
                break
        return leftover

    if mass <= EPSILON:
        # No signal: spread what is left as evenly as possible, deterministically.
        hand_out(remaining, keys)
        return out

    shares = {key: remaining * max(0.0, float(weights[key])) / mass for key in keys}
    for key in keys:
        grant = min(int(shares[key]), ceiling - out[key])
        out[key] += max(0, grant)
    # Remainder goes to the largest fractional parts first, then wherever it fits.
    order = sorted(keys, key=lambda key: (-(shares[key] % 1.0), -weights[key], key))
    hand_out(total - sum(out.values()), order)
    return out


# ------------------------------------------------------------------ channel policy

CHANNEL_DEPTH_FULL = "full"
CHANNEL_DEPTH_SHALLOW = "shallow"
CHANNEL_DEPTH_SKIP = "skip"

# What a shallow channel costs relative to a full one. A skipped channel is not queried.
CHANNEL_DEPTH_SCALE = {
    CHANNEL_DEPTH_FULL: 1.0,
    CHANNEL_DEPTH_SHALLOW: 0.25,
    CHANNEL_DEPTH_SKIP: 0.0,
}


def channel_policy(representation: Any, *, enabled_channels: Sequence[str]) -> dict[str, str]:
    """Query-conditioned depth per channel. EXPERIMENTAL.

    The signals are the ones the query representation already carries: whether the query
    produced object terms, whether it contains quoted or numeric text that a lexical
    channel could match, whether it has metadata-shaped terms. A channel with nothing to
    match on is asked shallowly rather than at full depth.

    Two rules are absolute:

    * **CLIP is never reduced.** It is the only channel that can retrieve on appearance
      alone, and the whole system's recall floor rests on it.
    * **This is a cost experiment, not a quality claim.** Skipping a channel is not
      asserted to help; the question it exists to answer — can equivalent quality be had
      for less work — cannot be answered until ground truth exists.
    """
    object_terms = tuple(getattr(representation, "object_terms", ()) or ())
    tokens = tuple(getattr(representation, "tokens_folded", ()) or ())
    numbers = tuple(getattr(representation, "number_terms", ()) or ())
    lexical = tuple(getattr(representation, "lexical_terms", ()) or ())

    policy: dict[str, str] = {}
    for channel in enabled_channels:
        if channel == "clip":
            policy[channel] = CHANNEL_DEPTH_FULL
        elif channel == "objects":
            policy[channel] = (
                CHANNEL_DEPTH_FULL if object_terms else CHANNEL_DEPTH_SHALLOW
            )
        elif channel == "bm25":
            policy[channel] = (
                CHANNEL_DEPTH_FULL if (numbers or len(tokens) >= 3) else CHANNEL_DEPTH_SHALLOW
            )
        elif channel == "metadata":
            policy[channel] = CHANNEL_DEPTH_FULL if lexical else CHANNEL_DEPTH_SHALLOW
        else:
            policy[channel] = CHANNEL_DEPTH_SHALLOW
    return policy


def apply_channel_policy(depths: dict[str, int], policy: dict[str, str]) -> dict[str, int]:
    """Scale configured depths by the policy. A channel is never deepened beyond config."""
    out = dict(depths)
    for channel, level in policy.items():
        if channel not in out:
            continue
        scale = CHANNEL_DEPTH_SCALE.get(level, 1.0)
        out[channel] = max(0, int(out[channel] * scale)) if scale < 1.0 else out[channel]
    return out


__all__ = [
    "ACTIONS",
    "CHANNEL_DEPTH_FULL",
    "CHANNEL_DEPTH_SCALE",
    "CHANNEL_DEPTH_SHALLOW",
    "CHANNEL_DEPTH_SKIP",
    "apply_channel_policy",
    "channel_policy",
    "ACTION_DEEPEN_CHANNEL",
    "ACTION_DENSE_TEMPORAL_ZOOM",
    "ACTION_OFFICIAL_GRID_REFINE",
    "ACTION_QA_VLM_CALL",
    "ACTION_SPARSE_VIDEO_SAMPLE",
    "ACTION_UNIT_COST",
    "BudgetAction",
    "BudgetLedger",
    "EventUncertainty",
    "UncertaintySignals",
    "allocate",
    "channel_disagreement",
    "kis_uncertainty",
    "margin_uncertainty",
    "prioritize",
    "split_budget_by_uncertainty",
    "support_concentration",
    "temporal_ambiguity",
    "trake_event_uncertainty",
]

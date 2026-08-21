"""Grounded per-video-hypothesis Q&A: evidence, backends, and answer normalization.

An AIC Q&A prediction is a triple `(video_id, frame_id, answer)`, and the answer must be
grounded in **that video's** visual evidence. Before Phase 6 the engine answered once for
the globally top-ranked candidate and attached that single answer to every prediction
row, so a row for video B routinely carried an answer derived from video A's frames.
This module supplies the pieces that make one answer per video hypothesis possible:

* `QAVideoHypothesis` / `QAEvidenceBundle` / `QAEvidenceFrame` -- evidence that is
  structurally incapable of crossing videos, because a bundle is built for one video and
  carries its own `video_id`;
* `VisualQAAnswerer` -- the backend contract, with honest capability reporting;
* answer-type canonicalization and Vietnamese/English normalization;
* `answer_reliability_score` -- a transparent heuristic, never a calibrated probability.

No part of this module claims accuracy. There is no AIC ground truth in this repository.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

from ingestion.schemas import KeyframeRecord
from retrieval.vqa_module import MockVqaAnswerer, VqaAnswer, VqaAnswerer

from .query_normalization import fold_accents, normalize_query

# --------------------------------------------------------------------- answer types

ANSWER_TYPE_AUTO = "auto"
ANSWER_TYPE_NUMBER = "number"
ANSWER_TYPE_BOOLEAN = "boolean"
ANSWER_TYPE_COLOR = "color"
ANSWER_TYPE_SHORT_TEXT = "short_text"
ANSWER_TYPES = (
    ANSWER_TYPE_AUTO,
    ANSWER_TYPE_NUMBER,
    ANSWER_TYPE_BOOLEAN,
    ANSWER_TYPE_COLOR,
    ANSWER_TYPE_SHORT_TEXT,
)
# The UI has always sent these spellings; they are accepted rather than rejected.
_ANSWER_TYPE_ALIASES = {
    "": ANSWER_TYPE_AUTO,
    "auto": ANSWER_TYPE_AUTO,
    "number": ANSWER_TYPE_NUMBER,
    "count": ANSWER_TYPE_NUMBER,
    "int": ANSWER_TYPE_NUMBER,
    "boolean": ANSWER_TYPE_BOOLEAN,
    "bool": ANSWER_TYPE_BOOLEAN,
    "yes/no": ANSWER_TYPE_BOOLEAN,
    "yesno": ANSWER_TYPE_BOOLEAN,
    "color": ANSWER_TYPE_COLOR,
    "colour": ANSWER_TYPE_COLOR,
    "text": ANSWER_TYPE_SHORT_TEXT,
    "short_text": ANSWER_TYPE_SHORT_TEXT,
}

ANSWER_STATUS_ANSWERED = "answered"
ANSWER_STATUS_ABSTAINED = "abstained"
ANSWER_STATUS_BACKEND_FAILED = "backend_failed"
ANSWER_STATUS_VISUAL_UNAVAILABLE = "visual_unavailable"
# The per-query VLM budget ran out before this hypothesis was reached. It is NOT an
# answer and never becomes one: a budget is a spending limit, not a reason to guess.
ANSWER_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"

BACKEND_STATE_READY = "ready"
BACKEND_STATE_NOT_LOADED = "not_loaded"
BACKEND_STATE_NOT_AVAILABLE = "not_available"

UNKNOWN_ANSWER = "unknown"
_UNKNOWN_FORMS = {
    "", "unknown", "none", "n/a", "khong xac dinh", "không xác định",
    "khong co mo ta", "không có mô tả", "khong co du lieu", "không có dữ liệu",
    "khong tim thay canh lien quan",
}

# `khong` is deliberately absent: it means BOTH "no" and "zero" in Vietnamese, so it is
# only resolved when `expected_answer_type` says which one is being asked for.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "mot": "1", "một": "1", "hai": "2", "ba": "3", "bon": "4", "bốn": "4",
    "nam": "5", "năm": "5", "sau": "6", "sáu": "6", "bay": "7", "bảy": "7",
    "tam": "8", "tám": "8", "chin": "9", "chín": "9", "muoi": "10", "mười": "10",
}
_ZERO_WORDS = {"khong", "không", "zero", "no one", "nobody", "khong ai", "không ai"}
_BOOLEAN_TRUE = {"yes", "yeah", "yep", "true", "co", "có", "dung", "đúng", "phai", "phải", "1"}
_BOOLEAN_FALSE = {"no", "nope", "false", "khong", "không", "sai", "khong phai", "không phải", "0"}
_COLOR_WORDS = {
    "red": "red", "do": "red", "đỏ": "red",
    "blue": "blue", "xanh duong": "blue", "xanh dương": "blue", "xanh nuoc bien": "blue",
    "green": "green", "xanh la": "green", "xanh lá": "green", "xanh luc": "green",
    "yellow": "yellow", "vang": "yellow", "vàng": "yellow",
    "white": "white", "trang": "white", "trắng": "white",
    "black": "black", "den": "black", "đen": "black",
    "grey": "grey", "gray": "grey", "xam": "grey", "xám": "grey",
    "orange": "orange", "cam": "orange",
    "brown": "brown", "nau": "brown", "nâu": "brown",
    "pink": "pink", "hong": "pink", "hồng": "pink",
    "purple": "purple", "tim": "purple", "tím": "purple",
    "silver": "silver", "bac": "silver", "bạc": "silver",
}
# Kept for the `auto` path only: the historical whole-string aliases, minus the
# ambiguous bare "khong"/"không".
_AUTO_ALIASES = {
    **{word: "yes" for word in _BOOLEAN_TRUE - {"1"}},
    **{word: "no" for word in (_BOOLEAN_FALSE - {"khong", "không", "0"})},
    **_COLOR_WORDS,
}


def canonical_answer_type(value: Any) -> str:
    """Map a request/config answer type onto one canonical form."""
    if value is None:
        return ANSWER_TYPE_AUTO
    key = str(value).strip().casefold()
    if key in _ANSWER_TYPE_ALIASES:
        return _ANSWER_TYPE_ALIASES[key]
    if key in ANSWER_TYPES:
        return key
    raise ValueError(
        f"Unsupported expected_answer_type {value!r}; supported: {', '.join(ANSWER_TYPES)}"
    )


def _clean(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def is_unknown_answer(value: str) -> bool:
    return _clean(value) in _UNKNOWN_FORMS


def normalize_answer(
    value: str,
    aliases: dict[str, str] | None = None,
    *,
    expected_type: Any = None,
) -> str:
    """Canonicalize an answer, using the expected type to resolve ambiguity.

    Vietnamese `không` is the reason this takes a type: it means "no" for a boolean
    question and "zero" for a counting question. With no declared type the word is left
    alone rather than guessed at, because guessing wrong silently changes the submitted
    answer.
    """
    kind = canonical_answer_type(expected_type)
    text = _clean(value)
    mapping = dict(aliases or {})
    if text in mapping:
        return mapping[text]
    if text in _UNKNOWN_FORMS:
        # "I don't know" must stay "I don't know". Vietnamese unknown phrases such as
        # "khong co mo ta" begin with "khong", so type normalization would otherwise
        # turn a refusal into a confident "no" (boolean) or "0" (number).
        return text

    if kind == ANSWER_TYPE_NUMBER:
        return _normalize_number(text)
    if kind == ANSWER_TYPE_BOOLEAN:
        return _normalize_boolean(text)
    if kind == ANSWER_TYPE_COLOR:
        return _normalize_color(text)
    if kind == ANSWER_TYPE_SHORT_TEXT:
        # Minimal on purpose: a short noun phrase must survive intact.
        return text
    # auto: historical behaviour, minus the ambiguous bare "khong".
    if text in _AUTO_ALIASES:
        return _AUTO_ALIASES[text]
    return " ".join(_NUMBER_WORDS.get(word, word) for word in text.split())


def _normalize_number(text: str) -> str:
    if not text:
        return ""
    digits = re.search(r"-?\d+", text)
    if digits:
        return digits.group()
    if text in _ZERO_WORDS:
        return "0"
    for word in text.split():
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
        if word in _ZERO_WORDS:
            return "0"
    return text


def _normalize_boolean(text: str) -> str:
    if not text:
        return ""
    if text in _BOOLEAN_TRUE:
        return "yes"
    if text in _BOOLEAN_FALSE:
        return "no"
    words = text.split()
    for word in words:
        if word in _BOOLEAN_TRUE:
            return "yes"
        if word in _BOOLEAN_FALSE:
            return "no"
    return text


def _normalize_color(text: str) -> str:
    if not text:
        return ""
    if text in _COLOR_WORDS:
        return _COLOR_WORDS[text]
    # Longest phrase first so "xanh duong" beats "xanh".
    for phrase in sorted(_COLOR_WORDS, key=len, reverse=True):
        if " " in phrase and phrase in text:
            return _COLOR_WORDS[phrase]
    for word in text.split():
        if word in _COLOR_WORDS:
            return _COLOR_WORDS[word]
    # A colour was asked for and none was said: do NOT invent one.
    return text


def answer_matches_type(answer: str, expected_type: Any) -> bool:
    """Does a normalized answer actually look like the requested type?"""
    kind = canonical_answer_type(expected_type)
    text = _clean(answer)
    if not text or text in _UNKNOWN_FORMS:
        return False
    if kind == ANSWER_TYPE_NUMBER:
        return bool(re.fullmatch(r"-?\d+", text))
    if kind == ANSWER_TYPE_BOOLEAN:
        return text in {"yes", "no"}
    if kind == ANSWER_TYPE_COLOR:
        return text in set(_COLOR_WORDS.values())
    return True


# ------------------------------------------------------------------------ data models


@dataclass(frozen=True)
class QAInput:
    event_description: str
    question: str
    expected_answer_type: str | None = None


@dataclass(frozen=True)
class QAFrameHypothesis:
    """One candidate frame of one video, before evidence is loaded."""

    keyframe_id: str
    video_id: str
    frame_idx: Optional[int]
    timestamp: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QAEvidenceFrame:
    """One piece of visual evidence, always tagged with the video it came from."""

    video_id: str
    frame_idx: Optional[int]
    timestamp: float
    source: str
    keyframe_id: Optional[str] = None
    role: str = "context"
    retrieval_score: float = 0.0
    visual_score: Optional[float] = None
    text: str = ""
    objects: tuple[str, ...] = ()
    # Whether a visual source EXISTS for this frame (a cheap check, no decoding).
    image_available: bool = False
    # Pixels, loaded only when the backend can actually look at them; never serialized.
    image_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)

    @property
    def has_image(self) -> bool:
        return bool(self.image_bytes)

    @property
    def evidence_id(self) -> str:
        if self.keyframe_id:
            return self.keyframe_id
        return f"{self.video_id}/decoded_{int(self.frame_idx or 0):08d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "evidence_id": self.evidence_id,
            "keyframe_id": self.keyframe_id,
            "frame_idx": self.frame_idx,
            "timestamp": round(float(self.timestamp), 3),
            "source": self.source,
            "role": self.role,
            "retrieval_score": round(float(self.retrieval_score), 6),
            "visual_score": (
                None if self.visual_score is None else round(float(self.visual_score), 6)
            ),
            "image_available": self.image_available,
            "image_loaded": self.has_image,
        }


@dataclass(frozen=True)
class QAEvidenceBundle:
    """Everything a backend may look at for ONE video hypothesis.

    A bundle carries its own `video_id`, and every frame in it carries the same one, so
    a backend cannot be handed evidence from two videos by accident.
    """

    video_id: str
    question: str
    expected_answer_type: str = ANSWER_TYPE_AUTO
    frames: tuple[QAEvidenceFrame, ...] = ()

    def __post_init__(self) -> None:
        stray = sorted({frame.video_id for frame in self.frames} - {self.video_id})
        if stray:
            raise ValueError(
                f"Evidence bundle for {self.video_id!r} contains frames from {stray}; "
                "Q&A evidence must never cross videos."
            )

    @property
    def visual_frames(self) -> tuple[QAEvidenceFrame, ...]:
        """Frames whose pixels are actually in hand for a backend call."""
        return tuple(frame for frame in self.frames if frame.has_image)

    @property
    def visual_available(self) -> bool:
        """Whether a visual source EXISTS, independently of whether it was loaded.

        Kept separate from `visual_frames` so a non-visual backend does not make the
        dataset look like it has no pixels.
        """
        return any(frame.image_available for frame in self.frames)

    @property
    def primary(self) -> Optional[QAEvidenceFrame]:
        return next((frame for frame in self.frames if frame.role == "primary"), None)

    def images(self) -> dict[str, bytes]:
        return {
            frame.evidence_id: frame.image_bytes
            for frame in self.frames
            if frame.image_bytes
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "expected_answer_type": self.expected_answer_type,
            "visual_available": self.visual_available,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True)
class QAAnswererStatus:
    """Honest capability report. `visual_capable` is the field that must not lie."""

    backend_type: str
    state: str = BACKEND_STATE_NOT_LOADED
    visual_capable: bool = False
    supports_multi_image: bool = False
    production_ready: bool = False
    model_name: Optional[str] = None
    device: Optional[str] = None
    warning: Optional[str] = None
    fallback_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.state == BACKEND_STATE_READY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QAAnswerResult:
    """One backend's answer for ONE video, with its own provenance."""

    video_id: str
    answer: str
    normalized_answer: str
    status: str = ANSWER_STATUS_ANSWERED
    backend_type: str = "unknown"
    visual: bool = False
    value: Optional[int] = None
    reasoning: str = ""
    used_evidence_ids: tuple[str, ...] = ()
    warning: Optional[str] = None

    @property
    def answered(self) -> bool:
        return self.status == ANSWER_STATUS_ANSWERED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["used_evidence_ids"] = list(self.used_evidence_ids)
        return data


@dataclass(frozen=True)
class QAVideoHypothesis:
    """One video considered as the answer's source, with its own frames."""

    video_id: str
    rank: int
    retrieval_score: float
    best_candidate_score: float
    support_count: int
    frames: tuple[QAFrameHypothesis, ...] = ()
    refinement: Optional[dict[str, Any]] = None

    @property
    def submission_frame(self) -> Optional[QAFrameHypothesis]:
        return self.frames[0] if self.frames else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "rank": self.rank,
            "retrieval_score": round(float(self.retrieval_score), 6),
            "best_candidate_score": round(float(self.best_candidate_score), 6),
            "support_count": self.support_count,
            "frames": [frame.to_dict() for frame in self.frames],
            "refinement": self.refinement,
        }


class QABackendUnavailable(RuntimeError):
    """Raised when a backend cannot answer at all. Never becomes a fabricated answer."""


@runtime_checkable
class VisualQAAnswerer(Protocol):
    """The Phase 6 backend contract.

    `answer` receives evidence for exactly one video and must return an answer for that
    video. Raising `QABackendUnavailable` (or anything else) is caught by the engine and
    turned into a `backend_failed` result for that hypothesis alone.
    """

    def answer(
        self,
        question: str,
        evidence: QAEvidenceBundle,
        *,
        expected_answer_type: str | None = None,
    ) -> QAAnswerResult: ...

    def status(self) -> QAAnswererStatus: ...


# ------------------------------------------------------------------ evidence policy


def _score_of(frame: QAFrameHypothesis | QAEvidenceFrame) -> float:
    return float(getattr(frame, "score", None) or getattr(frame, "retrieval_score", 0.0))


def select_evidence_frames(
    frames: Sequence[QAEvidenceFrame],
    *,
    count: int,
    diversity_s: float = 1.5,
) -> tuple[QAEvidenceFrame, ...]:
    """Pick a bounded, temporally diverse evidence set, strongest first.

    The pre-Phase-6 selector seeded its choice with the first and last frames of the
    window, so asking for one or two frames returned window boundaries rather than the
    evidence the retriever actually liked. This one is explicit:

    1. the strongest frame (role `primary`);
    2. the strongest remaining frame at least `diversity_s` away from it;
    3. the strongest remaining frame on the OTHER side, so three frames give before /
       event / after context;
    4. any further slots by score, still respecting the diversity gap, and only then
       relaxing it rather than returning fewer frames than asked for.

    The result is returned in timestamp order so a backend reads it chronologically,
    with roles recording what each frame was chosen for.
    """
    pool = list(frames)
    limit = max(1, int(count))
    if not pool:
        return ()

    def rank_key(frame: QAEvidenceFrame) -> tuple:
        return (-_score_of(frame), float(frame.timestamp), frame.evidence_id)

    ordered = sorted(pool, key=rank_key)
    primary = ordered[0]
    chosen: list[QAEvidenceFrame] = [primary]

    def far_enough(frame: QAEvidenceFrame) -> bool:
        return all(
            abs(float(frame.timestamp) - float(picked.timestamp)) >= float(diversity_s)
            for picked in chosen
        )

    if limit > 1:
        second = next((f for f in ordered[1:] if far_enough(f)), None)
        if second is not None:
            chosen.append(second)

    if limit > 2 and len(chosen) == 2:
        # Opposite side of the second pick, so three frames straddle the event.
        wanted_after = float(chosen[1].timestamp) < float(primary.timestamp)
        third = next(
            (
                f
                for f in ordered[1:]
                if f not in chosen
                and far_enough(f)
                and ((float(f.timestamp) > float(primary.timestamp)) == wanted_after)
            ),
            None,
        )
        if third is not None:
            chosen.append(third)

    for frame in ordered:
        if len(chosen) >= limit:
            break
        if frame not in chosen and far_enough(frame):
            chosen.append(frame)
    for frame in ordered:  # relax diversity rather than under-fill the budget
        if len(chosen) >= limit:
            break
        if frame not in chosen:
            chosen.append(frame)

    center = float(primary.timestamp)
    selected = sorted(chosen[:limit], key=lambda f: (float(f.timestamp), f.evidence_id))
    out: list[QAEvidenceFrame] = []
    for frame in selected:
        if frame is primary:
            role = "primary"
        elif float(frame.timestamp) < center:
            role = "before"
        elif float(frame.timestamp) > center:
            role = "after"
        else:
            role = "context"
        out.append(_with_role(frame, role))
    return tuple(out)


def _with_role(frame: QAEvidenceFrame, role: str) -> QAEvidenceFrame:
    if frame.role == role:
        return frame
    return QAEvidenceFrame(
        video_id=frame.video_id,
        frame_idx=frame.frame_idx,
        timestamp=frame.timestamp,
        source=frame.source,
        keyframe_id=frame.keyframe_id,
        role=role,
        retrieval_score=frame.retrieval_score,
        visual_score=frame.visual_score,
        text=frame.text,
        objects=frame.objects,
        image_available=frame.image_available,
        image_bytes=frame.image_bytes,
    )


def select_temporally_diverse(
    frames: Sequence[QAFrameHypothesis], *, count: int, diversity_s: float
) -> tuple[QAFrameHypothesis, ...]:
    """Top-scoring frame hypotheses of one video, spread out in time.

    Five adjacent BTC keyframes of the same second describe one moment; taking all of
    them would spend the whole per-video budget on a single instant.
    """
    limit = max(1, int(count))
    ordered = sorted(
        frames, key=lambda f: (-float(f.score), float(f.timestamp), f.keyframe_id)
    )
    chosen: list[QAFrameHypothesis] = []
    for frame in ordered:
        if len(chosen) >= limit:
            break
        if all(
            abs(float(frame.timestamp) - float(picked.timestamp)) >= float(diversity_s)
            for picked in chosen
        ):
            chosen.append(frame)
    for frame in ordered:  # never return fewer than asked for when frames exist
        if len(chosen) >= limit:
            break
        if frame not in chosen:
            chosen.append(frame)
    return tuple(chosen[:limit])


def group_hypotheses_by_video(
    candidates: Iterable[Any],
    *,
    top_video_hypotheses: int,
    frame_hypotheses_per_video: int,
    diversity_s: float,
    support_bonus: float,
    frame_idx_of=None,
) -> tuple[QAVideoHypothesis, ...]:
    """Turn a flat candidate pool into ranked, per-video hypotheses.

    A video's score is its best candidate plus a small bounded bonus for having more
    supporting candidates -- deliberately simple and deterministic, not a learned ranker.
    """
    grouped: dict[str, list[QAFrameHypothesis]] = {}
    for candidate in candidates:
        video_id = str(candidate.video_id)
        frame_idx = None
        if frame_idx_of is not None:
            try:
                frame_idx = frame_idx_of(candidate)
            except Exception:  # noqa: BLE001 - an unmapped candidate is simply unusable
                frame_idx = None
        grouped.setdefault(video_id, []).append(
            QAFrameHypothesis(
                keyframe_id=str(candidate.keyframe_id),
                video_id=video_id,
                frame_idx=None if frame_idx is None else int(frame_idx),
                timestamp=float(candidate.timestamp),
                score=float(candidate.score),
            )
        )

    scored: list[tuple[float, float, int, str, list[QAFrameHypothesis]]] = []
    for video_id, frames in grouped.items():
        best = max(float(frame.score) for frame in frames)
        support = max(0, len(frames) - 1)
        # Bounded and relative, so it can nudge ordering but never invert it.
        video_score = best * (1.0 + float(support_bonus) * min(support, 4) / 4.0)
        scored.append((-video_score, -best, -support, video_id, frames))
    scored.sort()

    out: list[QAVideoHypothesis] = []
    for rank, (negative_score, negative_best, negative_support, video_id, frames) in enumerate(
        scored[: max(1, int(top_video_hypotheses))], start=1
    ):
        out.append(
            QAVideoHypothesis(
                video_id=video_id,
                rank=rank,
                retrieval_score=-negative_score,
                best_candidate_score=-negative_best,
                support_count=-negative_support,
                frames=select_temporally_diverse(
                    frames, count=frame_hypotheses_per_video, diversity_s=diversity_s
                ),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------- reliability


def answer_reliability_score(
    *,
    backend: QAAnswererStatus,
    evidence_count: int,
    visual_evidence_count: int,
    answer: str,
    expected_answer_type: Any = None,
    retrieval_margin: float = 0.0,
) -> float:
    """A transparent additive heuristic in [0, 1]. NOT a calibrated probability.

    Deliberately named `answer_reliability_score` rather than a confidence or
    probability, because nothing here was fitted to data:

        0.10  base
      + 0.25  the backend actually looked at images (visual_capable AND images present)
      + 0.10  the backend is production-ready
      + 0.05  per evidence frame, capped at 4 frames (max 0.20)
      + 0.15  the answer is non-empty and not an "unknown" form
      + 0.10  the answer validates against the requested answer type
      + 0.10  scaled by the coarse retrieval margin of the winning video

    Every term is observable in the diagnostics, so a reader can reconstruct the number.
    """
    score = 0.10
    if backend.visual_capable and visual_evidence_count > 0:
        score += 0.25
    if backend.production_ready:
        score += 0.10
    score += 0.05 * min(max(0, int(evidence_count)), 4)
    known = bool(str(answer).strip()) and not is_unknown_answer(answer)
    if known:
        score += 0.15
    if known and canonical_answer_type(expected_answer_type) != ANSWER_TYPE_AUTO:
        if answer_matches_type(answer, expected_answer_type):
            score += 0.10
    score += 0.10 * max(0.0, min(1.0, float(retrieval_margin)))
    return round(max(0.0, min(1.0, score)), 6)


# ------------------------------------------------------------------------- backends


class MockTextQAAnswerer:
    """Offline text-only fallback. It does NOT look at images, and says so.

    Wraps the historical `MockVqaAnswerer`, which reasons over captions, OCR, ASR, and
    object labels. Useful for tests and for keeping the pipeline runnable without a
    model, but its output must never be presented as visual Q&A.
    """

    backend_type = "mock"

    def __init__(self, answerer: VqaAnswerer | None = None):
        self._answerer = answerer or MockVqaAnswerer()

    def answer(
        self,
        question: str,
        evidence: QAEvidenceBundle,
        *,
        expected_answer_type: str | None = None,
    ) -> QAAnswerResult:
        records = [_as_record(frame) for frame in evidence.frames]
        raw: VqaAnswer = self._answerer.answer(question, records, evidence.images())
        text = str(raw.value) if raw.value is not None else str(raw.answer or "")
        kind = canonical_answer_type(expected_answer_type or evidence.expected_answer_type)
        return QAAnswerResult(
            video_id=evidence.video_id,
            answer=text,
            normalized_answer=normalize_answer(text, expected_type=kind),
            status=ANSWER_STATUS_ANSWERED,
            backend_type=self.backend_type,
            visual=False,
            value=raw.value,
            reasoning=str(raw.reasoning or ""),
            used_evidence_ids=tuple(frame.evidence_id for frame in evidence.frames),
            warning=(
                "Answered by the non-visual mock backend from caption/OCR/ASR text; "
                "this is not visual Q&A."
            ),
        )

    def status(self) -> QAAnswererStatus:
        return QAAnswererStatus(
            backend_type=self.backend_type,
            state=BACKEND_STATE_READY,
            visual_capable=False,
            supports_multi_image=False,
            production_ready=False,
            model_name="mock-text-heuristics",
            device="cpu",
            warning="Non-visual fallback: reasons over text signals, never over pixels.",
        )


class LocalVlmQAAnswerer:
    """Adapter for a locally hosted multimodal model.

    The contract is real and fully exercised by tests through an injected `model`, which
    must expose ``generate(images: list[bytes], prompt: str) -> str``. What this class
    deliberately does NOT do is fetch a model: no multi-gigabyte download is ever
    triggered, so with nothing injected and nothing configured it reports
    `not_available` and refuses to answer rather than pretending.
    """

    backend_type = "local_vlm"

    def __init__(
        self,
        model: Any = None,
        *,
        model_name: str = "",
        device: str = "auto",
        supports_multi_image: bool = True,
        max_answer_tokens: int = 64,
        temperature: float = 0.0,
    ):
        self._model = model
        self.model_name = model_name or (type(model).__name__ if model is not None else "")
        self.device = device
        self.supports_multi_image = bool(supports_multi_image)
        self.max_answer_tokens = int(max_answer_tokens)
        self.temperature = float(temperature)

    @property
    def available(self) -> bool:
        return self._model is not None

    def answer(
        self,
        question: str,
        evidence: QAEvidenceBundle,
        *,
        expected_answer_type: str | None = None,
    ) -> QAAnswerResult:
        if self._model is None:
            raise QABackendUnavailable(
                "No local VLM is configured. A local multimodal model must be provided "
                "explicitly; weights are never downloaded automatically."
            )
        kind = canonical_answer_type(expected_answer_type or evidence.expected_answer_type)
        frames = evidence.visual_frames
        if not frames:
            raise QABackendUnavailable(
                f"No visual evidence is available for video {evidence.video_id!r}."
            )
        if not self.supports_multi_image:
            # One-image model: use the strongest frame explicitly rather than silently
            # dropping the rest.
            primary = evidence.primary
            frames = (primary,) if primary is not None and primary.has_image else frames[:1]
        prompt = build_answer_prompt(question, kind)
        text = str(
            self._model.generate([frame.image_bytes for frame in frames], prompt) or ""
        ).strip()
        return QAAnswerResult(
            video_id=evidence.video_id,
            answer=text,
            normalized_answer=normalize_answer(text, expected_type=kind),
            status=ANSWER_STATUS_ANSWERED,
            backend_type=self.backend_type,
            visual=True,
            reasoning="Local multimodal model over this video's evidence frames.",
            used_evidence_ids=tuple(frame.evidence_id for frame in frames),
        )

    def status(self) -> QAAnswererStatus:
        if self._model is None:
            return QAAnswererStatus(
                backend_type=self.backend_type,
                state=BACKEND_STATE_NOT_AVAILABLE,
                visual_capable=True,
                supports_multi_image=self.supports_multi_image,
                production_ready=False,
                model_name=self.model_name or None,
                device=self.device,
                fallback_reason=(
                    "No local VLM is configured and weights are never downloaded "
                    "automatically."
                ),
                warning="Local visual Q&A is unavailable.",
            )
        return QAAnswererStatus(
            backend_type=self.backend_type,
            state=BACKEND_STATE_READY,
            visual_capable=True,
            supports_multi_image=self.supports_multi_image,
            production_ready=True,
            model_name=self.model_name or None,
            device=self.device,
        )


class ApiVqaAnswerer:
    """Optional hosted vision backend. Lazy, never required, never logs a key.

    Construction performs no import and no network call; the client is created on the
    first answer. A missing SDK or key is reported through `status()` and raised as
    `QABackendUnavailable`, which the engine confines to a single hypothesis.
    """

    backend_type = "api"

    def __init__(
        self,
        *,
        model_name: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_images: int = 8,
        max_answer_tokens: int = 64,
        client: Any = None,
    ):
        self.model_name = model_name or "claude-opus-4-8"
        self.api_key_env = api_key_env
        self.max_images = max(1, int(max_images))
        self.max_answer_tokens = max(1, int(max_answer_tokens))
        self._client = client
        self._answerer: Any = None
        self._failure: Optional[str] = None

    def _key_present(self) -> bool:
        import os

        return bool(os.environ.get(self.api_key_env))

    def _load(self):
        if self._answerer is not None:
            return self._answerer
        from retrieval.vqa_module import ClaudeVqaAnswerer

        try:
            self._answerer = ClaudeVqaAnswerer(
                model=self.model_name,
                api_key_env=self.api_key_env,
                max_frames=self.max_images,
                max_tokens=self.max_answer_tokens,
                client=self._client,
            )
        except Exception as exc:  # noqa: BLE001 - the reason, never the key, is kept
            self._failure = f"{type(exc).__name__}: {exc}"
            raise QABackendUnavailable(
                f"Hosted visual Q&A backend is unavailable: {self._failure}"
            ) from exc
        return self._answerer

    def answer(
        self,
        question: str,
        evidence: QAEvidenceBundle,
        *,
        expected_answer_type: str | None = None,
    ) -> QAAnswerResult:
        kind = canonical_answer_type(expected_answer_type or evidence.expected_answer_type)
        frames = evidence.visual_frames[: self.max_images]
        if not frames:
            raise QABackendUnavailable(
                f"No visual evidence is available for video {evidence.video_id!r}."
            )
        backend = self._load()
        records = [_as_record(frame) for frame in frames]
        images = {frame.evidence_id: frame.image_bytes for frame in frames}
        try:
            raw = backend.answer(build_answer_prompt(question, kind), records, images)
        except Exception as exc:  # noqa: BLE001 - one hypothesis, not the whole search
            raise QABackendUnavailable(
                f"Hosted visual Q&A call failed: {type(exc).__name__}: {exc}"
            ) from exc
        text = str(raw.value) if raw.value is not None else str(raw.answer or "")
        return QAAnswerResult(
            video_id=evidence.video_id,
            answer=text,
            normalized_answer=normalize_answer(text, expected_type=kind),
            status=ANSWER_STATUS_ANSWERED,
            backend_type=self.backend_type,
            visual=True,
            value=raw.value,
            reasoning=str(raw.reasoning or ""),
            used_evidence_ids=tuple(frame.evidence_id for frame in frames),
        )

    def status(self) -> QAAnswererStatus:
        if self._client is not None or self._answerer is not None:
            return QAAnswererStatus(
                backend_type=self.backend_type,
                state=BACKEND_STATE_READY,
                visual_capable=True,
                supports_multi_image=True,
                production_ready=True,
                model_name=self.model_name,
                device="remote",
            )
        if self._failure is not None:
            return QAAnswererStatus(
                backend_type=self.backend_type,
                state=BACKEND_STATE_NOT_AVAILABLE,
                visual_capable=True,
                supports_multi_image=True,
                model_name=self.model_name,
                device="remote",
                fallback_reason=self._failure,
            )
        if not self._key_present():
            return QAAnswererStatus(
                backend_type=self.backend_type,
                state=BACKEND_STATE_NOT_AVAILABLE,
                visual_capable=True,
                supports_multi_image=True,
                model_name=self.model_name,
                device="remote",
                # The variable NAME only; the value is never read into a report.
                fallback_reason=f"{self.api_key_env} is not set.",
            )
        return QAAnswererStatus(
            backend_type=self.backend_type,
            state=BACKEND_STATE_NOT_LOADED,
            visual_capable=True,
            supports_multi_image=True,
            production_ready=True,
            model_name=self.model_name,
            device="remote",
            warning="Hosted backend has not been contacted yet.",
        )


ANSWER_PROMPTS = {
    ANSWER_TYPE_NUMBER: "Answer with a single integer and nothing else.",
    ANSWER_TYPE_BOOLEAN: "Answer with exactly one word: yes or no.",
    ANSWER_TYPE_COLOR: "Answer with a single colour name and nothing else.",
    ANSWER_TYPE_SHORT_TEXT: "Answer with a concise noun phrase, at most a few words.",
    ANSWER_TYPE_AUTO: "Answer as briefly as possible, with no explanation.",
}


def build_answer_prompt(question: str, expected_answer_type: Any = None) -> str:
    """A submission-shaped prompt: the answer column wants a value, not an essay."""
    kind = canonical_answer_type(expected_answer_type)
    return (
        f"{str(question).strip()}\n"
        f"{ANSWER_PROMPTS[kind]} Base the answer only on the supplied frames of this "
        "one video."
    )


def build_qa_answerer(
    backend_type: str = "auto",
    *,
    model_name: str = "",
    device: str = "auto",
    max_answer_tokens: int = 64,
    temperature: float = 0.0,
    max_images: int = 8,
    local_model: Any = None,
    api_client: Any = None,
) -> VisualQAAnswerer:
    """Select a backend by configuration. `auto` never downloads anything.

    `auto` prefers a hosted backend only when its key is already present, then an
    explicitly supplied local model, and otherwise falls back to the non-visual mock --
    which reports itself as non-visual, so the fallback is visible rather than silent.
    """
    kind = str(backend_type or "auto").strip().lower()
    if kind == "mock":
        return MockTextQAAnswerer()
    if kind == "local_vlm":
        return LocalVlmQAAnswerer(
            local_model,
            model_name=model_name,
            device=device,
            max_answer_tokens=max_answer_tokens,
            temperature=temperature,
        )
    if kind == "api":
        return ApiVqaAnswerer(
            model_name=model_name,
            max_images=max_images,
            max_answer_tokens=max_answer_tokens,
            client=api_client,
        )
    if kind != "auto":
        raise ValueError(
            f"Unsupported qa.backend.type {backend_type!r}; supported: mock, local_vlm, "
            "api, auto"
        )
    if local_model is not None:
        return LocalVlmQAAnswerer(
            local_model,
            model_name=model_name,
            device=device,
            max_answer_tokens=max_answer_tokens,
            temperature=temperature,
        )
    api = ApiVqaAnswerer(
        model_name=model_name, max_images=max_images,
        max_answer_tokens=max_answer_tokens, client=api_client,
    )
    if api.status().state != BACKEND_STATE_NOT_AVAILABLE:
        return api
    return MockTextQAAnswerer()


def _as_record(frame: QAEvidenceFrame) -> KeyframeRecord:
    """Adapt one evidence frame to the legacy `KeyframeRecord` the old answerers take."""
    import numpy as np

    return KeyframeRecord(
        id=frame.evidence_id,
        video_id=frame.video_id,
        timestamp=float(frame.timestamp),
        clip_embedding=np.zeros(1, dtype=np.float32),
        objects=list(frame.objects),
        llm_caption=frame.text or None,
    )


# ------------------------------------------------------------ legacy compatibility


@dataclass(frozen=True)
class EvidenceFrame:
    """Pre-Phase-6 evidence wrapper, kept for callers that still use records."""

    record: KeyframeRecord
    role: str
    relevance: float = 0.0


@dataclass(frozen=True)
class GroundedQAResult:
    video_id: str
    frame_id: str
    answer: str
    answer_normalized: str
    evidence_frame_ids: tuple[str, ...]
    grounding_score: float
    answer_confidence: float
    warning: str | None = None


class LocalVlmAnswerer(VqaAnswerer):
    """Deprecated stub. Use `LocalVlmQAAnswerer`, which has a real contract."""

    def answer(self, question: str, frames: Sequence[KeyframeRecord], images=None) -> VqaAnswer:
        raise QABackendUnavailable(
            "LocalVlmAnswerer is a legacy stub; use aic2026.qa.LocalVlmQAAnswerer."
        )


class OptionalApiVlmAnswerer(VqaAnswerer):
    """Deprecated stub. Use `ApiVqaAnswerer`, which has a real contract."""

    def answer(self, question: str, frames: Sequence[KeyframeRecord], images=None) -> VqaAnswer:
        raise QABackendUnavailable(
            "OptionalApiVlmAnswerer is a legacy stub; use aic2026.qa.ApiVqaAnswerer."
        )


_QUESTION_STOP_TERMS = frozenset(
    {
        "ai", "cai", "gi", "gì", "nao", "nào", "o", "ở", "dau", "đâu", "la", "là",
        "co", "có", "khong", "không", "phai", "phải", "what", "who", "where",
        "when", "why", "how", "is", "are", "does", "do", "did", "the", "a", "an",
    }
)


def build_evidence_retrieval_query(data: QAInput) -> str:
    """Retrieval text for evidence-first Q&A.

    The answerer may be weak or non-visual, so the first job is to find frames a human
    can inspect. Keep the user's event text, add question cues that describe visual
    evidence, and append cheap bilingual/object expansions. No external translation or
    model call happens here.
    """
    event = (data.event_description or "").strip()
    question = (data.question or "").strip()
    combined = " ".join(part for part in (event, question) if part)
    representation = normalize_query(combined)
    folded_question = fold_accents(question.casefold())
    hints: list[str] = []
    if any(term in folded_question for term in ("cam gi", "holding", "hold", "carrying", "wearing", "mac gi")):
        hints.extend(["person holding object", "hands", "item"])
    if any(term in folded_question for term in ("mau", "color", "colour")):
        hints.extend(["color", "clothing", "shirt", "vehicle"])
    if any(term in folded_question for term in ("bao nhieu", "how many", "count", "so luong")):
        hints.extend(["count", "many", "people", "objects"])
    if any(term in folded_question for term in ("sau khi", "after", "then", "tiep theo")):
        hints.extend(["after", "then", "next action"])
    if any(term in folded_question for term in ("truoc khi", "before")):
        hints.extend(["before", "previous action"])

    question_terms = [
        token for token in representation.tokens_folded
        if token not in _QUESTION_STOP_TERMS and len(token) > 1
    ]
    parts = [
        event,
        " ".join(question_terms[:12]),
        " ".join(representation.object_terms[:16]),
        " ".join(dict.fromkeys(hints)),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def build_retrieval_query(data: QAInput, mode: str = "event_only", question_weight: float = 0.35) -> str:
    if mode == "event_only":
        return data.event_description.strip() or data.question.strip()
    if mode == "evidence":
        return build_evidence_retrieval_query(data)
    if mode == "event_plus_question":
        return " ".join(part.strip() for part in (data.event_description, data.question) if part.strip())
    if mode == "weighted_combination":
        repeats = max(1, round(question_weight * 3))
        return " ".join([data.event_description.strip()] + [data.question.strip()] * repeats).strip()
    raise ValueError(f"Unknown Q&A retrieval query mode: {mode}")


def select_diverse_evidence(
    frames: Sequence[KeyframeRecord], center_time: float, count: int = 8, query: str = ""
) -> list[EvidenceFrame]:
    """Record-based wrapper over `select_evidence_frames`.

    Scores each record by query-token overlap with its text signals, then defers to the
    Phase 6 policy, so record-based callers get the same strongest-first behaviour.
    """
    query_terms = set(normalize_answer(query).split())
    by_id: dict[str, KeyframeRecord] = {}
    evidence: list[QAEvidenceFrame] = []
    for index, record in enumerate(frames):
        text = " ".join(
            part
            for part in (
                " ".join(record.objects or ()),
                record.ocr_text or "",
                record.asr_text or "",
                record.llm_caption or "",
            )
            if part
        )
        overlap = len(query_terms & set(normalize_answer(text).split()))
        # Closeness to the requested centre breaks ties, so an unscored window still
        # selects around the event rather than at its edges.
        proximity = 1.0 / (1.0 + abs(float(record.timestamp) - float(center_time)))
        identifier = str(record.id) or f"record_{index}"
        by_id[identifier] = record
        evidence.append(
            QAEvidenceFrame(
                video_id=str(record.video_id),
                frame_idx=None,
                timestamp=float(record.timestamp),
                source="keyframe_record",
                keyframe_id=identifier,
                retrieval_score=float(overlap) + 0.001 * proximity,
                text=text,
                objects=tuple(record.objects or ()),
            )
        )
    selected = select_evidence_frames(evidence, count=count, diversity_s=0.0)
    out: list[EvidenceFrame] = []
    for item in selected:
        record = by_id[item.evidence_id]
        role = (
            "event"
            if float(record.timestamp) == float(center_time)
            else "before" if float(record.timestamp) < float(center_time) else "after"
        )
        out.append(EvidenceFrame(record, role, item.retrieval_score))
    return out


def confidence_from_evidence(grounding_score: float, evidence_count: int, answer: str) -> float:
    """Pre-Phase-6 confidence. Superseded by `answer_reliability_score`."""
    known = not is_unknown_answer(answer)
    return max(0.0, min(1.0, 0.65 * grounding_score + 0.05 * min(evidence_count, 6) + (0.05 if known else 0.0)))


MockVqaAnswerer = MockVqaAnswerer
ABLATIONS = ("single_frame", "temporal_window", "diverse_multi_frame", "multi_frame_plus_text_signals", "full_grounded_qa")

__all__ = [
    "ANSWER_PROMPTS",
    "ANSWER_STATUS_ABSTAINED",
    "ANSWER_STATUS_ANSWERED",
    "ANSWER_STATUS_BACKEND_FAILED",
    "ANSWER_STATUS_BUDGET_EXHAUSTED",
    "ANSWER_STATUS_VISUAL_UNAVAILABLE",
    "ANSWER_TYPES",
    "ANSWER_TYPE_AUTO",
    "ANSWER_TYPE_BOOLEAN",
    "ANSWER_TYPE_COLOR",
    "ANSWER_TYPE_NUMBER",
    "ANSWER_TYPE_SHORT_TEXT",
    "BACKEND_STATE_NOT_AVAILABLE",
    "BACKEND_STATE_NOT_LOADED",
    "BACKEND_STATE_READY",
    "UNKNOWN_ANSWER",
    "ApiVqaAnswerer",
    "EvidenceFrame",
    "GroundedQAResult",
    "LocalVlmQAAnswerer",
    "MockTextQAAnswerer",
    "QAAnswerResult",
    "QAAnswererStatus",
    "QABackendUnavailable",
    "QAEvidenceBundle",
    "QAEvidenceFrame",
    "QAFrameHypothesis",
    "QAInput",
    "QAVideoHypothesis",
    "VisualQAAnswerer",
    "answer_matches_type",
    "answer_reliability_score",
    "build_answer_prompt",
    "build_evidence_retrieval_query",
    "build_qa_answerer",
    "build_retrieval_query",
    "canonical_answer_type",
    "confidence_from_evidence",
    "group_hypotheses_by_video",
    "is_unknown_answer",
    "normalize_answer",
    "select_diverse_evidence",
    "select_evidence_frames",
    "select_temporally_diverse",
]

"""Query-conditioned local refinement of coarse retrieval candidates.

Global retrieval is unchanged and stays authoritative for recall: BTC's precomputed
CLIP features index every mapped keyframe, and nothing here re-embeds the dataset. What
this module adds is a *local* second look. BTC keyframes are sparse in time, so the
frame that actually shows the queried moment often sits between two mapped keyframes
and was never indexed. For a handful of top candidates only, this module:

1. selects a small number of candidate regions (deduplicated in time per video),
2. decides — always, never, or by an explicit uncertainty heuristic — whether to look,
3. densely samples a bounded window of the ORIGINAL MP4 around each coarse frame,
4. scores those frames against the query with a `FrameScorer`,
5. reports the strongest local frame and a reranking contribution.

Three invariants hold regardless of what the scorer finds:

* **The official submission frame never changes.** Under the default
  `frame_output_policy="preserve_coarse"` the refined frame is evidence and a score,
  and `submission_frame_idx` remains the coarse mapped `frame_idx`. AIC has not
  confirmed the frame-ID semantics of an arbitrary decoded frame, so the alternative
  policy exists in the interface but is not the default.
* **Refinement can never lose a candidate.** A missing MP4, a decode failure, or an
  unavailable scorer marks that candidate `applied=False` and keeps its coarse result.
* **No accuracy claim.** There is no AIC ground truth in this repository. Everything
  reported here is structural: what was sampled, what was scored, how far the chosen
  frame sits from the coarse one. None of it is a quality measurement.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .frame_provider import DecodedFrame, FrameProvider
from .frame_scorer import FrameScorer, validate_scores

MODE_ALWAYS = "always"
MODE_UNCERTAINTY = "uncertainty"
MODE_DISABLED = "disabled"
REFINEMENT_MODES = (MODE_ALWAYS, MODE_UNCERTAINTY, MODE_DISABLED)

# preserve_coarse: the official frame_idx of the coarse candidate is submitted, whatever
# the local search finds. decoded_frame: submit the refined frame index instead — kept
# for a future in which AIC confirms the frame-ID semantics, never the default now.
FRAME_OUTPUT_PRESERVE_COARSE = "preserve_coarse"
FRAME_OUTPUT_DECODED_FRAME = "decoded_frame"
FRAME_OUTPUT_POLICIES = (FRAME_OUTPUT_PRESERVE_COARSE, FRAME_OUTPUT_DECODED_FRAME)

# Skip reasons. They are recorded verbatim in diagnostics so a run can be explained.
REASON_DISABLED = "refinement_disabled"
REASON_NO_CANDIDATES = "no_candidates"
REASON_ALWAYS = "mode_always"
REASON_MARGIN_BELOW_THRESHOLD = "top_score_margin_below_threshold"
REASON_MARGIN_ABOVE_THRESHOLD = "top_score_margin_above_threshold"
REASON_SINGLE_REGION = "single_candidate_region"
REASON_SCORER_UNAVAILABLE = "scorer_unavailable"
REASON_SCORER_FAILED = "scorer_failed"
REASON_VIDEO_UNAVAILABLE = "video_unavailable"
REASON_METADATA_UNAVAILABLE = "video_metadata_unavailable"
REASON_DECODE_FAILED = "decode_failed"
REASON_NO_COARSE_FRAME_IDX = "no_coarse_frame_idx"
REASON_REFINED = "refined"


@dataclass(frozen=True)
class RefinementConfig:
    """Runtime settings for local refinement.

    Defaults are the pre-Phase-5 values wherever one existed; the fields Phase 5 adds
    are deliberately conservative, because there is no ground truth to tune against.
    """

    enabled: bool = True
    # always | uncertainty | disabled. `uncertainty_only: true` in an older config file
    # is accepted as an alias for `mode: uncertainty` (see aic2026/config.py).
    mode: str = MODE_UNCERTAINTY
    # How many coarse candidates are examined when forming regions...
    top_hypotheses: int = 10
    # ...and how many of the resulting regions may actually be decoded and scored.
    candidate_budget: int = 5
    # Coarse candidates of one video closer than this are treated as a single region.
    region_merge_s: float = 1.0
    window_before_s: float = 4.0
    window_after_s: float = 4.0
    fine_fps: float = 4.0
    max_frames: int = 32
    batch_size: int = 16
    # Uncertainty heuristic: refine when the relative gap between the two best regions
    # is at most this. Not tuned against accuracy; it is a documented default.
    margin_threshold: float = 0.03
    cache_size_mb: int = 256
    # Frames are resized before scoring; CLIP works at 224px, so full-resolution frames
    # would cost ~30x the memory for no change in the embedding.
    scorer_input_max_side: int = 336
    # refined = coarse_fusion_score + alpha * (best_visual - coarse_visual).
    rerank_alpha: float = 0.10
    frame_output_policy: str = FRAME_OUTPUT_PRESERVE_COARSE
    scorer_type: str = "clip"
    scorer_model_name: str = "openai/clip-vit-base-patch32"
    scorer_device: str = "auto"
    # Production: fail loudly instead of pretending refinement happened.
    scorer_required: bool = False

    @property
    def effective_mode(self) -> str:
        return MODE_DISABLED if not self.enabled else str(self.mode)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.effective_mode,
            "candidate_budget": int(self.candidate_budget),
            "top_hypotheses": int(self.top_hypotheses),
            "window_before_s": float(self.window_before_s),
            "window_after_s": float(self.window_after_s),
            "sample_fps": float(self.fine_fps),
            "max_sampled_frames": int(self.max_frames),
            "batch_size": int(self.batch_size),
            "margin_threshold": float(self.margin_threshold),
            "rerank_alpha": float(self.rerank_alpha),
            "frame_output_policy": str(self.frame_output_policy),
            "scorer_type": str(self.scorer_type),
            "scorer_model_name": str(self.scorer_model_name),
            "scorer_device": str(self.scorer_device),
            "scorer_required": bool(self.scorer_required),
        }


@dataclass(frozen=True)
class RefinementCandidate:
    """One coarse candidate offered for refinement."""

    keyframe_id: str
    video_id: str
    coarse_frame_idx: Optional[int]
    timestamp: float
    coarse_score: float
    source_video: Optional[str] = None


@dataclass(frozen=True)
class LocalRefinementRequest:
    query: str
    candidates: tuple[RefinementCandidate, ...]


@dataclass(frozen=True)
class CandidateRegion:
    """A deduplicated local region: one anchor plus the candidates merged into it."""

    anchor: RefinementCandidate
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalRefinementFrame:
    """One sampled frame and what the scorer thought of it."""

    frame_idx: int
    timestamp: float
    score: float
    is_coarse_frame: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefinementDecision:
    """Why refinement did or did not run for this query."""

    mode: str
    triggered: bool
    reason: str
    margin: Optional[float] = None
    relative_margin: Optional[float] = None
    threshold: Optional[float] = None
    candidates_considered: int = 0
    regions_found: int = 0
    regions_selected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRefinement:
    """Full provenance for one candidate: coarse, refined, and what is submitted."""

    keyframe_id: str
    video_id: str
    applied: bool
    reason: str
    coarse_official_frame_idx: Optional[int]
    coarse_timestamp: float
    coarse_score: float
    submission_frame_idx: Optional[int]
    refined_score: float
    merged_keyframe_ids: tuple[str, ...] = ()
    sampled_frame_count: int = 0
    frames_decoded: int = 0
    best_visual_frame_idx: Optional[int] = None
    best_timestamp: Optional[float] = None
    best_visual_score: Optional[float] = None
    coarse_visual_score: Optional[float] = None
    score_gain: float = 0.0
    window_start_s: Optional[float] = None
    window_end_s: Optional[float] = None
    selected_offset_frames: Optional[int] = None
    selected_offset_seconds: Optional[float] = None
    best_is_coarse_frame: Optional[bool] = None
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    warning: Optional[str] = None
    frames: tuple[LocalRefinementFrame, ...] = ()

    def to_dict(self, *, include_frames: bool = False) -> dict[str, Any]:
        data = {
            key: value
            for key, value in asdict(self).items()
            if key != "frames"
        }
        data["merged_keyframe_ids"] = list(self.merged_keyframe_ids)
        if include_frames:
            data["frames"] = [frame.to_dict() for frame in self.frames]
        return data


@dataclass(frozen=True)
class LocalRefinementResult:
    decision: RefinementDecision
    refinements: tuple[CandidateRefinement, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        return any(item.applied for item in self.refinements)

    def by_keyframe(self) -> dict[str, CandidateRefinement]:
        return {item.keyframe_id: item for item in self.refinements}

    def to_dict(self, *, include_frames: bool = False) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "applied": self.applied,
            "candidates": [
                item.to_dict(include_frames=include_frames) for item in self.refinements
            ],
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------- sampling


def build_sample_plan(
    coarse_frame_idx: int,
    fps: float,
    frame_count: int,
    config: RefinementConfig,
) -> tuple[tuple[int, ...], float, float]:
    """Deterministic frame indices to sample around one coarse frame.

    The coarse frame is always included, sampling expands symmetrically outward so the
    budget is spent closest to the coarse hit first, the window is clamped to the real
    video, and `max_frames` is a hard cap. Videos do not share one frame rate, so the
    step is computed from this video's fps.
    """
    if fps <= 0:
        return (), 0.0, 0.0
    step = max(1, int(round(fps / max(1e-6, float(config.fine_fps)))))
    before = int(round(max(0.0, float(config.window_before_s)) * fps))
    after = int(round(max(0.0, float(config.window_after_s)) * fps))
    low = max(0, int(coarse_frame_idx) - before)
    high = int(coarse_frame_idx) + after
    if frame_count > 0:
        high = min(high, frame_count - 1)
    high = max(high, 0)
    low = min(low, high)
    anchor = min(max(int(coarse_frame_idx), low), high)

    limit = max(1, int(config.max_frames))
    plan: list[int] = [anchor]
    offset = 1
    while len(plan) < limit:
        earlier = anchor - offset * step
        later = anchor + offset * step
        if earlier < low and later > high:
            break
        for value in (earlier, later):
            if low <= value <= high and value not in plan and len(plan) < limit:
                plan.append(value)
        offset += 1
    return tuple(sorted(plan)), low / fps, high / fps


def merge_candidate_regions(
    candidates: Sequence[RefinementCandidate], merge_s: float
) -> tuple[CandidateRegion, ...]:
    """Collapse near-duplicate coarse candidates of one video into single regions.

    Two keyframes of the same video one second apart describe the same moment; decoding
    two overlapping windows for them would spend the budget twice on one region. The
    highest-scoring candidate anchors the region, and the others are recorded as members
    rather than discarded, so nothing disappears from the response.
    """
    ordered = sorted(
        candidates,
        key=lambda item: (-float(item.coarse_score), item.video_id, float(item.timestamp), item.keyframe_id),
    )
    anchors: list[RefinementCandidate] = []
    members: dict[str, list[str]] = {}
    threshold = max(0.0, float(merge_s))
    for candidate in ordered:
        host = next(
            (
                anchor
                for anchor in anchors
                if anchor.video_id == candidate.video_id
                and abs(float(anchor.timestamp) - float(candidate.timestamp)) <= threshold
            ),
            None,
        )
        if host is None:
            anchors.append(candidate)
            members[candidate.keyframe_id] = []
        else:
            members[host.keyframe_id].append(candidate.keyframe_id)
    return tuple(
        CandidateRegion(anchor=anchor, members=tuple(members[anchor.keyframe_id]))
        for anchor in anchors
    )


def decide_refinement(
    regions: Sequence[CandidateRegion], config: RefinementConfig, *, considered: int
) -> RefinementDecision:
    """Apply the configured trigger policy. Deterministic and fully reported.

    The uncertainty heuristic is intentionally simple: compare the two best candidate
    *regions*. Fused scores are not calibrated and not bounded to [0, 1], so the raw gap
    is normalized by the top score before it meets the threshold. A close pair means the
    coarse ranking did not separate them and a local look may help; a clear leader means
    it did.
    """
    mode = config.effective_mode
    base = {
        "mode": mode,
        "candidates_considered": int(considered),
        "regions_found": len(regions),
        "threshold": float(config.margin_threshold),
    }
    if mode == MODE_DISABLED:
        return RefinementDecision(triggered=False, reason=REASON_DISABLED, **base)
    if not regions:
        return RefinementDecision(triggered=False, reason=REASON_NO_CANDIDATES, **base)

    budget = max(1, int(config.candidate_budget))
    selected = min(budget, len(regions))
    if mode == MODE_ALWAYS:
        return RefinementDecision(
            triggered=True, reason=REASON_ALWAYS, regions_selected=selected, **base
        )

    if len(regions) < 2:
        # No second region means there is no evidence of separation at all, so the top
        # hit is treated as unconfirmed rather than as confidently correct.
        return RefinementDecision(
            triggered=True, reason=REASON_SINGLE_REGION, regions_selected=selected, **base
        )
    top = float(regions[0].anchor.coarse_score)
    second = float(regions[1].anchor.coarse_score)
    margin = top - second
    scale = max(abs(top), 1e-9)
    relative = margin / scale
    triggered = relative <= float(config.margin_threshold)
    return RefinementDecision(
        triggered=triggered,
        reason=REASON_MARGIN_BELOW_THRESHOLD if triggered else REASON_MARGIN_ABOVE_THRESHOLD,
        margin=float(margin),
        relative_margin=float(relative),
        regions_selected=selected if triggered else 0,
        **base,
    )


# --------------------------------------------------------------------- refiner


class LocalFrameRefiner:
    """Runs bounded, query-conditioned local search over original MP4 frames.

    It knows nothing about CLIP: video access goes through `FrameProvider` (the one
    shared OpenCV implementation) and visual scoring through a `FrameScorer`. Both are
    injected, which is what makes the algorithm testable offline with a fake scorer and
    tiny synthetic MP4s.
    """

    def __init__(
        self,
        config: RefinementConfig | None = None,
        *,
        frame_provider: FrameProvider | None = None,
        scorer: FrameScorer | None = None,
    ):
        self.config = config or RefinementConfig()
        self.frame_provider = frame_provider or FrameProvider()
        self.scorer = scorer
        self._window_cache: OrderedDict[tuple, list[DecodedFrame]] = OrderedDict()
        self._window_cache_bytes = 0

    # ------------------------------------------------------------------ helpers

    def scorer_status(self, *, initialize: bool = False) -> dict[str, Any]:
        if self.scorer is None:
            return {
                "backend": str(self.config.scorer_type),
                "model_name": str(self.config.scorer_model_name),
                "device": str(self.config.scorer_device),
                "state": "unavailable",
                "available": False,
                "production_ready": False,
                "fallback_reason": "No visual frame scorer is attached to this refiner.",
                "warning": "Local refinement is skipped; coarse retrieval is unaffected.",
            }
        status_fn = getattr(self.scorer, "status", None)
        if not callable(status_fn):
            return {
                "backend": type(self.scorer).__name__,
                "model_name": "unknown",
                "device": "unknown",
                "state": "ready",
                "available": True,
                "production_ready": False,
                "warning": "Injected scorer has no status contract.",
            }
        try:
            status = status_fn(initialize=initialize)
        except TypeError:
            status = status_fn()
        return status.to_dict() if hasattr(status, "to_dict") else dict(status)

    def _cache_get(self, key: tuple) -> list[DecodedFrame] | None:
        frames = self._window_cache.get(key)
        if frames is not None:
            self._window_cache.move_to_end(key)
        return frames

    def _cache_put(self, key: tuple, frames: list[DecodedFrame]) -> None:
        limit = max(0, int(self.config.cache_size_mb)) * 1024 * 1024
        if limit <= 0 or not frames:
            return
        size = sum(int(getattr(frame.image, "nbytes", 0)) for frame in frames)
        if size > limit:
            return
        self._window_cache[key] = frames
        self._window_cache_bytes += size
        while self._window_cache and self._window_cache_bytes > limit:
            _, removed = self._window_cache.popitem(last=False)
            self._window_cache_bytes -= sum(
                int(getattr(frame.image, "nbytes", 0)) for frame in removed
            )

    def _downscale(self, image: Any) -> Any:
        """Shrink to the scorer's working scale before anything is held in memory."""
        max_side = max(32, int(self.config.scorer_input_max_side))
        array = np.asarray(image)
        if array.ndim < 2:
            return array
        height, width = int(array.shape[0]), int(array.shape[1])
        longest = max(height, width)
        if longest <= max_side:
            return array
        scale = max_side / float(longest)
        try:
            import cv2
        except ImportError:  # pragma: no cover - OpenCV is a project dependency
            return array
        return cv2.resize(
            array,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    # -------------------------------------------------------------------- refine

    def refine(self, request: LocalRefinementRequest) -> LocalRefinementResult:
        """Refine a query's candidates. Never raises for data or model problems."""
        started = time.perf_counter()
        config = self.config
        candidates = tuple(request.candidates)
        considered = candidates[: max(1, int(config.top_hypotheses))]

        if config.effective_mode == MODE_DISABLED:
            decision = RefinementDecision(
                mode=MODE_DISABLED,
                triggered=False,
                reason=REASON_DISABLED,
                candidates_considered=0,
                threshold=float(config.margin_threshold),
            )
            return LocalRefinementResult(
                decision=decision,
                diagnostics=self._diagnostics(decision, (), started),
            )

        regions = merge_candidate_regions(considered, config.region_merge_s)
        decision = decide_refinement(regions, config, considered=len(considered))
        if not decision.triggered:
            return LocalRefinementResult(
                decision=decision, diagnostics=self._diagnostics(decision, (), started)
            )

        selected = regions[: max(1, int(config.candidate_budget))]
        warnings: list[str] = []

        prepared, scorer_error = self._prepare_query(request.query)
        if scorer_error is not None:
            if config.scorer_required:
                raise RuntimeError(
                    "refinement.scorer_required is set but the visual scorer is "
                    f"unavailable: {scorer_error}"
                )
            warnings.append(scorer_error)
            skipped = tuple(
                self._skipped(region, REASON_SCORER_UNAVAILABLE, scorer_error)
                for region in selected
            )
            return LocalRefinementResult(
                decision=decision,
                refinements=skipped,
                diagnostics=self._diagnostics(decision, skipped, started),
                warnings=tuple(warnings),
            )

        # 1) Decode every selected window first, one capture per video.
        plans: list[tuple[CandidateRegion, tuple[int, ...], float, float, list[DecodedFrame], float, str | None]] = []
        for region in selected:
            plans.append(self._decode_region(region))

        # 2) Score everything in ONE batched call: the query embedding is prepared once
        #    per request and the frames go through the scorer together.
        flat_frames: list[Any] = []
        spans: list[tuple[int, int]] = []
        for _, _, _, _, decoded, _, _ in plans:
            start = len(flat_frames)
            flat_frames.extend(frame.image for frame in decoded)
            spans.append((start, len(flat_frames)))

        inference_ms = 0.0
        scores: tuple[float, ...] = ()
        scoring_error: str | None = None
        if flat_frames:
            inference_started = time.perf_counter()
            try:
                raw_scores = self.scorer.score_frames(prepared, flat_frames)
                scores = validate_scores(raw_scores, len(flat_frames))
            except Exception as exc:  # noqa: BLE001 - degrades to coarse results
                if config.scorer_required:
                    raise
                scoring_error = f"Visual scoring failed: {type(exc).__name__}: {exc}"
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
        if scoring_error:
            warnings.append(scoring_error)

        refinements: list[CandidateRefinement] = []
        share = inference_ms / max(1, len(plans))
        for (region, plan, window_start, window_end, decoded, decode_ms, decode_warning), (
            lo,
            hi,
        ) in zip(plans, spans):
            if scoring_error:
                refinements.append(
                    self._skipped(
                        region,
                        REASON_SCORER_FAILED,
                        scoring_error,
                        decode_ms=decode_ms,
                        frames_decoded=len(decoded),
                        sampled=len(plan),
                    )
                )
                continue
            refinements.append(
                self._assemble(
                    region,
                    plan,
                    window_start,
                    window_end,
                    decoded,
                    scores[lo:hi],
                    decode_ms=decode_ms,
                    inference_ms=share,
                    warning=decode_warning,
                )
            )

        result = tuple(refinements)
        return LocalRefinementResult(
            decision=decision,
            refinements=result,
            diagnostics=self._diagnostics(decision, result, started),
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------ internals

    def _prepare_query(self, query: str) -> tuple[Any, str | None]:
        """Embed the query exactly once per refinement request."""
        if self.scorer is None:
            return None, "No visual frame scorer is available; local refinement skipped."
        try:
            return self.scorer.prepare_query(str(query)), None
        except Exception as exc:  # noqa: BLE001 - a model load failure is not a bug
            return None, f"Visual scorer unavailable: {type(exc).__name__}: {exc}"

    def _decode_region(
        self, region: CandidateRegion
    ) -> tuple[CandidateRegion, tuple[int, ...], float, float, list[DecodedFrame], float, str | None]:
        anchor = region.anchor
        if anchor.coarse_frame_idx is None:
            return region, (), 0.0, 0.0, [], 0.0, REASON_NO_COARSE_FRAME_IDX
        metadata = self.frame_provider.video_metadata(
            anchor.video_id, source_video=anchor.source_video
        )
        if metadata is None or not metadata.usable:
            return region, (), 0.0, 0.0, [], 0.0, REASON_METADATA_UNAVAILABLE
        plan, window_start, window_end = build_sample_plan(
            int(anchor.coarse_frame_idx), metadata.fps, metadata.frame_count, self.config
        )
        if not plan:
            return region, (), window_start, window_end, [], 0.0, REASON_DECODE_FAILED

        cache_key = (str(anchor.video_id), str(anchor.source_video or ""), plan,
                     int(self.config.scorer_input_max_side))
        cached = self._cache_get(cache_key)
        if cached is not None:
            return region, plan, window_start, window_end, cached, 0.0, None

        started = time.perf_counter()
        decoded, warning = self.frame_provider.decode_frames(
            anchor.video_id, plan, source_video=anchor.source_video
        )
        shrunk = [
            DecodedFrame(
                requested_frame_idx=frame.requested_frame_idx,
                decoded_frame_idx=frame.decoded_frame_idx,
                timestamp=frame.timestamp,
                image=self._downscale(frame.image),
            )
            for frame in decoded
        ]
        decode_ms = (time.perf_counter() - started) * 1000.0
        self._cache_put(cache_key, shrunk)
        return region, plan, window_start, window_end, shrunk, decode_ms, warning

    def _skipped(
        self,
        region: CandidateRegion,
        reason: str,
        warning: str | None,
        *,
        decode_ms: float = 0.0,
        frames_decoded: int = 0,
        sampled: int = 0,
    ) -> CandidateRefinement:
        """A candidate that was not refined keeps its coarse result untouched."""
        anchor = region.anchor
        return CandidateRefinement(
            keyframe_id=anchor.keyframe_id,
            video_id=anchor.video_id,
            applied=False,
            reason=reason,
            coarse_official_frame_idx=anchor.coarse_frame_idx,
            coarse_timestamp=float(anchor.timestamp),
            coarse_score=float(anchor.coarse_score),
            submission_frame_idx=anchor.coarse_frame_idx,
            refined_score=float(anchor.coarse_score),
            merged_keyframe_ids=region.members,
            sampled_frame_count=int(sampled),
            frames_decoded=int(frames_decoded),
            decode_ms=round(float(decode_ms), 3),
            total_ms=round(float(decode_ms), 3),
            warning=warning,
        )

    def _assemble(
        self,
        region: CandidateRegion,
        plan: tuple[int, ...],
        window_start: float,
        window_end: float,
        decoded: Sequence[DecodedFrame],
        scores: Sequence[float],
        *,
        decode_ms: float,
        inference_ms: float,
        warning: str | None,
    ) -> CandidateRefinement:
        anchor = region.anchor
        if not decoded:
            reason = warning if warning in {
                REASON_NO_COARSE_FRAME_IDX,
                REASON_METADATA_UNAVAILABLE,
                REASON_DECODE_FAILED,
            } else REASON_VIDEO_UNAVAILABLE
            message = (
                "No frame could be decoded from the original MP4; the coarse candidate "
                "is unchanged."
            )
            return self._skipped(region, reason, message, decode_ms=decode_ms, sampled=len(plan))

        coarse_idx = None if anchor.coarse_frame_idx is None else int(anchor.coarse_frame_idx)
        frames = tuple(
            LocalRefinementFrame(
                frame_idx=int(frame.requested_frame_idx),
                timestamp=float(frame.timestamp),
                score=float(score),
                is_coarse_frame=int(frame.requested_frame_idx) == coarse_idx,
            )
            for frame, score in zip(decoded, scores)
        )
        # Deterministic: on an exact tie the earliest frame index wins, so equal scores
        # never produce run-to-run variation.
        best = min(frames, key=lambda item: (-item.score, item.frame_idx))
        coarse_frame = next((item for item in frames if item.is_coarse_frame), None)
        coarse_visual = None if coarse_frame is None else float(coarse_frame.score)
        gain = 0.0 if coarse_visual is None else float(best.score) - coarse_visual
        # Bounded on purpose: the contribution is a cosine-similarity difference, so a
        # pathological scorer cannot dominate the fused score.
        bounded_gain = float(max(-1.0, min(1.0, gain)))
        refined_score = float(anchor.coarse_score) + float(self.config.rerank_alpha) * bounded_gain

        # The official submission frame follows the configured policy and defaults to
        # the coarse mapped frame_idx. It is never derived from the decoder's behaviour.
        submission = coarse_idx
        if str(self.config.frame_output_policy) == FRAME_OUTPUT_DECODED_FRAME:
            submission = int(best.frame_idx)

        offset_frames = None if coarse_idx is None else int(best.frame_idx) - coarse_idx
        offset_seconds = (
            None
            if coarse_frame is None
            else float(best.timestamp) - float(coarse_frame.timestamp)
        )
        return CandidateRefinement(
            keyframe_id=anchor.keyframe_id,
            video_id=anchor.video_id,
            applied=True,
            reason=REASON_REFINED,
            coarse_official_frame_idx=coarse_idx,
            coarse_timestamp=float(anchor.timestamp),
            coarse_score=float(anchor.coarse_score),
            submission_frame_idx=submission,
            refined_score=refined_score,
            merged_keyframe_ids=region.members,
            sampled_frame_count=len(plan),
            frames_decoded=len(frames),
            best_visual_frame_idx=int(best.frame_idx),
            best_timestamp=float(best.timestamp),
            best_visual_score=float(best.score),
            coarse_visual_score=coarse_visual,
            score_gain=float(gain),
            window_start_s=float(window_start),
            window_end_s=float(window_end),
            selected_offset_frames=offset_frames,
            selected_offset_seconds=offset_seconds,
            best_is_coarse_frame=bool(best.is_coarse_frame),
            decode_ms=round(float(decode_ms), 3),
            inference_ms=round(float(inference_ms), 3),
            total_ms=round(float(decode_ms) + float(inference_ms), 3),
            warning=warning,
            frames=frames,
        )

    def _diagnostics(
        self,
        decision: RefinementDecision,
        refinements: Sequence[CandidateRefinement],
        started: float,
    ) -> dict[str, Any]:
        """Structural counters only. None of this is an accuracy measurement."""
        applied = [item for item in refinements if item.applied]
        moved = [item for item in applied if item.best_is_coarse_frame is False]
        return {
            "refinement_triggered": bool(decision.triggered),
            "trigger_reason": decision.reason,
            "mode": decision.mode,
            "candidates_considered": decision.candidates_considered,
            "candidates_selected": decision.regions_selected,
            "candidates_refined": len(applied),
            "frames_decoded": sum(item.frames_decoded for item in refinements),
            "frames_scored": sum(item.frames_decoded for item in applied),
            "decode_failures": sum(
                1
                for item in refinements
                if not item.applied
                and item.reason in {REASON_VIDEO_UNAVAILABLE, REASON_DECODE_FAILED,
                                    REASON_METADATA_UNAVAILABLE}
            ),
            "scorer_failures": sum(
                1
                for item in refinements
                if not item.applied
                and item.reason in {REASON_SCORER_UNAVAILABLE, REASON_SCORER_FAILED}
            ),
            "best_differs_from_coarse": len(moved),
            "mean_visual_score_gain": (
                round(sum(item.score_gain for item in applied) / len(applied), 6)
                if applied
                else 0.0
            ),
            "mean_absolute_offset_seconds": (
                round(
                    sum(abs(item.selected_offset_seconds or 0.0) for item in applied)
                    / len(applied),
                    3,
                )
                if applied
                else 0.0
            ),
            "decode_ms": round(sum(item.decode_ms for item in refinements), 3),
            "inference_ms": round(sum(item.inference_ms for item in refinements), 3),
            "refinement_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


def aggregate_diagnostics(results: Sequence[LocalRefinementResult]) -> dict[str, Any]:
    """Roll several searches into run-level structural statistics.

    Explicitly NOT precision, recall, or accuracy: without AIC ground truth none of
    these numbers can say whether a refined frame is better, only that it differs.
    """
    items = list(results)
    if not items:
        return {"searches": 0}
    latencies = sorted(float(item.diagnostics.get("refinement_ms", 0.0)) for item in items)
    refined = [
        candidate
        for item in items
        for candidate in item.refinements
        if candidate.applied
    ]
    moved = [c for c in refined if c.best_is_coarse_frame is False]

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        position = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
        return round(values[position], 3)

    return {
        "searches": len(items),
        "trigger_rate": round(
            sum(1 for item in items if item.decision.triggered) / len(items), 4
        ),
        "candidates_refined_total": len(refined),
        "mean_candidates_refined": round(len(refined) / len(items), 3),
        "frames_decoded_total": sum(
            int(item.diagnostics.get("frames_decoded", 0)) for item in items
        ),
        "mean_frames_decoded": round(
            sum(int(item.diagnostics.get("frames_decoded", 0)) for item in items) / len(items), 3
        ),
        "refinement_ms_p50": percentile(latencies, 0.5),
        "refinement_ms_p95": percentile(latencies, 0.95),
        "best_differs_from_coarse": len(moved),
        "fraction_best_differs_from_coarse": (
            round(len(moved) / len(refined), 4) if refined else 0.0
        ),
        "mean_absolute_offset_seconds": (
            round(sum(abs(c.selected_offset_seconds or 0.0) for c in moved) / len(moved), 3)
            if moved
            else 0.0
        ),
        "mean_visual_score_gain": (
            round(sum(c.score_gain for c in refined) / len(refined), 6) if refined else 0.0
        ),
        "decode_failures": sum(
            int(item.diagnostics.get("decode_failures", 0)) for item in items
        ),
        "scorer_failures": sum(
            int(item.diagnostics.get("scorer_failures", 0)) for item in items
        ),
        "note": "Structural diagnostics only; no AIC ground truth exists, so none of "
                "these values measures retrieval accuracy.",
    }


__all__ = [
    "FRAME_OUTPUT_DECODED_FRAME",
    "FRAME_OUTPUT_POLICIES",
    "FRAME_OUTPUT_PRESERVE_COARSE",
    "MODE_ALWAYS",
    "MODE_DISABLED",
    "MODE_UNCERTAINTY",
    "REFINEMENT_MODES",
    "REASON_ALWAYS",
    "REASON_DECODE_FAILED",
    "REASON_DISABLED",
    "REASON_MARGIN_ABOVE_THRESHOLD",
    "REASON_MARGIN_BELOW_THRESHOLD",
    "REASON_REFINED",
    "REASON_SCORER_FAILED",
    "REASON_SCORER_UNAVAILABLE",
    "REASON_SINGLE_REGION",
    "REASON_VIDEO_UNAVAILABLE",
    "CandidateRefinement",
    "CandidateRegion",
    "LocalFrameRefiner",
    "LocalRefinementFrame",
    "LocalRefinementRequest",
    "LocalRefinementResult",
    "RefinementCandidate",
    "RefinementConfig",
    "RefinementDecision",
    "aggregate_diagnostics",
    "build_sample_plan",
    "decide_refinement",
    "merge_candidate_regions",
]

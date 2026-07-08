"""Temporal Consistency Check — lọc ràng buộc thứ tự thời gian (CLAUDE.md Mục 4.5).

BÀI TOÁN (thách thức #3 — temporal logic): query kiểu "cởi mũ TRƯỚC KHI vào phòng"
đòi hỏi các sự kiện xảy ra ĐÚNG THỨ TỰ. Sau khi đã có ứng viên keyframe cho TỪNG sự
kiện, ta chỉ giữ lại các tổ hợp (cùng video) mà timestamp tăng dần đúng thứ tự yêu cầu.

ĐÂY LÀ LỌC LOGIC CỨNG (hard-constraint), KHÔNG phải similarity — Mục 4.5 nhấn mạnh
KHÔNG được gộp bước này vào fusion điểm số (4.3). Một cặp sai thứ tự bị loại DỨT KHOÁT,
dù điểm tương đồng có cao đến đâu.

Thuần logic (không cần model/API) -> chạy/test offline hoàn toàn.

Đầu vào `candidates_by_event`: ánh xạ tên-sự-kiện -> danh sách ứng viên. Ứng viên chỉ
cần có 3 thuộc tính (duck-typing): `keyframe_id`, `video_id`, `timestamp` (và tuỳ chọn
`score`). Nhờ vậy dùng được trực tiếp Candidate (Phase 3) lẫn RerankedCandidate (Phase 5).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class _TemporalCandidate(Protocol):
    keyframe_id: str
    video_id: str
    timestamp: float


@dataclass
class TemporalStep:
    """Một mắt xích trong chuỗi sự kiện đã khớp thứ tự."""

    event: str
    keyframe_id: str
    timestamp: float


@dataclass
class TemporalMatch:
    """Một tổ hợp keyframe (cùng video) thoả ĐÚNG thứ tự thời gian yêu cầu."""

    video_id: str
    steps: list[TemporalStep] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def timestamps(self) -> list[float]:
        return [s.timestamp for s in self.steps]


def _group_by_video(candidates: Sequence[_TemporalCandidate]) -> dict[str, list]:
    by_video: dict[str, list] = defaultdict(list)
    for c in candidates:
        by_video[c.video_id].append(c)
    return by_video


def _enumerate_chains(
    event_lists: list[list],
    idx: int,
    prev_ts: float,
    acc: list,
    out: list[list],
    max_results: int,
) -> None:
    """Đệ quy sinh các chuỗi có timestamp TĂNG NGHIÊM NGẶT theo thứ tự sự kiện.

    event_lists[k] = ứng viên của sự kiện thứ k (trong CÙNG 1 video), đã sắp theo
    timestamp. Ta chọn 1 ứng viên mỗi sự kiện sao cho timestamp > mắt xích trước.
    """
    if len(out) >= max_results:
        return
    if idx == len(event_lists):
        out.append(list(acc))
        return
    for cand in event_lists[idx]:
        if cand.timestamp > prev_ts:  # strictly increasing (t_k < t_{k+1})
            acc.append(cand)
            _enumerate_chains(event_lists, idx + 1, cand.timestamp, acc, out, max_results)
            acc.pop()
            if len(out) >= max_results:
                return


def temporal_consistency_filter(
    temporal_order: list[dict],
    candidates_by_event: dict[str, Sequence[_TemporalCandidate]],
    max_results: int = 200,
) -> list[TemporalMatch]:
    """Trả các TemporalMatch thoả đúng thứ tự thời gian (Mục 4.5).

    Args:
        temporal_order: [{"event": str, "order": int}, ...] (từ StructuredQuery).
        candidates_by_event: tên-sự-kiện -> ứng viên keyframe của sự kiện đó.
        max_results: trần số tổ hợp trả về (tránh bùng nổ tổ hợp khi nhiều ứng viên).

    Returns:
        Danh sách TemporalMatch, sắp giảm dần theo total_score (rồi theo thời điểm
        bắt đầu sớm hơn), mỗi cái là một chuỗi keyframe cùng video có timestamp tăng
        dần đúng thứ tự. Rỗng nếu KHÔNG tồn tại tổ hợp hợp lệ (vd đúng sự kiện nhưng
        sai thứ tự thời gian).

    Ràng buộc: cần >= 2 sự kiện mới có "thứ tự" để kiểm.
    """
    events_sorted = sorted(temporal_order, key=lambda e: e["order"])
    event_names = [e["event"] for e in events_sorted]
    if len(event_names) < 2:
        raise ValueError(
            "temporal_consistency_filter cần >= 2 sự kiện có thứ tự để kiểm tra."
        )
    # Sự kiện nào thiếu ứng viên -> không thể tạo chuỗi đầy đủ -> rỗng.
    if any(not candidates_by_event.get(name) for name in event_names):
        return []

    # Nhóm ứng viên từng sự kiện theo video.
    per_event_by_video = {
        name: _group_by_video(candidates_by_event[name]) for name in event_names
    }
    # Chỉ xét video XUẤT HIỆN Ở MỌI sự kiện (mới có thể tạo chuỗi đủ mắt xích).
    common_videos = set(per_event_by_video[event_names[0]])
    for name in event_names[1:]:
        common_videos &= set(per_event_by_video[name])

    matches: list[TemporalMatch] = []
    for video in common_videos:
        event_lists = [
            sorted(per_event_by_video[name][video], key=lambda c: c.timestamp)
            for name in event_names
        ]
        chains: list[list] = []
        _enumerate_chains(event_lists, 0, float("-inf"), [], chains, max_results)
        for chain in chains:
            steps = [
                TemporalStep(event=event_names[k], keyframe_id=c.keyframe_id,
                             timestamp=c.timestamp)
                for k, c in enumerate(chain)
            ]
            total = sum(float(getattr(c, "score", 0.0)) for c in chain)
            matches.append(TemporalMatch(video_id=video, steps=steps, total_score=total))

    # Sắp: điểm cao trước; hoà điểm thì chuỗi bắt đầu sớm hơn trước (tất định).
    matches.sort(key=lambda m: (-m.total_score, m.timestamps[0]))
    return matches[:max_results]


def has_valid_ordering(
    temporal_order: list[dict],
    candidates_by_event: dict[str, Sequence[_TemporalCandidate]],
) -> bool:
    """Tiện ích: có tồn tại ÍT NHẤT một tổ hợp đúng thứ tự không?"""
    return bool(
        temporal_consistency_filter(temporal_order, candidates_by_event, max_results=1)
    )

"""Deduplication keyframe gần trùng trước khi index (CLAUDE.md Mục 5.1).

VẤN ĐỀ: video lifelog có rất nhiều frame liên tiếp gần giống hệt nhau (người đứng
yên, cảnh tĩnh). Nếu index tất cả, ta phí RAM/thời gian cho hàng loạt vector gần
như đồng nhất về ngữ nghĩa. Mục 5.1 ước tính gộp chúng giảm 30-60% số vector.

RÀNG BUỘC CHI PHỐI THIẾT KẾ (Mục 1.2 — KHÔNG THƯƠNG LƯỢNG): "không được đánh mất
recall ở tầng lọc thô". Keyframe bị loại ở đây KHÔNG BAO GIỜ được xét lại. Do đó
dedup chỉ được phép gộp các frame *thực sự* gần trùng — tuyệt đối không gộp nhầm 2
cảnh khác ngữ nghĩa.

QUYẾT ĐỊNH KỸ THUẬT — vì sao so khớp theo ANCHOR chứ không theo frame liền trước:
  Mục 5.1 mô tả "cosine similarity giữa các keyframe LIÊN TIẾP". Nếu chỉ so mỗi
  frame với frame ngay trước nó, một chuỗi frame biến đổi CHẬM có thể khiến từng cặp
  liên tiếp đều > ngưỡng, nhưng frame đầu và frame cuối cụm lại KHÁC HẲN nhau
  ("semantic drift"). Gộp cả chuỗi đó thành 1 đại diện sẽ làm mất frame cuối khỏi
  index -> mất recall (vi phạm Mục 1.2).
  Vì vậy ta vẫn duyệt frame theo thứ tự thời gian LIÊN TIẾP (cụm luôn là một đoạn
  liền mạch), nhưng điều kiện để một frame gia nhập cụm là nó phải giống ĐẠI DIỆN
  (anchor = frame đầu cụm), không phải chỉ giống frame liền trước. Cách này chặn
  drift: ngay khi frame trôi đủ xa khỏi anchor, ta mở cụm mới -> không mất recall.

Cách chạy offline hoàn toàn (Mục 1.5): chỉ dùng numpy, không GPU/API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

import numpy as np

from .schemas import KeyframeRecord


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity giữa 2 vector, có bảo vệ vector không (norm = 0).

    Trả về giá trị trong [-1, 1]. Nếu một trong hai vector có norm 0 (không hợp lệ
    cho hướng ngữ nghĩa), trả 0.0 để KHÔNG vô tình gộp (an toàn cho recall).
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class DedupResult:
    """Kết quả dedup + số liệu để test/benchmark kiểm chứng DoD Phase 1."""

    representatives: list[KeyframeRecord]
    num_input: int
    num_output: int
    num_clusters: int
    # cluster_members[i] = danh sách id các keyframe thuộc cụm của representatives[i]
    cluster_members: list[list[str]] = field(default_factory=list)

    @property
    def reduction_ratio(self) -> float:
        """Tỉ lệ giảm số record: 1 - out/in. DoD Phase 1 yêu cầu > 0.30."""
        if self.num_input == 0:
            return 0.0
        return 1.0 - (self.num_output / self.num_input)


def _dedup_single_video(
    frames: list[KeyframeRecord],
    similarity_threshold: float,
    embedding_attr: str,
) -> list[tuple[KeyframeRecord, list[KeyframeRecord]]]:
    """Dedup các frame CỦA MỘT video (đã cùng video_id).

    Trả về danh sách (representative, [members...]) theo thứ tự thời gian.
    """
    # Sắp theo timestamp để "liên tiếp" đúng nghĩa thời gian. Ổn định (stable) để
    # các frame cùng timestamp giữ nguyên thứ tự đầu vào -> kết quả tất định.
    frames = sorted(frames, key=lambda kf: kf.timestamp)

    clusters: list[tuple[KeyframeRecord, list[KeyframeRecord]]] = []
    anchor: KeyframeRecord | None = None
    members: list[KeyframeRecord] = []

    for kf in frames:
        emb = getattr(kf, embedding_attr)
        if emb is None:
            raise ValueError(
                f"Keyframe {kf.id!r} thiếu embedding {embedding_attr!r} — "
                "không thể dedup. Hãy trích xuất embedding trước (Phase 2)."
            )
        if anchor is None:
            # Bắt đầu cụm đầu tiên.
            anchor, members = kf, [kf]
            continue

        sim = cosine_similarity(getattr(anchor, embedding_attr), emb)
        if sim >= similarity_threshold:
            # Đủ giống ĐẠI DIỆN -> cùng cụm (frame gần trùng, gộp an toàn).
            members.append(kf)
        else:
            # Trôi quá xa anchor -> chốt cụm cũ, mở cụm mới với kf làm anchor.
            clusters.append((anchor, members))
            anchor, members = kf, [kf]

    if anchor is not None:
        clusters.append((anchor, members))
    return clusters


def deduplicate_keyframes(
    records: list[KeyframeRecord],
    similarity_threshold: float = 0.97,
    embedding_attr: str = "clip_embedding",
) -> DedupResult:
    """Gộp keyframe gần trùng theo từng video (Mục 5.1).

    Args:
        records: danh sách KeyframeRecord đầu vào (nhiều video trộn lẫn được).
        similarity_threshold: ngưỡng cosine để coi là "gần trùng" (mặc định 0.97
            theo configs/settings.yaml / Mục 5.1). Càng CAO càng thận trọng (gộp ít,
            an toàn recall hơn); càng thấp càng gộp nhiều (rủi ro mất recall).
        embedding_attr: dùng embedding nào để so khớp (mặc định clip_embedding — có
            sẵn từ BTC nên không tốn compute).

    Returns:
        DedupResult: chỉ chứa các keyframe ĐẠI DIỆN (1 per cụm), mỗi đại diện được
        gắn is_cluster_representative=True và cluster_span=(t_start, t_end) nếu cụm có
        >1 frame (singleton giữ cluster_span=None vì không thực sự là "cụm").

    Đảm bảo (được test ở Phase 1): mỗi cụm giữ ĐÚNG 1 đại diện -> không cụm nào mất
    đại diện; và trên dữ liệu giả mô phỏng chuỗi frame gần trùng, giảm > 30% record.
    """
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError(
            f"similarity_threshold phải trong (0, 1], nhận {similarity_threshold}"
        )

    representatives: list[KeyframeRecord] = []
    cluster_members: list[list[str]] = []

    # Nhóm theo video_id. Sắp xếp trước để groupby gom đúng (groupby chỉ gom các
    # phần tử LIỀN KỀ có cùng key).
    records_sorted = sorted(records, key=lambda kf: kf.video_id)
    for video_id, group in groupby(records_sorted, key=lambda kf: kf.video_id):
        clusters = _dedup_single_video(
            list(group), similarity_threshold, embedding_attr
        )
        for anchor, members in clusters:
            anchor.is_cluster_representative = True
            if len(members) > 1:
                t_start = min(m.timestamp for m in members)
                t_end = max(m.timestamp for m in members)
                anchor.cluster_span = (t_start, t_end)
            else:
                anchor.cluster_span = None
            representatives.append(anchor)
            cluster_members.append([m.id for m in members])

    return DedupResult(
        representatives=representatives,
        num_input=len(records),
        num_output=len(representatives),
        num_clusters=len(representatives),
        cluster_members=cluster_members,
    )

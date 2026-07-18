"""Adapter tích hợp dialogue với retriever THẬT (CLAUDE.md Phase 8, Mục 8).

MỤC TIÊU: thay `MockRetriever` của dialogue bằng một `HybridRetriever` thật chạy
trên KeyframeIndex + CoarseRetriever (Phase 3), để vòng hội thoại KISC hoạt động trên
index thật thay vì dữ liệu sinh ngẫu nhiên trong bộ nhớ.

TÔN TRỌNG Mục 10.6: KHÔNG sửa bất kỳ file nào trong dialogue/. Ta chỉ:
  - Kế thừa interface `dialogue.HybridRetriever` (đã có sẵn) và implement `search`.
  - Trả về đúng kiểu `dialogue.Keyframe` để dialogue_manager/ambiguity dùng như cũ.

CẦU NỐI 2 SCHEMA (điểm mấu chốt): KISC lọc theo các thuộc tính lifelog rời rạc
(location_type, gender, clothing_color, activity, time_period, location_desc) lưu trong
`Keyframe.attributes`. Hệ thống retrieval của ta lọc bằng metadata `objects` + BM25.
Cầu nối:
  - MÃ HOÁ mỗi thuộc tính thành 1 "tag" dạng "attr:value" và nhét vào `objects` của
    KeyframeRecord -> tái dùng được metadata pre-filter `objects_all` sẵn có.
  - Sinh thêm `llm_caption` tiếng Việt từ thuộc tính -> BM25 khớp được câu người dùng
    (cùng từ vựng mà SlotExtractor nhận diện).
  - Khi trả kết quả, GIẢI MÃ tag ngược lại thành `attributes` để KISC tính entropy /
    hỏi tiếp như bình thường.
"""
from __future__ import annotations

import numpy as np

from ingestion.build_index import KeyframeIndex
from ingestion.embed_clip import CLIP_DIM, deterministic_unit_vector
from ingestion.schemas import KeyframeRecord
from dialogue.dialogue_manager import CANDIDATE_ATTRIBUTES
from dialogue.retriever import HybridRetriever
from dialogue.schemas import Keyframe
from retrieval.coarse_retriever import CoarseRetriever

# Chỉ mã hoá các thuộc tính KISC biết cách hỏi (khớp CANDIDATE_ATTRIBUTES).
KNOWN_ATTRS = list(CANDIDATE_ATTRIBUTES)

# Từ vựng tiếng Việt để sinh caption từ giá trị thuộc tính -> giúp BM25 khớp câu người
# dùng (dùng CHÍNH từ khoá mà dialogue/slot_extractor.py nhận diện).
_VALUE_TO_VI = {
    "location_type": {"outdoor": "ngoài trời", "indoor": "trong nhà"},
    "location_desc": {
        "coffee_shop": "quán cà phê", "restaurant": "nhà hàng", "park": "công viên",
        "home": "ở nhà", "office": "văn phòng", "street": "trên đường",
        "mall": "trung tâm thương mại",
    },
    "gender": {"male": "người đàn ông", "female": "người phụ nữ"},
    "clothing_color": {
        "blue": "áo xanh dương", "red": "áo đỏ", "white": "áo trắng",
        "black": "áo đen", "green": "áo xanh lá", "yellow": "áo vàng",
    },
    "activity": {
        "talking": "nói chuyện", "walking": "đi bộ", "eating": "đang ăn",
        "sitting": "ngồi", "shopping": "mua sắm",
    },
    "time_period": {
        "last_week": "tuần trước", "yesterday": "hôm qua",
        "this_month": "tháng này", "two_weeks_ago": "hai tuần trước",
    },
}


def attribute_tag(attr: str, value: str) -> str:
    """Mã hoá 1 thuộc tính lifelog thành tag đặt trong objects: 'attr:value'."""
    return f"{attr}:{value}"


def attributes_to_tags(attrs: dict) -> list[str]:
    """Các tag cho những thuộc tính ĐÃ BIẾT có giá trị (bỏ None / attr lạ)."""
    return [
        attribute_tag(a, v)
        for a, v in attrs.items()
        if a in KNOWN_ATTRS and v is not None
    ]


def tags_to_attributes(tags: list[str]) -> dict:
    """Giải mã ngược các tag 'attr:value' trong objects thành dict attributes."""
    attrs: dict = {}
    for tag in tags:
        attr, sep, value = tag.partition(":")
        if sep and attr in KNOWN_ATTRS:
            attrs[attr] = value
    return attrs


def attributes_to_caption(attrs: dict) -> str:
    """Sinh caption tiếng Việt từ thuộc tính (cho BM25 khớp câu người dùng)."""
    parts = []
    for attr in KNOWN_ATTRS:
        val = attrs.get(attr)
        if val is not None:
            parts.append(_VALUE_TO_VI.get(attr, {}).get(val, val))
    return " ".join(parts)


def build_lifelog_record(kf_id: str, video_id: str, timestamp: float, attrs: dict) -> KeyframeRecord:
    """Tạo 1 KeyframeRecord mang thuộc tính lifelog (tags trong objects + caption VN).

    clip_embedding sinh tất định để index dựng được; ở đây retrieval KISC dựa vào
    filter + BM25 (không cần vector text encoder), nên embedding chỉ đóng vai trò chỗ
    dựa cho HNSW, không ảnh hưởng logic hội thoại."""
    return KeyframeRecord(
        id=kf_id,
        video_id=video_id,
        timestamp=timestamp,
        clip_embedding=deterministic_unit_vector(kf_id, CLIP_DIM, salt="clip"),
        objects=attributes_to_tags(attrs),
        llm_caption=attributes_to_caption(attrs),
    )


class RealKISCRetriever(HybridRetriever):
    """HybridRetriever THẬT cho dialogue, chạy trên KeyframeIndex/CoarseRetriever.

    Thay thế MockRetriever: dialogue_manager gọi search(query_text, filters, top_k) y
    như cũ, nhưng giờ truy vấn trên index thật.
    """

    def __init__(self, index: KeyframeIndex):
        self.index = index
        self.coarse = CoarseRetriever(index)

    def search(self, query_text: str, filters: dict, top_k: int = 100) -> list[Keyframe]:
        tags = attributes_to_tags(filters or {})
        coarse_filters = {"objects_all": tags} if tags else None

        candidates = self.coarse.search(
            query_text=query_text, filters=coarse_filters, top_k=top_k
        )

        # Nếu BM25 không khớp token nào nhưng filter vẫn có ứng viên -> trả theo filter
        # (điểm = số tag khớp). Đảm bảo KISC không mất ứng viên đúng chỉ vì câu chữ
        # người dùng không trùng caption (bảo toàn recall — Mục 1.2).
        if not candidates and coarse_filters is not None:
            rows = self.coarse._apply_filters(coarse_filters) or []
            return [self._row_to_keyframe(r, float(len(tags))) for r in rows[:top_k]]

        return [self._candidate_to_keyframe(c) for c in candidates]

    def _candidate_to_keyframe(self, cand) -> Keyframe:
        attrs = tags_to_attributes(self.index.objects[cand.row])
        return Keyframe(
            id=cand.keyframe_id, video_id=cand.video_id,
            timestamp=cand.timestamp, attributes=attrs, score=cand.score,
        )

    def _row_to_keyframe(self, row: int, score: float) -> Keyframe:
        attrs = tags_to_attributes(self.index.objects[row])
        return Keyframe(
            id=self.index.ids[row], video_id=self.index.video_ids[row],
            timestamp=self.index.timestamps[row], attributes=attrs, score=score,
        )


# ---------------------------------------------------------------- dataset mẫu
# Không gian thuộc tính khớp dialogue/retriever.py::MockRetriever để tái hiện đúng
# kịch bản case-study (slide 15).
LOCATION_TYPES = ["indoor", "outdoor"]
LOCATION_DESCS = ["coffee_shop", "restaurant", "park", "home", "office", "street", "mall"]
GENDERS = ["male", "female"]
CLOTHING_COLORS = ["blue", "red", "white", "black", "green", "yellow"]
ACTIVITIES = ["talking", "walking", "eating", "sitting", "shopping"]
TIME_PERIODS = ["last_week", "yesterday", "this_month", "two_weeks_ago"]

# Thuộc tính của "đáp án ground-truth" trong kịch bản demo (gặp bạn cũ, quán cà phê
# ngoài trời, đàn ông áo xanh dương, tuần trước).
TARGET_ATTRS = {
    "location_type": "outdoor", "location_desc": "coffee_shop", "gender": "male",
    "clothing_color": "blue", "activity": "talking", "time_period": "last_week",
}


def build_sample_index(num_keyframes: int = 200, seed: int = 42) -> KeyframeIndex:
    """Dựng KeyframeIndex mẫu (đóng vai 'dữ liệu thật') để chạy demo KISC tích hợp.

    Sinh ngẫu nhiên thuộc tính như MockRetriever, và ÉP record đầu tiên là đáp án
    ground-truth khớp kịch bản demo -> hội thoại hội tụ về đúng nó."""
    rng = np.random.default_rng(seed)
    records: list[KeyframeRecord] = []
    for i in range(num_keyframes):
        attrs = {
            "location_type": rng.choice(LOCATION_TYPES),
            "location_desc": rng.choice(LOCATION_DESCS),
            "gender": rng.choice(GENDERS),
            "clothing_color": rng.choice(CLOTHING_COLORS),
            "activity": rng.choice(ACTIVITIES),
            "time_period": rng.choice(TIME_PERIODS),
        }
        records.append(
            build_lifelog_record(f"kf_{i:04d}", f"video_{i // 5:04d}",
                                 float(rng.integers(0, 3600)), attrs)
        )
    # Ép đáp án ground-truth vào record 0 (giống MockRetriever).
    records[0] = build_lifelog_record(records[0].id, records[0].video_id,
                                      records[0].timestamp, dict(TARGET_ATTRS))
    return KeyframeIndex.build(records)

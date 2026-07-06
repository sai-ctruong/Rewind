"""
retriever.py
Interface trừu tượng cho retrieval engine + 1 bản Mock để demo/test offline
(không cần Faiss/CLIP/GPU thật).
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random
from .schemas import Keyframe


class HybridRetriever(ABC):
    """
    Interface cho retrieval engine thật (Faiss/Milvus + CLIP embedding + metadata filter).

    KHI TÍCH HỢP VÀO HỆ THỐNG CHÍNH: implement 1 class con, ví dụ FaissCLIPRetriever,
    trong đó search() sẽ:
      1. Encode query_text bằng CLIP text encoder -> vector
      2. Vector search trên Faiss/Milvus (top-N theo cosine similarity)
      3. Áp filters (location_type, gender, color...) lên metadata/Objects đã detect sẵn
         để loại bớt kết quả không khớp (post-filtering), hoặc pre-filter nếu vector DB hỗ trợ
      4. Trả về List[Keyframe] kèm score
    """

    @abstractmethod
    def search(self, query_text: str, filters: Dict[str, Any], top_k: int = 100) -> List[Keyframe]:
        ...


class MockRetriever(HybridRetriever):
    """
    Retriever giả lập — sinh ngẫu nhiên 1 tập keyframe với attributes mô phỏng
    dữ liệu lifelog (location, gender, clothing color, activity, time_period).
    Dùng để demo/test toàn bộ vòng lặp hội thoại KISC mà KHÔNG cần Faiss/CLIP thật.
    THAY class này bằng retriever thật khi tích hợp vào hệ thống chính.
    """

    LOCATION_TYPES = ["indoor", "outdoor"]
    LOCATION_DESCS = ["coffee_shop", "restaurant", "park", "home", "office", "street", "mall"]
    GENDERS = ["male", "female"]
    CLOTHING_COLORS = ["blue", "red", "white", "black", "green", "yellow"]
    ACTIVITIES = ["talking", "walking", "eating", "sitting", "shopping"]
    TIME_PERIODS = ["last_week", "yesterday", "this_month", "two_weeks_ago"]

    def __init__(self, num_keyframes: int = 200, seed: int = 42):
        random.seed(seed)
        self._all_keyframes = self._generate_dataset(num_keyframes)

    def _generate_dataset(self, n: int) -> List[Keyframe]:
        dataset = []
        for i in range(n):
            attrs = {
                "location_type": random.choice(self.LOCATION_TYPES),
                "location_desc": random.choice(self.LOCATION_DESCS),
                "gender": random.choice(self.GENDERS),
                "clothing_color": random.choice(self.CLOTHING_COLORS),
                "activity": random.choice(self.ACTIVITIES),
                "time_period": random.choice(self.TIME_PERIODS),
            }
            dataset.append(Keyframe(
                id=f"kf_{i:04d}",
                video_id=f"video_{i // 5:04d}",
                timestamp=float(random.randint(0, 3600)),
                attributes=attrs,
            ))
        # Đảm bảo có đúng 1 "đáp án ground-truth" khớp kịch bản demo trong demo.py
        dataset[0].attributes.update({
            "location_type": "outdoor",
            "location_desc": "coffee_shop",
            "gender": "male",
            "clothing_color": "blue",
            "activity": "talking",
            "time_period": "last_week",
        })
        return dataset

    def search(self, query_text: str, filters: Dict[str, Any], top_k: int = 100) -> List[Keyframe]:
        results = []
        for kf in self._all_keyframes:
            match = True
            for key, value in filters.items():
                if value is None:
                    continue
                if kf.attributes.get(key) != value:
                    match = False
                    break
            if match:
                # Score giả lập: khớp càng nhiều filter -> score càng cao + nhiễu nhỏ ngẫu nhiên
                kf.score = len(filters) + random.uniform(0, 0.05)
                results.append(kf)
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]

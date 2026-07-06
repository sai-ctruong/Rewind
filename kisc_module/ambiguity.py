"""
ambiguity.py
Lõi lý thuyết của KISC: đo độ mơ hồ của tập ứng viên và chọn thuộc tính nên hỏi tiếp
theo nguyên lý Information Gain (Shannon Entropy) — cùng nền tảng với thuật toán ID3/C4.5
dùng để xây Decision Tree, hoặc chiến lược tối ưu trong trò chơi Twenty Questions.
"""
import math
from collections import Counter
from typing import List, Optional
from .schemas import Keyframe


def attribute_entropy(candidates: List[Keyframe], attribute: str) -> float:
    """
    H(attribute) = -sum(p_i * log2(p_i)) trên phân bố giá trị của `attribute`
    trong tập `candidates` hiện tại.

    Entropy càng cao => giá trị thuộc tính càng phân tán đều giữa các ứng viên
    => hỏi về thuộc tính này kỳ vọng chia tập ứng viên thành các nhóm cân bằng nhất
    => giảm độ bất định (uncertainty) nhanh nhất cho mỗi câu hỏi.
    Entropy = 0 nghĩa là mọi ứng viên còn lại đều giống nhau ở thuộc tính này
    => hỏi về nó sẽ VÔ ÍCH (không loại được ứng viên nào).
    """
    values = [c.attributes.get(attribute) for c in candidates if c.attributes.get(attribute) is not None]
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for _, cnt in counts.items():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def best_attribute_to_ask(
    candidates: List[Keyframe],
    candidate_attributes: List[str],
    already_asked: List[str],
) -> Optional[str]:
    """
    Chọn attribute CHƯA hỏi có entropy cao nhất trên tập ứng viên hiện tại.
    Trả về None nếu không còn attribute nào đáng hỏi (đã hỏi hết, hoặc mọi
    attribute còn lại đều có entropy = 0 — tức hỏi thêm cũng không giúp ích).
    """
    best_attr = None
    best_entropy = 0.0
    for attr in candidate_attributes:
        if attr in already_asked:
            continue
        e = attribute_entropy(candidates, attr)
        if e > best_entropy:
            best_entropy = e
            best_attr = attr
    return best_attr


def is_confident_enough(
    candidates: List[Keyframe],
    max_candidates: int = 5,
    score_gap_threshold: float = 0.15,
) -> bool:
    """
    Điều kiện DỪNG hội thoại (đủ tự tin để trả kết quả cuối), khi MỘT trong hai đúng:
    1) Số ứng viên còn lại đã đủ nhỏ (<= max_candidates) để người dùng tự chọn bằng mắt, hoặc
    2) Có khoảng cách điểm số (score gap) rõ rệt giữa Top-1 và Top-2
       => một kết quả nổi trội hẳn so với phần còn lại, không cần hỏi thêm.
    """
    if len(candidates) <= max_candidates:
        return True
    if len(candidates) >= 2:
        sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
        gap = sorted_c[0].score - sorted_c[1].score
        if gap >= score_gap_threshold:
            return True
    return False

"""
slot_extractor.py
Trích xuất filters (slot values) từ câu trả lời tự nhiên của người dùng.
"""
from typing import Dict, Any, List


class SlotExtractor:
    """
    Bản demo dùng rule-based/keyword matching (tiếng Việt + tiếng Anh) để chạy
    OFFLINE, không cần gọi API. Trong hệ thống thật NÊN thay bằng LLM-based
    extraction (xem extract_with_llm) vì ngôn ngữ tự nhiên linh hoạt hơn nhiều
    so với keyword matching (ví dụ: "chiếc áo màu của bầu trời" sẽ không khớp
    keyword "xanh dương" nhưng LLM hiểu được).
    """

    KEYWORD_MAP: Dict[str, Dict[str, str]] = {
        "location_type": {
            "ngoài trời": "outdoor", "outdoor": "outdoor",
            "trong nhà": "indoor", "indoor": "indoor",
        },
        "location_desc": {
            "cà phê": "coffee_shop", "coffee": "coffee_shop",
            "nhà hàng": "restaurant", "restaurant": "restaurant",
            "công viên": "park", "park": "park",
            "văn phòng": "office", "office": "office",
            "đường": "street", "street": "street",
            "trung tâm thương mại": "mall", "mall": "mall",
            "nhà": "home", "home": "home",
        },
        "gender": {
            "anh ấy": "male", " nam ": "male", "male": "male",
            "cô ấy": "female", " nữ ": "female", "female": "female",
        },
        "clothing_color": {
            "xanh dương": "blue", "xanh biển": "blue", "blue": "blue",
            "đỏ": "red", "red": "red",
            "trắng": "white", "white": "white",
            "đen": "black", "black": "black",
            "xanh lá": "green", "green": "green",
            "vàng": "yellow", "yellow": "yellow",
        },
        "activity": {
            "nói chuyện": "talking", "trò chuyện": "talking", "talking": "talking",
            "đi bộ": "walking", "walking": "walking",
            "ăn": "eating", "eating": "eating",
            "ngồi": "sitting", "sitting": "sitting",
            "mua sắm": "shopping", "shopping": "shopping",
        },
        "time_period": {
            "tuần trước": "last_week", "last week": "last_week",
            "hôm qua": "yesterday", "yesterday": "yesterday",
            "hai tuần trước": "two_weeks_ago",
            "tháng này": "this_month", "this month": "this_month",
        },
    }

    def extract(self, text: str) -> Dict[str, Any]:
        """Trả về dict {attribute: value} trích được từ `text`. Attribute không
        xuất hiện trong text sẽ không có mặt trong dict trả về (không phải None)."""
        text_lower = f" {text.lower()} "
        extracted: Dict[str, Any] = {}
        for attribute, keyword_dict in self.KEYWORD_MAP.items():
            for keyword, value in keyword_dict.items():
                if keyword in text_lower:
                    extracted[attribute] = value
                    break  # lấy giá trị khớp đầu tiên cho mỗi attribute
        return extracted

    def extract_with_llm(self, text: str, known_attributes: List[str]) -> Dict[str, Any]:
        """
        ĐIỂM TÍCH HỢP CHO PRODUCTION.
        Gọi Claude API (hoặc LLM khác) với prompt yêu cầu trả JSON thuần các slot
        trích xuất được. Ví dụ prompt:

            "Trích xuất các thông tin sau từ câu trả lời của người dùng.
             Chỉ trả về JSON hợp lệ, không kèm giải thích.
             Các trường: location_type (indoor/outdoor/null), location_desc,
             gender (male/female/null), clothing_color, activity, time_period.
             Nếu thông tin không xuất hiện, để giá trị null.
             Câu trả lời của người dùng: '{text}'"

        Đây là stub — implement bằng anthropic client thật khi tích hợp.
        """
        raise NotImplementedError(
            "Implement bằng cách gọi Claude API (hoặc LLM khác) tại đây khi tích hợp "
            "vào hệ thống chính, để xử lý ngôn ngữ tự nhiên linh hoạt hơn keyword matching."
        )

"""
question_generator.py
Sinh câu hỏi làm rõ (clarifying question) dựa trên attribute được chọn bởi
ambiguity.best_attribute_to_ask().
"""
from typing import List
from .schemas import Keyframe

QUESTION_TEMPLATES = {
    "location_type": "Khoảnh khắc đó diễn ra trong nhà hay ngoài trời?",
    "location_desc": "Bạn có nhớ địa điểm cụ thể là ở đâu không (quán cà phê, công viên, nhà hàng...)?",
    "gender": "Người đó là nam hay nữ?",
    "clothing_color": "Người đó mặc trang phục màu gì?",
    "activity": "Lúc đó mọi người đang làm gì (nói chuyện, ăn uống, đi bộ...)?",
    "time_period": "Sự việc đó xảy ra khi nào (tuần trước, hôm qua, tháng này...)?",
}


class QuestionGenerator:
    """
    Bản demo dùng template cố định (tra cứu theo attribute). Trong hệ thống
    thật, nên thay bằng LLM để câu hỏi tự nhiên hơn và có thể tham chiếu ngữ
    cảnh các câu trả lời trước đó (xem generate_with_llm).
    """

    def generate(self, attribute: str, candidates: List[Keyframe]) -> str:
        return QUESTION_TEMPLATES.get(
            attribute, f"Bạn có thể cho biết thêm thông tin về '{attribute}' không?"
        )

    def generate_with_llm(
        self, attribute: str, candidates: List[Keyframe], dialogue_history: List[str]
    ) -> str:
        """
        ĐIỂM TÍCH HỢP CHO PRODUCTION.
        Gọi LLM để sinh câu hỏi tự nhiên hơn, có thể tham chiếu lại các câu trả
        lời trước đó của người dùng (giống case study "Lượt 2" trong slide AIC 2026:
        hệ thống không hỏi cộc lốc mà đặt câu hỏi liền mạch với ngữ cảnh hội thoại).
        """
        raise NotImplementedError(
            "Implement bằng cách gọi Claude API thật ở đây khi tích hợp vào hệ thống chính."
        )

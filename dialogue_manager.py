"""
dialogue_manager.py
Bộ điều phối chính cho bài toán Conversational KIS (KISC) — ghép các module
schemas / ambiguity / retriever / slot_extractor / question_generator thành
1 vòng lặp hội thoại hoàn chỉnh.
"""
from typing import List, Optional
from .schemas import DialogueState, DialogueTurn, Keyframe
from .retriever import HybridRetriever
from .slot_extractor import SlotExtractor
from .question_generator import QuestionGenerator
from . import ambiguity


# Danh sách attribute mà hệ thống BIẾT CÁCH hỏi (khớp với slot_extractor + question_generator).
# Khi mở rộng hệ thống thật, thêm attribute mới vào đây + cả 2 module trên.
CANDIDATE_ATTRIBUTES = [
    "time_period", "location_type", "location_desc",
    "gender", "clothing_color", "activity",
]


class KISCDialogueManager:
    """
    Vòng lặp hội thoại (Dialogue State Tracking + Information Gain):

    1. Nhận input (câu hỏi ban đầu hoặc câu trả lời tiếp theo của người dùng)
    2. Trích xuất slot/filter mới từ input           -> SlotExtractor
    3. Cộng dồn vào DialogueState.filters             -> "bộ nhớ" hội thoại
    4. Truy vấn lại candidate set với filters hiện tại -> HybridRetriever
    5. Kiểm tra độ tự tin                              -> ambiguity.is_confident_enough
       - Đủ tự tin HOẶC hết lượt tối đa -> kết thúc, trả kết quả cuối
       - Chưa đủ tự tin -> chọn attribute entropy cao nhất để hỏi tiếp
         -> ambiguity.best_attribute_to_ask + QuestionGenerator
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        slot_extractor: Optional[SlotExtractor] = None,
        question_generator: Optional[QuestionGenerator] = None,
        max_turns: int = 5,
        max_candidates_to_stop: int = 5,
    ):
        self.retriever = retriever
        self.slot_extractor = slot_extractor or SlotExtractor()
        self.question_generator = question_generator or QuestionGenerator()
        self.max_turns = max_turns
        self.max_candidates_to_stop = max_candidates_to_stop
        self.state = DialogueState()

    def start(self, initial_query: str) -> str:
        """Bắt đầu 1 phiên hội thoại mới với truy vấn đầu tiên."""
        self.state = DialogueState()
        return self._process_user_input(initial_query)

    def respond(self, user_answer: str) -> str:
        """Xử lý câu trả lời tiếp theo của người dùng trong phiên đang diễn ra."""
        if self.state.finished:
            return "Hội thoại đã kết thúc. Gọi start() để bắt đầu truy vấn mới."
        return self._process_user_input(user_answer)

    def _process_user_input(self, text: str) -> str:
        turn_index = len(self.state.turns)
        new_filters = self.slot_extractor.extract(text)
        self.state.filters.update({k: v for k, v in new_filters.items() if v is not None})
        self.state.turns.append(
            DialogueTurn(turn_index=turn_index, speaker="user", text=text, extracted_filters=new_filters)
        )

        candidates = self.retriever.search(query_text=text, filters=self.state.filters, top_k=200)
        self.state.candidates = candidates

        if len(candidates) == 0:
            response = (
                "Không tìm thấy khoảnh khắc nào khớp với tất cả mô tả hiện có. "
                "Bạn có thể mô tả lại chi tiết vừa cung cấp không?"
            )
            self.state.turns.append(DialogueTurn(turn_index + 1, "system", response))
            return response

        if ambiguity.is_confident_enough(candidates, self.max_candidates_to_stop) or \
                turn_index >= self.max_turns:
            self.state.finished = True
            response = self._format_final_answer(candidates)
            self.state.turns.append(DialogueTurn(turn_index + 1, "system", response))
            return response

        next_attribute = ambiguity.best_attribute_to_ask(
            candidates, CANDIDATE_ATTRIBUTES, self.state.asked_attributes
        )
        if next_attribute is None:
            self.state.finished = True
            response = self._format_final_answer(candidates)
        else:
            self.state.asked_attributes.append(next_attribute)
            response = self.question_generator.generate(next_attribute, candidates)

        self.state.turns.append(DialogueTurn(turn_index + 1, "system", response))
        return response

    def _format_final_answer(self, candidates: List[Keyframe]) -> str:
        top = sorted(candidates, key=lambda c: c.score, reverse=True)[: self.max_candidates_to_stop]
        lines = [f"Đã thu hẹp còn {len(candidates)} ứng viên. Top kết quả:"]
        for c in top:
            lines.append(f"  - {c.id} (video {c.video_id}, t={c.timestamp:.0f}s, score={c.score:.2f})")
        return "\n".join(lines)

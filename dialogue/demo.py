"""
demo.py
Mô phỏng lại đúng kịch bản Case Study KISC trong slide tập huấn AIC 2026
(slide 15: "Tìm giúp tôi đoạn video tôi gặp một người bạn cũ vào tuần trước").
Chạy: python -m dialogue.demo   (từ thư mục cha chứa dialogue/)
"""
from dialogue.retriever import MockRetriever
from dialogue.dialogue_manager import KISCDialogueManager


def run_demo():
    retriever = MockRetriever(num_keyframes=200)
    manager = KISCDialogueManager(retriever, max_turns=5, max_candidates_to_stop=5)

    print("=== Demo hội thoại KISC (Conversational KIS) - AIC 2026 ===\n")

    turn1 = "Tìm giúp tôi đoạn video tôi gặp một người bạn cũ vào tuần trước."
    print(f"[User]   {turn1}")
    response = manager.start(turn1)
    print(f"[System] {response}")
    print(f"         (Số ứng viên còn lại: {len(manager.state.candidates)})\n")

    if manager.state.finished:
        return

    turn2 = "Chúng tôi gặp nhau ở một quán cà phê ngoài trời, anh ấy mặc áo sơ mi màu xanh dương."
    print(f"[User]   {turn2}")
    response = manager.respond(turn2)
    print(f"[System] {response}")
    print(f"         (Số ứng viên còn lại: {len(manager.state.candidates)})\n")

    if manager.state.finished:
        return

    turn3 = "Lúc đó chúng tôi đang ngồi nói chuyện."
    print(f"[User]   {turn3}")
    response = manager.respond(turn3)
    print(f"[System] {response}")


if __name__ == "__main__":
    run_demo()

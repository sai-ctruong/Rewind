"""Demo KISC chạy trên RETRIEVER THẬT (CLAUDE.md Phase 8).

Song song với kisc_module/demo.py nhưng thay MockRetriever bằng RealKISCRetriever
(chạy trên KeyframeIndex/CoarseRetriever thật). Đặt ở retrieval/ để KHÔNG phải sửa
kisc_module/demo.py (tôn trọng Mục 10.6). Cùng kịch bản case-study slide 15 -> kiểm
chứng vòng hội thoại KISC hội tụ đúng trên hạ tầng retrieval thật.

Chạy: python -m retrieval.kisc_real_demo
"""
from __future__ import annotations

from kisc_module.dialogue_manager import KISCDialogueManager
from retrieval.kisc_adapter import RealKISCRetriever, build_sample_index


def run_real_demo() -> KISCDialogueManager:
    index = build_sample_index(num_keyframes=200)
    retriever = RealKISCRetriever(index)
    manager = KISCDialogueManager(retriever, max_turns=5, max_candidates_to_stop=5)

    print("=== Demo KISC trên RETRIEVER THẬT (KeyframeIndex + CoarseRetriever) ===\n")

    turn1 = "Tìm giúp tôi đoạn video tôi gặp một người bạn cũ vào tuần trước."
    print(f"[User]   {turn1}")
    response = manager.start(turn1)
    print(f"[System] {response}")
    print(f"         (Số ứng viên còn lại: {len(manager.state.candidates)})\n")

    if not manager.state.finished:
        turn2 = (
            "Chúng tôi gặp nhau ở một quán cà phê ngoài trời, "
            "anh ấy mặc áo sơ mi màu xanh dương."
        )
        print(f"[User]   {turn2}")
        response = manager.respond(turn2)
        print(f"[System] {response}")
        print(f"         (Số ứng viên còn lại: {len(manager.state.candidates)})\n")

    if not manager.state.finished:
        turn3 = "Lúc đó chúng tôi đang ngồi nói chuyện."
        print(f"[User]   {turn3}")
        response = manager.respond(turn3)
        print(f"[System] {response}")

    return manager


if __name__ == "__main__":
    run_real_demo()

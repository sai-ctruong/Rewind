"""Unit test cho retrieval/image_filter — bộ lọc ẢNH hội thoại (thu hẹp dần).

Dùng video mock 3 cảnh (đỏ/lá/dương) + ColorMockEncoder như các test khác: màu = ngữ
nghĩa, nên kiểm được cả LOGIC THU HẸP lẫn ĐỘ ĐÚNG (thêm mô tả "đỏ" -> giữ cảnh đỏ).
Không cần GPU/API.
"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from retrieval.image_filter import ImageFilterSession  # noqa: E402
from retrieval.video_engine import VideoSearchEngine  # noqa: E402
from tests.test_video_engine import ColorMockEncoder, _make_video  # noqa: E402


@pytest.fixture()
def engine_entry(tmp_path):
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return engine, entry


def _sess(engine, entry, **kw):
    # min_k=1 để thấy rõ pool co lại trên dataset nhỏ (3 keyframe).
    kw.setdefault("min_k", 1)
    kw.setdefault("start_k", 20)
    return ImageFilterSession(engine, entry, **kw)


# --------------------------------- start -------------------------------------
def test_start_returns_image_pool(engine_entry) -> None:
    engine, entry = engine_entry
    s = _sess(engine, entry)
    r = s.start("cảnh")
    assert r.count >= 2 and r.count == len(r.results)
    # mỗi kết quả có đủ field để UI vẽ ảnh (id + vị trí)
    for it in r.results:
        assert it["id"] and "timestamp" in it and "video_id" in it
    assert r.turn == 1 and r.count_before == 0


def test_start_requires_query(engine_entry) -> None:
    engine, entry = engine_entry
    with pytest.raises(ValueError):
        _sess(engine, entry).start("   ")


def test_refine_before_start_raises(engine_entry) -> None:
    engine, entry = engine_entry
    with pytest.raises(RuntimeError):
        _sess(engine, entry).refine(text="đỏ")


# --------------------------------- thu hẹp ------------------------------------
def test_pool_shrinks_each_turn(engine_entry) -> None:
    engine, entry = engine_entry
    s = _sess(engine, entry)
    r1 = s.start("cảnh")
    r2 = s.refine(text="đỏ")
    assert r2.count < r1.count            # LƯỚI ẢNH CO LẠI — điểm cốt lõi
    assert r2.count_before == r1.count    # UI hiện được "3 → 2"
    if r2.count > 1:
        r3 = s.refine()
        assert r3.count < r2.count        # tiếp tục co


def test_adding_text_narrows_to_matching_color(engine_entry) -> None:
    engine, entry = engine_entry
    s = _sess(engine, entry)
    s.start("cảnh")
    r = s.refine(text="đỏ")
    assert r.results[0]["timestamp"] < 1.0     # cảnh đỏ ~[0,1)s lên đầu
    assert "đỏ" in r.query and "cảnh" in r.query  # truy vấn TÍCH LUỸ


def test_results_stay_within_pool(engine_entry) -> None:
    engine, entry = engine_entry
    s = _sess(engine, entry)
    r1 = s.start("cảnh")
    pool1 = {it["id"] for it in r1.results}
    r2 = s.refine(text="xanh dương")
    assert {it["id"] for it in r2.results} <= pool1   # chỉ lọc trong pool, không nhảy ra


def test_positive_feedback_pulls_toward_liked_image(engine_entry) -> None:
    engine, entry = engine_entry
    blue = engine.search(entry, "xanh dương", top_k=1)[0].keyframe_id
    s = _sess(engine, entry)
    s.start("cảnh")
    r = s.refine(positive=[blue])
    assert blue in [it["id"] for it in r.results]
    assert r.results[0]["id"] == blue                # ảnh 👍 lên đầu
    assert blue in s.memory.positive_ids             # nhớ xuyên lượt (G3)


def test_pick_marks_others_negative(engine_entry) -> None:
    engine, entry = engine_entry
    red = engine.search(entry, "đỏ", top_k=1)[0].keyframe_id
    blue = engine.search(entry, "xanh dương", top_k=1)[0].keyframe_id
    s = _sess(engine, entry)
    s.start("cảnh")
    r = s.refine(pick=red, others=[red, blue])
    assert red in s.memory.positive_ids
    assert blue in s.memory.negative_ids     # ảnh KHÔNG chọn -> tín hiệu tương phản
    assert r.results[0]["id"] == red


def test_combines_text_and_feedback_same_turn(engine_entry) -> None:
    engine, entry = engine_entry
    red = engine.search(entry, "đỏ", top_k=1)[0].keyframe_id
    s = _sess(engine, entry)
    r1 = s.start("cảnh")
    r2 = s.refine(text="đỏ", positive=[red])   # cả 2 tín hiệu cùng lượt
    assert r2.count < r1.count
    assert r2.results[0]["id"] == red


# --------------------------------- hỏi lại / reset ----------------------------
def test_finished_when_single_candidate(engine_entry) -> None:
    engine, entry = engine_entry
    s = _sess(engine, entry)
    s.start("cảnh")
    r = s.refine(k=1)
    assert r.count == 1 and r.finished is True
    assert r.question is None            # còn 1 ảnh -> không hỏi nữa


def test_reset_clears_session(engine_entry) -> None:
    engine, entry = engine_entry
    s = _sess(engine, entry)
    s.start("cảnh")
    s.refine(text="đỏ")
    s.reset()
    assert s.pool == [] and s.query == "" and not s.memory.has_feedback()
    r = s.start("cảnh")                  # lọc lại được từ đầu
    assert r.count >= 2

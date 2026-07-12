"""Unit test cho retrieval/video_engine — engine tìm kiếm video độ chính xác cao.

Không tải model thật (nặng): bơm MOCK ENCODER đọc MÀU TRUNG BÌNH của ảnh keyframe
làm "ngữ nghĩa" (embed) và map từ khoá màu trong câu (encode_text). Nhờ đó kiểm được
TRỌN pipeline: cắt keyframe -> embed 2 encoder -> dedup ngữ nghĩa -> index 2 slot ->
query prompt ensemble -> RRF fuse -> top-k, với ngữ nghĩa thật (màu khớp màu).

Cần OpenCV để dựng video tổng hợp (importorskip như test_video_ingest).
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ingestion.schemas import RawKeyframe, load_cv2_image  # noqa: E402
from retrieval.video_engine import VideoSearchEngine  # noqa: E402


# ------------------------- video + mock encoder helpers -----------------------
def _make_video(path, colors, frames_per_color=8, fps=10, size=64):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    assert vw.isOpened()
    for color in colors:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = color  # BGR
        for _ in range(frames_per_color):
            vw.write(frame)
    vw.release()


class ColorMockEncoder:
    """Encoder giả: embedding = hướng màu trung bình của ảnh (unit vector 3D + pad).

    encode_text: bắt từ khoá màu trong câu -> cùng không gian với ảnh. `salt` xoay
    nhẹ không gian để 2 instance mô phỏng 2 encoder KHÁC nhau (ensemble thật sự).
    """

    KEYWORDS = {"đỏ": (0, 0, 255), "red": (0, 0, 255),
                "xanh lá": (0, 255, 0), "green": (0, 255, 0),
                "xanh dương": (255, 0, 0), "blue": (255, 0, 0)}

    def __init__(self, dim: int = 8, salt: float = 0.0):
        self.dim = dim
        self.salt = salt
        self.text_calls: list[str] = []

    def _vec(self, bgr) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[:3] = np.asarray(bgr, dtype=np.float32)
        v[3] = self.salt  # dịch không gian theo encoder
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed(self, raw: RawKeyframe) -> np.ndarray:
        img = load_cv2_image(raw)  # ảnh trong RAM (mặc định) hoặc trên đĩa
        return self._vec(img.reshape(-1, 3).mean(axis=0))

    def encode_text(self, text: str) -> np.ndarray:
        self.text_calls.append(text)
        low = text.lower()
        for kw, bgr in sorted(self.KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in low:
                return self._vec(bgr)
        return self._vec((1, 1, 1))  # không nhận ra màu -> hướng trung tính


@pytest.fixture()
def engine_and_entry(tmp_path):
    video = tmp_path / "scenes.mp4"
    # 3 cảnh: đỏ, xanh lá, xanh dương — mỗi cảnh 10 frame tĩnh (nhiều bản sao).
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")
    return engine, entry


# ---------------------------------- tests -------------------------------------
def test_index_dedups_static_scenes(engine_and_entry) -> None:
    _, entry = engine_and_entry
    # Cảnh tĩnh -> nhiều frame lấy mẫu nhưng dedup ngữ nghĩa gộp về ~1 đại diện/cảnh.
    assert entry.num_sampled >= entry.num_indexed
    assert 3 <= entry.num_indexed <= 5


def test_search_correct_scene_per_color(engine_and_entry) -> None:
    engine, entry = engine_and_entry
    # Cảnh theo thời gian: đỏ ~[0,1)s, xanh lá ~[1,2)s, xanh dương ~[2,3)s.
    for query, (lo, hi) in [("màu đỏ", (0.0, 1.0)), ("green", (1.0, 2.0)),
                             ("xanh dương", (2.0, 3.0))]:
        results = engine.search(entry, query, top_k=3)
        assert results, query
        assert lo <= results[0].timestamp < hi, (
            f"query {query!r}: top-1 t={results[0].timestamp} ngoài cảnh [{lo},{hi})"
        )


def test_ensemble_uses_both_encoders(engine_and_entry) -> None:
    engine, entry = engine_and_entry
    results = engine.search(entry, "red", top_k=3)
    # RRF phải nhận ranked list từ CẢ 2 slot (clip = encoder 1, siglip = encoder 2).
    assert set(results[0].source_ranks.keys()) == {"clip", "siglip"}


def test_query_prompt_ensemble_calls_all_templates(engine_and_entry) -> None:
    engine, entry = engine_and_entry
    enc1 = engine._encoders[0]
    before = len(enc1.text_calls)
    engine.encode_query("màu đỏ")
    called = enc1.text_calls[before:]
    # Mỗi template 1 lần gọi, câu gốc nằm trong biến thể đầu.
    assert len(called) == len(engine.query_templates)
    assert any("màu đỏ" in c for c in called)


def test_encode_query_returns_unit_vectors(engine_and_entry) -> None:
    engine, _ = engine_and_entry
    for v in engine.encode_query("blue"):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_index_dataset_searches_across_videos(tmp_path) -> None:
    # 2 video khác nhau: video A có cảnh đỏ + xanh lá; video B có cảnh xanh dương.
    va = tmp_path / "vidA.mp4"; vb = tmp_path / "vidB.mp4"
    _make_video(va, [(0, 0, 255), (0, 255, 0)], frames_per_color=10)
    _make_video(vb, [(255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])

    entry = engine.index_dataset([va, vb], tmp_path / "frames")
    assert entry.video_id == "__dataset__"
    # Keyframe từ CẢ 2 video đều có trong raws (id toàn cục '{video}/{n}').
    vids = {r.video_id for r in entry.raws.values()}
    assert vids == {"vidA", "vidB"}

    # Query "xanh dương" -> phải trả keyframe từ VIDEO B (chỉ B có cảnh xanh dương).
    res = engine.search(entry, "xanh dương", top_k=3)
    assert res[0].video_id == "vidB"
    # Query "đỏ" -> keyframe từ video A.
    res2 = engine.search(entry, "màu đỏ", top_k=3)
    assert res2[0].video_id == "vidA"


class SignMockOcr:
    """OCR giả: trả token DUY NHẤT theo màu cảnh (không phải tên màu) -> kiểm BM25
    trên OCR text hoạt động (dense encoder KHÔNG biết token này)."""

    def extract(self, raw: RawKeyframe):
        b, g, r = load_cv2_image(raw).reshape(-1, 3).mean(axis=0)  # BGR
        if r > 150 and g < 100 and b < 100:
            return "bienbao SPECIALREDSIGN"
        if g > 150 and r < 100:
            return "bienbao SPECIALGREENSIGN"
        if b > 150 and r < 100:
            return "bienbao SPECIALBLUESIGN"
        return None


def test_ocr_text_search_finds_sign(tmp_path) -> None:
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    engine.set_ocr(SignMockOcr())            # bật OCR (mock)
    entry = engine.index_video(video, tmp_path / "frames")

    # Tìm bằng CHỮ trên biển — token chỉ có trong OCR, dense không biết.
    res = engine.search(entry, "SPECIALGREENSIGN", top_k=3)
    assert res, "phải có kết quả từ BM25 trên OCR"
    assert 1.0 <= res[0].timestamp < 2.0     # cảnh xanh lá ~[1,2)s
    assert "bm25" in res[0].source_ranks     # kết quả đến từ BM25 (OCR text)


class BatchMockEncoder(ColorMockEncoder):
    """Mock có `embed_batch` -> kiểm engine đi đường LÔ (A1), và lô == lẻ về kết quả."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def embed_batch(self, raws, batch_size=256):
        self.batch_calls += 1
        self.batch_sizes.append(len(raws))
        return [self.embed(r) for r in raws]   # cùng nội dung với đường lẻ


def test_embed_batch_path_used_and_consistent(tmp_path) -> None:
    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    enc0, enc1 = BatchMockEncoder(salt=0.0), BatchMockEncoder(salt=0.3)
    engine.set_encoders([enc0, enc1])
    entry = engine.index_video(video, tmp_path / "frames")
    # Engine phải gọi embed_batch (1 lần/encoder cho toàn bộ raws), KHÔNG lặp embed lẻ.
    assert enc0.batch_calls == 1 and enc1.batch_calls == 1
    assert enc0.batch_sizes[0] == entry.num_sampled   # cả loạt raws vào 1 lô
    # Kết quả search vẫn đúng cảnh -> embed lô không làm sai thứ tự/nội dung.
    res = engine.search(entry, "màu đỏ", top_k=3)
    assert res[0].timestamp < 1.0


def test_save_load_roundtrip(tmp_path) -> None:
    """A2: lưu index ra đĩa rồi nạp lại -> search vẫn đúng, KHÔNG cần embed lại.
    Ảnh không lưu theo index (cố ý) mà decode lại từ video gốc qua source_video."""
    from retrieval.video_engine import VideoIndexEntry

    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    entry = engine.index_video(video, tmp_path / "frames")

    save_dir = tmp_path / "idx"
    entry.save(save_dir)
    loaded = VideoIndexEntry.load(save_dir)
    assert loaded.num_indexed == entry.num_indexed
    assert loaded.video_id == entry.video_id

    # raws sau nạp: KHÔNG còn image_bytes (nhẹ đĩa), nhưng giữ source_video để dựng lại.
    r = next(iter(loaded.raws.values()))
    assert r.image_bytes is None and r.source_video is not None
    img = load_cv2_image(r)                    # decode lại frame từ video gốc
    assert img is not None and img.ndim == 3

    # Search trên index NẠP TỪ ĐĨA vẫn trả đúng cảnh -> embedding được bảo toàn.
    res = engine.search(loaded, "màu đỏ", top_k=3)
    assert res[0].timestamp < 1.0


def test_search_temporal_respects_order(engine_and_entry) -> None:
    """B4: 'cảnh A TRƯỚC cảnh B' — chỉ giữ chuỗi cùng video, timestamp tăng dần.
    Cảnh: đỏ ~[0,1)s, xanh lá ~[1,2)s, xanh dương ~[2,3)s."""
    engine, entry = engine_and_entry

    # Đúng thứ tự: đỏ trước xanh dương -> có chuỗi, timestamp tăng dần.
    m = engine.search_temporal(entry, ["màu đỏ", "xanh dương"], per_event_k=1)
    assert m, "phải có tổ hợp đỏ->xanh dương"
    assert m[0].timestamps == sorted(m[0].timestamps)
    assert m[0].timestamps[0] < 1.0 and m[0].timestamps[1] >= 2.0

    # Ngược thứ tự: xanh dương TRƯỚC đỏ -> KHÔNG tồn tại (xanh dương luôn sau đỏ).
    assert engine.search_temporal(entry, ["xanh dương", "màu đỏ"], per_event_k=1) == []

    # Ba cảnh đúng thứ tự -> chuỗi 3 mắt xích tăng dần.
    m3 = engine.search_temporal(entry, ["màu đỏ", "xanh lá", "xanh dương"], per_event_k=1)
    assert m3 and len(m3[0].steps) == 3
    assert m3[0].timestamps == sorted(m3[0].timestamps)

    # < 2 cảnh -> lỗi rõ ràng.
    with pytest.raises(ValueError):
        engine.search_temporal(entry, ["màu đỏ"])


def test_parallel_index_matches_sequential(tmp_path) -> None:
    """A4: pipeline song song (decode‖embed) cho KẾT QUẢ GIỐNG đường tuần tự —
    cùng số keyframe, cùng thứ tự id, cùng kết quả search."""
    va = tmp_path / "a.mp4"; vb = tmp_path / "b.mp4"
    _make_video(va, [(0, 0, 255), (0, 255, 0)], frames_per_color=10)
    _make_video(vb, [(255, 0, 0)], frames_per_color=10)

    def build(parallel):
        eng = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False,
                                embed_batch_size=3, parallel_index=parallel)
        eng.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
        return eng, eng.index_dataset([va, vb], tmp_path / "f")

    eng_p, entry_p = build(True)
    _, entry_s = build(False)
    # Cùng tập keyframe id, cùng số lượng (index song song không làm mất/nhân đôi).
    assert entry_p.num_indexed == entry_s.num_indexed
    assert sorted(entry_p.raws) == sorted(entry_s.raws)
    # Search vẫn định tuyến đúng video theo màu.
    assert eng_p.search(entry_p, "xanh dương", top_k=3)[0].video_id == "b"
    assert eng_p.search(entry_p, "màu đỏ", top_k=3)[0].video_id == "a"


def test_asr_text_search_finds_spoken_word(tmp_path) -> None:
    """B3: câu NÓI (ASR) đưa vào BM25 -> tìm được cảnh theo lời nói, dù dense/OCR
    không biết token đó. Token độc nhất chỉ nằm trong transcript đoạn xanh lá."""
    from ingestion.ocr_asr_extract import MockVideoAsrEngine

    video = tmp_path / "scenes.mp4"
    _make_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0)], frames_per_color=10)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=50, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder(salt=0.0), ColorMockEncoder(salt=0.3)])
    engine.set_asr(MockVideoAsrEngine(canned={"scenes": [
        {"start": 0.0, "end": 1.0, "text": "xin chào mọi người"},
        {"start": 1.0, "end": 2.0, "text": "KEODUYNHAT đang được nói ra"},
        {"start": 2.0, "end": 3.0, "text": "tạm biệt nhé"},
    ]}))
    entry = engine.index_video(video, tmp_path / "f")

    # ASR đã điền vào entry, và có token độc nhất.
    assert any("KEODUYNHAT" in t for t in entry.asr_by_id.values())
    # Tìm bằng token CHỈ có trong lời nói -> BM25 kéo đúng cảnh xanh lá (~[1,2)s).
    res = engine.search(entry, "KEODUYNHAT", top_k=3)
    assert res and 1.0 <= res[0].timestamp < 2.0
    assert "bm25" in res[0].source_ranks


def test_single_encoder_mode(tmp_path) -> None:
    video = tmp_path / "v.mp4"
    _make_video(video, [(0, 0, 255), (255, 0, 0)], frames_per_color=8)
    engine = VideoSearchEngine(sample_every_s=0.2, max_frames=30, enable_ocr=False)
    engine.set_encoders([ColorMockEncoder()])   # chỉ 1 encoder
    entry = engine.index_video(video, tmp_path / "f")
    results = engine.search(entry, "red", top_k=2)
    assert results and results[0].timestamp < 1.0
    assert set(results[0].source_ranks.keys()) == {"clip"}  # không có slot siglip

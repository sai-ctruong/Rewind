"""Unit test Phase 2 — pipeline ingestion (mock-first). DoD: ra KeyframeRecord đầy
đủ field (kể cả llm_caption) cho một 'video mẫu' chạy hoàn toàn offline.

Vì hiện CHƯA có data/API thật (theo quyết định mock-first), ta chứng minh DoD trên
video mẫu tổng hợp qua các provider MOCK, và kiểm rằng khi cắm bản THẬT thì các guard
lazy-import báo lỗi rõ ràng (không vỡ ngầm). Toàn bộ chạy bằng numpy, không GPU/API.
"""
from __future__ import annotations

import numpy as np
import pytest

from ingestion.build_records import (
    IngestionPipeline,
    make_sample_video,
    searchable_text,
)
from ingestion.dedup import deduplicate_keyframes
from ingestion.embed_clip import (
    CLIP_DIM,
    MockClipEmbeddingProvider,
    NpyClipEmbeddingProvider,
    deterministic_unit_vector,
)
from ingestion.embed_siglip import SIGLIP_DIM, MockSiglipEmbeddingProvider
from ingestion.llm_captioning import ClaudeCaptioner, MockCaptioner
from ingestion.ocr_asr_extract import MockAsrEngine, MockOcrEngine
from ingestion.schemas import KeyframeRecord, RawKeyframe


# -----------------------------------------------------------------------------
# Embedding mock: tất định, đúng chiều, chuẩn hoá, và độc lập giữa CLIP/SigLIP
# -----------------------------------------------------------------------------
def test_deterministic_unit_vector_is_stable_and_normalized() -> None:
    v1 = deterministic_unit_vector("kf_1", 128, salt="clip")
    v2 = deterministic_unit_vector("kf_1", 128, salt="clip")
    assert np.allclose(v1, v2)                         # cùng key -> cùng vector
    assert v1.shape == (128,)
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-6)


def test_clip_and_siglip_spaces_differ() -> None:
    raw = RawKeyframe(id="kf_1", video_id="v", timestamp=0.0)
    c = MockClipEmbeddingProvider().embed(raw)
    s = MockSiglipEmbeddingProvider().embed(raw)
    assert c.shape == (CLIP_DIM,)
    assert s.shape == (SIGLIP_DIM,)
    # Salt khác nhau -> 2 encoder mock độc lập (tinh thần ensemble Mục 2.1).
    # (chiều khác nhau nên so trực tiếp không được; kiểm bằng seed khác -> nội dung khác)
    c2 = MockClipEmbeddingProvider(dim=SIGLIP_DIM).embed(raw)
    assert not np.allclose(c2, s)


def test_clip_mock_deterministic_across_instances() -> None:
    raw = RawKeyframe(id="abc", video_id="v", timestamp=1.0)
    a = MockClipEmbeddingProvider().embed(raw)
    b = MockClipEmbeddingProvider().embed(raw)
    assert np.allclose(a, b)


# -----------------------------------------------------------------------------
# DoD Phase 2: KeyframeRecord đầy đủ field qua pipeline mock
# -----------------------------------------------------------------------------
def test_pipeline_builds_full_records() -> None:
    raws = make_sample_video(num_frames=6, scene_size=3)
    pipeline = IngestionPipeline()  # toàn bộ mock mặc định
    records = pipeline.build(raws)

    assert len(records) == 6
    for rec in records:
        assert isinstance(rec, KeyframeRecord)
        # Các field bắt buộc phải được điền:
        assert rec.id and rec.video_id
        assert rec.clip_embedding is not None and rec.clip_embedding.shape == (CLIP_DIM,)
        assert rec.siglip_embedding is not None and rec.siglip_embedding.shape == (
            SIGLIP_DIM,
        )
        assert rec.objects, "objects phải có (BTC cấp) cho video mẫu này"
        # DoD nhấn mạnh llm_caption phải khác None:
        assert rec.llm_caption is not None and rec.llm_caption.strip() != ""
        assert rec.video_id in rec.llm_caption  # caption mock có nội dung liên quan


def test_pipeline_populates_ocr_asr_when_available() -> None:
    """OCR/ASR mặc định None, nhưng khi có tín hiệu (canned) thì field được điền."""
    raws = make_sample_video(num_frames=3, scene_size=3)
    target = raws[1].id
    pipeline = IngestionPipeline(
        ocr=MockOcrEngine(canned={target: "BIEN HIEU CAFE"}),
        asr=MockAsrEngine(canned={target: "xin chao cac ban"}),
    )
    records = {r.id: r for r in pipeline.build(raws)}
    assert records[target].ocr_text == "BIEN HIEU CAFE"
    assert records[target].asr_text == "xin chao cac ban"
    # Frame khác không có canned -> None (đúng thực tế đa số frame không có chữ).
    assert records[raws[0].id].ocr_text is None


def test_searchable_text_includes_caption_and_objects() -> None:
    raw = RawKeyframe(id="v/0", video_id="v", timestamp=0.0, objects=["person", "flower"])
    pipeline = IngestionPipeline(
        ocr=MockOcrEngine(canned={"v/0": "SHOP HOA"}),
        asr=MockAsrEngine(canned={"v/0": "mua hoa"}),
    )
    rec = pipeline.build_one(raw)
    text = searchable_text(rec)
    assert "person" in text and "flower" in text   # objects
    assert "SHOP HOA" in text                        # OCR
    assert "mua hoa" in text                         # ASR
    assert "mock" in text.lower()                     # llm_caption (mock) có mặt


# -----------------------------------------------------------------------------
# Tích hợp với dedup Phase 1: record đã làm giàu vẫn dedup được, không merge nhầm
# -----------------------------------------------------------------------------
def test_pipeline_then_dedup_keeps_enriched_fields() -> None:
    raws = make_sample_video(num_frames=6, scene_size=3)
    records = IngestionPipeline().build(raws)
    result = deduplicate_keyframes(records, similarity_threshold=0.97)
    # Embedding mock độc lập theo id (gần trực giao) -> KHÔNG bị gộp nhầm.
    assert result.num_output == 6
    # Đại diện vẫn giữ nguyên các field đã làm giàu.
    for rep in result.representatives:
        assert rep.llm_caption is not None
        assert rep.siglip_embedding is not None


# -----------------------------------------------------------------------------
# NpyClipEmbeddingProvider (bản thật nạp từ đĩa)
# -----------------------------------------------------------------------------
def test_npy_clip_loader_roundtrip(tmp_path) -> None:
    video_id = "vidX"
    ids = [f"{video_id}/0", f"{video_id}/1", f"{video_id}/2"]
    matrix = np.arange(3 * CLIP_DIM, dtype=np.float32).reshape(3, CLIP_DIM)
    np.save(tmp_path / f"{video_id}.npy", matrix)
    (tmp_path / f"{video_id}.txt").write_text("\n".join(ids), encoding="utf-8")

    loader = NpyClipEmbeddingProvider(tmp_path)
    raw = RawKeyframe(id=f"{video_id}/1", video_id=video_id, timestamp=0.0)
    got = loader.embed(raw)
    assert np.allclose(got, matrix[1])


def test_npy_clip_loader_missing_file_raises(tmp_path) -> None:
    loader = NpyClipEmbeddingProvider(tmp_path)
    raw = RawKeyframe(id="v/0", video_id="nope", timestamp=0.0)
    with pytest.raises(FileNotFoundError, match="CLIP feature"):
        loader.embed(raw)


# -----------------------------------------------------------------------------
# Guard lazy-import cho bản thật (chưa cài deps / chưa có key) -> lỗi rõ ràng
# -----------------------------------------------------------------------------
def test_claude_captioner_without_dep_or_key_raises() -> None:
    """anthropic chưa cài -> ImportError với hướng dẫn; không vỡ ngầm."""
    with pytest.raises((ImportError, RuntimeError)):
        ClaudeCaptioner()


def test_siglip_encoder_without_torch_raises() -> None:
    """Guard chỉ áp dụng khi CHƯA cài torch. Nếu torch đã có (bật tính năng video
    thật), bỏ qua vì không thể kích hoạt nhánh ImportError mà không tải model nặng."""
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch đã được cài — guard lazy-import không áp dụng")
    from ingestion.embed_siglip import SiglipEncoder

    with pytest.raises(ImportError, match="torch"):
        SiglipEncoder()


def test_mock_captioner_handles_no_objects() -> None:
    raw = RawKeyframe(id="v/0", video_id="v", timestamp=5.0, objects=[])
    cap = MockCaptioner().caption(raw)
    assert cap is not None and cap.strip() != ""  # vẫn có caption hợp lệ

"""Phase 0 smoke test — xác nhận Definition of Done của Phase 0.

Blueprint (CLAUDE.md Mục 8) đặt DoD cho Phase 0 là:
    "Chạy `pytest` trống không lỗi, cấu trúc thư mục đúng Mục 6".

Thay vì để pytest chạy rỗng (không assert gì), ta viết một smoke test tối thiểu
nhưng CÓ Ý NGHĨA để bắt lỗi cấu hình sớm:
  1. Cấu trúc thư mục Mục 6 tồn tại đầy đủ.
  2. configs/settings.yaml parse được và chứa đúng các ngưỡng mặc định Phase 0
     (dedup=0.97, coarse top-K=1000, rerank top-K=100, time_budget=20s).
  3. Package dialogue đã dời vào subfolder vẫn import được nguyên vẹn
     (không vỡ do việc di dời — Mục 10.6 cấm refactor dialogue).

Test chạy hoàn toàn offline, không cần GPU/API (Mục 1.5).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Project root = thư mục cha của tests/ (file này nằm ở tests/).
ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# 1. Cấu trúc thư mục theo Mục 6
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_dir",
    ["dialogue", "ingestion", "retrieval", "evaluation", "ui", "tests", "configs"],
)
def test_required_directories_exist(rel_dir: str) -> None:
    """Mỗi thư mục bắt buộc trong Mục 6 phải tồn tại."""
    path = ROOT / rel_dir
    assert path.is_dir(), f"Thiếu thư mục bắt buộc theo Mục 6: {rel_dir}/"


def test_settings_file_exists() -> None:
    assert (ROOT / "configs" / "settings.yaml").is_file(), (
        "Thiếu configs/settings.yaml (DoD Phase 0)"
    )


# -----------------------------------------------------------------------------
# 2. settings.yaml — parse được và đúng ngưỡng mặc định Phase 0
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def settings() -> dict:
    with (ROOT / "configs" / "settings.yaml").open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "settings.yaml phải parse ra một mapping/dict"
    return data


def test_default_dedup_threshold(settings: dict) -> None:
    # Mục 8 Phase 0: dedup threshold = 0.97
    assert settings["dedup"]["similarity_threshold"] == pytest.approx(0.97)


def test_default_coarse_top_k(settings: dict) -> None:
    # Mục 8 Phase 0: coarse top-K = 1000
    assert settings["coarse"]["top_k"] == 1000


def test_default_rerank_top_k(settings: dict) -> None:
    # Mục 8 Phase 0: rerank top-K = 100
    assert settings["fine_rerank"]["top_k"] == 100


def test_default_time_budget(settings: dict) -> None:
    # Mục 8 Phase 0: time_budget = 20s
    assert settings["runtime"]["time_budget_seconds"] == 20


def test_rrf_k_is_standard_default(settings: dict) -> None:
    # Mục 4.3: hằng số RRF chuẩn k=60 (không benchmark, [FIXED]).
    assert settings["fusion"]["rrf_k"] == 60


def test_brute_force_limit_matches_constraint(settings: dict) -> None:
    # Mục 1.3: cấm brute-force khi vượt ~50.000 vector.
    assert settings["index"]["brute_force_max_vectors"] == 50000


# -----------------------------------------------------------------------------
# 3. dialogue vẫn import được sau khi dời vào subfolder
# -----------------------------------------------------------------------------
def test_dialogue_imports_after_move() -> None:
    """Việc di dời KISC vào dialogue/ không được làm vỡ import công khai."""
    from dialogue import (  # noqa: F401  (import để kiểm tra, không dùng trực tiếp)
        HybridRetriever,
        KISCDialogueManager,
        MockRetriever,
    )

    # MockRetriever phải khởi tạo được offline (Mục 1.5) — không cần API/GPU.
    retriever = MockRetriever()
    assert retriever is not None

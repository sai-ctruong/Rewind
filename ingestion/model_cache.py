"""Trỏ cache model HuggingFace sang ổ D cho RIÊNG dự án này (không đụng máy chung).

VÌ SAO: `transformers` tải weights (hàng GB) vào HF_HOME — mặc định
`C:\\Users\\<user>\\.cache\\huggingface` (ổ C) — làm phồng ổ C. venv chỉ chứa CODE,
không chứa weights. Module này đặt HF_HOME sang ổ D để model của dự án nằm ở D.

CƠ CHẾ: chỉ cần import module này TRƯỚC khi `transformers`/`huggingface_hub` được
import lần đầu là đủ (chúng đọc HF_HOME lúc import). Các module nạp model
(embed_siglip, vlm_rerank) và entry point (ui/app.py, các demo) đều import module này
ở đầu file.

GHI ĐÈ: nếu người dùng đã tự đặt HF_HOME (biến môi trường) thì TÔN TRỌNG, không đổi.
Đổi thư mục đích qua biến môi trường AIC_HF_HOME.
"""
from __future__ import annotations

import os
from pathlib import Path

# Thư mục cache model trên ổ D (đổi qua env AIC_HF_HOME nếu muốn nơi khác).
DEFAULT_HF_CACHE = os.environ.get("AIC_HF_HOME", r"D:\hf_cache")


def setup_hf_cache(path: str | None = None) -> str | None:
    """Đặt HF_HOME sang ổ D nếu người dùng chưa tự đặt. Trả về đường dẫn cache đang dùng.

    An toàn: nếu ổ D không tồn tại / không tạo được thư mục -> bỏ qua (giữ mặc định C),
    KHÔNG làm vỡ import.
    """
    if os.environ.get("HF_HOME"):
        return os.environ["HF_HOME"]  # tôn trọng lựa chọn của người dùng
    target = path or DEFAULT_HF_CACHE
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # ổ D không sẵn -> dùng mặc định (C)
    os.environ["HF_HOME"] = target
    return target


# Đặt ngay khi import (side effect có chủ đích).
setup_hf_cache()

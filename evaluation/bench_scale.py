"""Benchmark QUY MÔ LỚN cho tầng ANN — trả lời "hệ có scale được không?" bằng SỐ ĐO.

VÌ SAO CÓ FILE NÀY: blueprint tuyên bố "hàng trăm giờ video → hàng triệu keyframe, dùng
Faiss HNSW", nhưng mọi benchmark trước đây chỉ chạy trên video ngắn (~50 keyframe). HNSW
ở quy mô 50 vector là VÔ NGHĨA (nó hành xử như brute-force) — nên tuyên bố "scale-ready"
thực chất CHƯA có bằng chứng nào. File này đo trực tiếp:

  1. RAM  theo số vector  -> HNSW có trụ nổi 1 triệu vector không, hay BẮT BUỘC IVF-PQ?
  2. Thời gian build      -> index 100 giờ video mất bao lâu?
  3. Latency vs efSearch  -> Mục 11.1.2 YÊU CẦU vẽ đường cong rồi chọn "điểm khuỷu tay",
                             không đoán mò (efSearch đang là [PROVISIONAL] trong settings).
  4. Recall@K vs efSearch -> tăng tốc có làm MẤT ứng viên đúng không (ràng buộc Mục 1.2)?

CÁCH ĐO CHO TRUNG THỰC:
  - Vector tổng hợp đúng số chiều THẬT (768), hai phân bố:
      * `clustered` (mặc định) — quanh 200 tâm, MÔ PHỎNG embedding thật vốn gom cụm theo
        ngữ nghĩa. Đây là dữ liệu dùng để CHỌN efSearch.
      * `random` — ngẫu nhiên đều: ở 768 chiều mọi vector gần như trực giao nhau nên HNSW
        không có cấu trúc để bám. Đây là ca BỆNH LÝ, recall tụt thảm; giữ lại để biết sàn
        tuyệt đối, KHÔNG dùng chọn tham số.
    (RAM/build/latency gần như không phụ thuộc phân bố -> tin được ở cả hai.)
  - Ground-truth = tích vô hướng ĐẦY ĐỦ bằng numpy (exact), không phải một ANN khác.
  - RAM đo bằng RSS tiến trình (psutil) TRƯỚC/SAU khi build -> tách riêng chi phí của
    index, không lẫn với ma trận gốc.
  - Chỉ đo tới cỡ vừa RAM máy rồi NGOẠI SUY tuyến tính (bytes/vector là hằng số với HNSW)
    — thà ngoại suy từ số đo thật còn hơn thử 1M rồi crash và không có dữ liệu gì.

Chạy:  python -m evaluation.bench_scale --sizes 10000,50000,100000 --dim 768
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Mặc định khớp IndexConfig (ingestion/build_index.py) — chính các giá trị [PROVISIONAL]
# đang cần được thay bằng số đo.
DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_VALUES = (16, 32, 64, 128, 256)

_OUT = Path(__file__).resolve().parent / "benchmarks" / "scale_bench.json"


@dataclass
class EfResult:
    """Một điểm trên đường cong recall/latency."""

    ef_search: int
    recall_at_k: float           # tỉ lệ khớp ĐÚNG ID với top-k exact
    score_ratio: float           # điểm TB thu được / điểm TB exact (1.0 = tốt ngang exact)
    latency_ms_p50: float
    latency_ms_p95: float


@dataclass
class SizeResult:
    """Kết quả cho một cỡ dataset."""

    n_vectors: int
    dim: int
    build_seconds: float
    index_ram_mb: float          # RAM riêng của index (không tính ma trận gốc)
    matrix_ram_mb: float         # ma trận float32 gốc
    bytes_per_vector: float      # (index) — dùng để ngoại suy
    dist: str = "clustered"      # phân bố dữ liệu đã dùng để đo
    ef_curve: list[EfResult] = field(default_factory=list)


def rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """n vector đơn vị NGẪU NHIÊN ĐỀU (float32) — ca xấu nhất, bệnh lý cho ANN.

    Ở 768 chiều, các vector ngẫu nhiên gần như TRỰC GIAO đôi một (inner product ≈ 0 ±
    1/√768): "top-10 gần nhất" gần như không khác gì phần còn lại, nên HNSW không có
    cấu trúc nào để bám mà đi. Recall đo trên dữ liệu này là cận dưới CỰC BI QUAN —
    KHÔNG nên dùng nó để chọn efSearch. Dùng `clustered_vectors` cho việc đó."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim), dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def clustered_vectors(n: int, dim: int, seed: int = 0, per_cluster: int = 500,
                      spread: float = 0.35) -> np.ndarray:
    """n vector đơn vị nằm quanh `n_clusters` tâm — MÔ PHỎNG embedding THẬT.

    VÌ SAO CẦN: embedding ảnh thật KHÔNG rải đều — chúng gom cụm theo ngữ nghĩa (cảnh
    phố, bãi biển, trong nhà…), và chính cấu trúc cụm đó là thứ HNSW dựa vào để định
    hướng. Đo recall trên dữ liệu ngẫu nhiên đều rồi kết luận "HNSW kém" là oan cho nó
    và sẽ dẫn tới chọn sai tham số.

    `spread` = ĐỘ DÀI nhiễu so với tâm (tâm có độ dài 1); nhỏ = cụm chặt.
    LƯU Ý (bẫy nhiều chiều): nhiễu `σ·N(0,I)` ở dim chiều có độ dài ≈ σ·√dim — nên PHẢI
    chia cho √dim. Quên bước này thì ở 768 chiều nhiễu dài ~9.7 sẽ át tâm (dài 1) và dữ
    liệu "có cụm" thực chất lại NGẪU NHIÊN ĐỀU — đúng lỗi này đã xảy ra khi viết hàm.

    `per_cluster` = số điểm MỖI cụm, GIỮ CỐ ĐỊNH khi N tăng (số cụm mới tăng theo N).
    VÌ SAO: nếu cố định số cụm, N lớn làm mỗi cụm dày đặc điểm gần trùng -> "top-10 đúng"
    chỉ hơn hàng trăm điểm khác ở số lẻ, và recall khớp-ID tụt vì LÝ DO ĐO ĐẠC chứ không
    phải vì HNSW kém. Thêm dữ liệu thật (sau dedup) cũng sinh ra CẢNH MỚI (cụm mới) chứ
    không chỉ nhồi thêm frame trùng vào cảnh cũ.
    """
    rng = np.random.default_rng(seed)
    n_clusters = max(10, n // max(per_cluster, 1))
    centers = rng.standard_normal((n_clusters, dim), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    which = rng.integers(0, n_clusters, size=n)
    sigma = spread / np.sqrt(dim)          # -> độ dài nhiễu ≈ spread, độc lập số chiều
    m = centers[which] + sigma * rng.standard_normal((n, dim), dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m.astype(np.float32)


def make_vectors(n: int, dim: int, seed: int = 0, dist: str = "clustered") -> np.ndarray:
    if dist == "random":
        return unit_vectors(n, dim, seed=seed)
    if dist == "clustered":
        return clustered_vectors(n, dim, seed=seed)
    raise ValueError(f"dist không hợp lệ: {dist!r} (dùng 'clustered'|'random').")


def exact_topk(mat: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground-truth: top-k theo inner product, tính ĐẦY ĐỦ (chia lô để khỏi ngốn RAM)."""
    out = np.empty((len(queries), k), dtype=np.int64)
    step = max(1, 2_000_000 // max(len(mat), 1))     # giới hạn ~2M ô mỗi lô
    for i in range(0, len(queries), step):
        block = queries[i:i + step] @ mat.T          # (b, n)
        idx = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(len(block))[:, None]
        order = np.argsort(-block[rows, idx], axis=1)
        out[i:i + step] = idx[rows, order]
    return out


def bench_size(n: int, dim: int, queries: np.ndarray, k: int, m: int,
               ef_construction: int, ef_values: tuple[int, ...],
               seed: int = 0, dist: str = "clustered") -> SizeResult:
    import faiss

    mat = make_vectors(n, dim, seed=seed, dist=dist)
    gt = exact_topk(mat, queries, k)
    # Điểm exact của top-k -> chuẩn để tính score_ratio. recall khớp-ID một mình DỄ GÂY
    # HIỂU LẦM khi nhiều ứng viên gần bằng điểm nhau: trả về hàng xóm "khác ID nhưng tốt
    # ngang" bị tính là SAI, dù với người dùng thì không khác gì.
    gt_scores = np.take_along_axis(queries @ mat.T, gt, axis=1)   # (q, k)

    gc.collect()
    before = rss_mb()
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(mat)
    build_s = time.perf_counter() - t0
    gc.collect()
    index_ram = max(rss_mb() - before, 0.0)

    res = SizeResult(
        n_vectors=n, dim=dim, dist=dist, build_seconds=round(build_s, 2),
        index_ram_mb=round(index_ram, 1),
        matrix_ram_mb=round(mat.nbytes / 2**20, 1),
        bytes_per_vector=round(index_ram * 2**20 / n, 1),
    )

    for ef in ef_values:
        index.hnsw.efSearch = ef
        lat: list[float] = []
        hits = 0
        got_sum = 0.0
        for qi in range(len(queries)):
            q = queries[qi:qi + 1]
            t = time.perf_counter()
            scores, rows = index.search(q, k)
            lat.append((time.perf_counter() - t) * 1000.0)
            hits += len(set(rows[0].tolist()) & set(gt[qi].tolist()))
            got_sum += float(scores[0][rows[0] != -1].sum())
        lat.sort()
        res.ef_curve.append(EfResult(
            ef_search=ef,
            recall_at_k=round(hits / (len(queries) * k), 4),
            score_ratio=round(got_sum / float(gt_scores.sum()), 4),
            latency_ms_p50=round(lat[len(lat) // 2], 3),
            latency_ms_p95=round(lat[int(len(lat) * 0.95)], 3),
        ))

    del index, mat, gt, gt_scores
    gc.collect()
    return res


def extrapolate(results: list[SizeResult], target: int) -> dict:
    """Ngoại suy RAM + thời gian build cho `target` vector từ các cỡ đã ĐO thật.

    RAM của HNSW tuyến tính theo N (mỗi vector: dữ liệu dim*4B + link 2*M*4B) nên ngoại
    suy tuyến tính là hợp lý. Build time siêu tuyến tính nhẹ (~N log N) -> ước lượng này
    là CẬN DƯỚI, ghi rõ để không ai tưởng là con số chắc chắn."""
    if not results:
        return {}
    big = results[-1]
    per_vec_mb = big.index_ram_mb / big.n_vectors
    mat_mb = target * big.dim * 4 / 2**20
    idx_mb = per_vec_mb * target
    return {
        "target_vectors": target,
        "index_ram_gb": round(idx_mb / 1024, 2),
        "matrix_ram_gb": round(mat_mb / 1024, 2),
        "total_ram_gb": round((idx_mb + mat_mb) / 1024, 2),
        "build_hours_lower_bound": round(
            big.build_seconds * (target / big.n_vectors) / 3600, 2),
        "note": ("Ngoại suy tuyến tính từ cỡ lớn nhất ĐO ĐƯỢC. Build time thực tế cao hơn "
                 "(HNSW ~N log N). total_ram = index + ma trận gốc, vì KeyframeIndex GIỮ "
                 "CẢ HAI (ma trận dùng cho exact rerank trên subset)."),
    }


def main(sizes: tuple[int, ...], dim: int, n_queries: int, k: int, m: int,
         ef_construction: int, ef_values: tuple[int, ...],
         out: Optional[Path] = None, dist: str = "clustered") -> dict:
    import psutil

    # Truy vấn lấy TỪ CÙNG phân bố với dữ liệu (khác seed) — giống thực tế: người dùng
    # tìm thứ NẰM TRONG phân bố của kho, không phải một điểm ngẫu nhiên ngoài vũ trụ.
    queries = make_vectors(n_queries, dim, seed=999, dist=dist)
    vm = psutil.virtual_memory()
    print(f"[scale] RAM máy: {vm.total/2**30:.1f} GB (trống {vm.available/2**30:.1f} GB) · "
          f"dim={dim} · M={m} · efConstruction={ef_construction} · dữ liệu={dist}")

    results: list[SizeResult] = []
    for n in sizes:
        need_gb = (n * dim * 4 * 2) / 2**30           # ma trận + bản sao trong index
        avail_gb = psutil.virtual_memory().available / 2**30
        if need_gb > avail_gb * 0.8:
            print(f"[scale] BỎ QUA n={n:,} — cần ~{need_gb:.1f} GB, chỉ còn "
                  f"{avail_gb:.1f} GB. (Chính đây là bằng chứng cho giới hạn RAM.)")
            continue
        print(f"[scale] đang đo n={n:,} …", flush=True)
        r = bench_size(n, dim, queries, k, m, ef_construction, ef_values, dist=dist)
        best = max(r.ef_curve, key=lambda e: e.recall_at_k)
        print(f"        build {r.build_seconds:.1f}s · index {r.index_ram_mb:.0f} MB "
              f"({r.bytes_per_vector:.0f} B/vec) · recall@{k} tốt nhất {best.recall_at_k:.3f} "
              f"@ ef={best.ef_search} ({best.latency_ms_p50:.2f} ms)")
        results.append(r)

    payload = {
        "config": {"dim": dim, "k": k, "n_queries": n_queries, "hnsw_m": m,
                   "ef_construction": ef_construction, "ef_values": list(ef_values),
                   "dist": dist, "ram_total_gb": round(vm.total / 2**30, 1)},
        "sizes": [asdict(r) for r in results],
        "extrapolation_1m": extrapolate(results, 1_000_000),
        "caveat": (
            "dist=clustered mô phỏng embedding thật (gom cụm ngữ nghĩa) -> recall dùng "
            "được để chọn efSearch. dist=random là ca BỆNH LÝ ở 768 chiều (mọi vector "
            "gần như trực giao, không có cấu trúc để HNSW bám) -> recall cực bi quan, "
            "KHÔNG dùng để chọn tham số. RAM/build/latency thì không phụ thuộc phân bố "
            "nên tin được ở cả hai."),
    }
    out = out or _OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[scale] đã lưu: {out}")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark quy mô ANN (HNSW): RAM/build/latency/recall.")
    p.add_argument("--sizes", default="10000,50000,100000",
                   help="Các cỡ dataset, cách nhau bởi dấu phẩy.")
    p.add_argument("--dim", type=int, default=768, help="Số chiều embedding (thật: 768).")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=DEFAULT_M)
    p.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    p.add_argument("--ef", default=",".join(str(x) for x in DEFAULT_EF_VALUES))
    p.add_argument("--dist", default="clustered", choices=("clustered", "random"),
                   help="clustered = giống embedding thật (mặc định); random = ca xấu nhất.")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    main(
        sizes=tuple(int(x) for x in a.sizes.split(",") if x.strip()),
        dim=a.dim, n_queries=a.queries, k=a.k, m=a.m,
        ef_construction=a.ef_construction,
        ef_values=tuple(int(x) for x in a.ef.split(",") if x.strip()),
        out=Path(a.out) if a.out else None, dist=a.dist,
    )

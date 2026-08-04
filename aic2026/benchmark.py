"""Reproducible benchmark and ablation logging for AIC 2026."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, Sequence


@dataclass
class QueryLog:
    task: str
    query_id: str
    query: str
    latency_ms: float
    rows: list[list[str]]
    meta: dict = field(default_factory=dict)


def environment_snapshot(*, encoder: dict | None = None, dataset: dict | None = None, seed: int = 42, query_count: int = 0, config_hash: str | None = None) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    packages = {}
    for name in ("numpy", "faiss-cpu", "rank-bm25", "flask", "torch", "transformers", "opencv-python-headless"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "git_commit": commit,
        "python": sys.version,
        "os": platform.platform(),
        "encoder": encoder,
        "device": None if encoder is None else encoder.get("device"),
        "gpu": gpu,
        "dependencies": packages,
        "dataset": dataset or {},
        "number_of_queries": query_count,
        "seed": seed,
        "config_hash": config_hash,
    }


class BenchmarkLogger:
    def __init__(self, out_dir: str | Path = "evaluation/benchmarks/aic2026"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _jsonl(path: Path, rows: Sequence[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_run(
        self,
        name: str,
        config: dict,
        logs: Sequence[QueryLog],
        summary: dict | None = None,
        *,
        predictions: Sequence[dict] = (),
        errors: Sequence[dict] = (),
        environment: dict | None = None,
        config_hash: str | None = None,
    ) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.out_dir / f"{stamp}-{name}"
        suffix = 1
        while run_dir.exists():
            run_dir = self.out_dir / f"{stamp}-{name}-{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        if config_hash is not None:
            (run_dir / "config_hash.txt").write_text(str(config_hash), encoding="utf-8")
        self._jsonl(run_dir / "queries.jsonl", [asdict(log) for log in logs])
        self._jsonl(run_dir / "predictions.jsonl", list(predictions))
        self._jsonl(run_dir / "errors.jsonl", list(errors))
        env = environment or environment_snapshot(query_count=len(logs), config_hash=config_hash)
        (run_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary or {}, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_dir


def time_call(fn: Callable, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, (time.perf_counter() - start) * 1000.0
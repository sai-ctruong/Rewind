from __future__ import annotations

import time
import tracemalloc


def measure(fn, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {"latency_ms": latency_ms, "peak_ram_bytes": peak}

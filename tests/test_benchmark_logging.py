import json

from aic2026.benchmark import BenchmarkLogger, QueryLog


def test_benchmark_logger_writes_complete_artifact_contract(tmp_path) -> None:
    run = BenchmarkLogger(tmp_path).write_run(
        "test", {"seed": 42}, [QueryLog("kis", "1", "query", 1.2, [["V", "1"]])],
        {"R@1": 1.0}, predictions=[{"query_id": "1"}], errors=[], environment={"seed": 42},
    )
    expected = {"config.json", "environment.json", "queries.jsonl", "predictions.jsonl", "summary.json", "errors.jsonl"}
    assert expected <= {path.name for path in run.iterdir()}
    assert json.loads((run / "environment.json").read_text()) == {"seed": 42}

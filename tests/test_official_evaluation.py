from evaluation.official_eval import evaluate_labels


def test_official_evaluation_writes_end_to_end_report(tmp_path) -> None:
    labels = [{"query_id": "q1", "query": "x", "task": "kis", "video_id": "V", "start": 10, "end": 20}]
    summary, run_dir = evaluate_labels(labels, lambda row: [["V", "15"]], out_dir=tmp_path)
    assert summary["Final Score"] == 1.0
    for name in ("config.json", "environment.json", "queries.jsonl", "predictions.jsonl", "summary.json", "errors.jsonl"):
        assert (run_dir / name).exists()

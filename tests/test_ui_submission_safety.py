"""End-to-end submission safety over HTTP: preflight, export, staleness, edits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ui.app as appmod
from aic2026.config import app_config_from_dict
from aic2026.submission_validation import read_submission_csv

FPS = 10.0
FRAME_IDS = (5, 15, 25, 35)
VIDEOS = ("L21_V001", "L21_V002")


def make_root(root: Path, *, extra_video: str | None = None) -> Path:
    videos = list(VIDEOS) + ([extra_video] if extra_video else [])
    (root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    (root / "clip-features-32").mkdir(parents=True, exist_ok=True)
    for position, video_id in enumerate(videos):
        rows = ["n,pts_time,fps,frame_idx"]
        for ordinal, frame_idx in enumerate(FRAME_IDS, start=1):
            rows.append(f"{ordinal},{frame_idx / FPS},{FPS},{frame_idx}")
        (root / "map-keyframes" / f"{video_id}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        angle = 0.2 + 0.7 * position
        np.save(
            root / "clip-features-32" / f"{video_id}.npy",
            np.array(
                [[np.cos(angle + 0.01 * i), np.sin(angle + 0.01 * i)] for i in range(len(FRAME_IDS))],
                dtype=np.float32,
            ),
        )
        folder = root / "keyframes" / video_id
        folder.mkdir(parents=True, exist_ok=True)
        for ordinal in range(1, len(FRAME_IDS) + 1):
            Image.new("RGB", (16, 16), (30 * position, 10, 5)).save(folder / f"{ordinal:03d}.jpg")
    return root


def make_config(root: Path, cache_dir: Path, frame_cache_dir: Path):
    return app_config_from_dict(
        {
            "aic2026": {
                "dataset": {
                    "root": str(root),
                    "cache_dir": str(cache_dir),
                    "frame_cache_dir": str(frame_cache_dir),
                    "validation": {"expected_feature_dim": 2},
                },
                "encoder": {"feature_dim": 2},
                "ranking": {"min_frame_gap": 0},
                "refinement": {"mode": "disabled"},
                "qa": {"backend": {"type": "mock"}, "top_video_hypotheses": 2},
                "trake": {"min_gap_s": 0.0, "per_event_top_k": 8, "top_video_hypotheses": 4},
            }
        }
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(appmod, "SUBMISSION_DIR", tmp_path / "submissions")
    root_a = make_root(tmp_path / "root_a")
    root_b = make_root(tmp_path / "root_b", extra_video="L21_V003")
    config = make_config(root_a, tmp_path / "cache_a", tmp_path / "frames")
    app = appmod.create_app(app_config=config)
    app.testing = True
    http = app.test_client()
    assert http.post("/api/video/index_folder", json={"path": str(root_a)}).status_code == 200
    return http, root_a, root_b, tmp_path / "submissions"


def kis_batch(http):
    body = http.post("/api/video/search", json={"query": "a person", "topk": 10}).get_json()
    return body, body["result_batch"]["result_id"]


# ------------------------------------------------------------------ preflight


def test_a_search_registers_an_editable_result_batch(client) -> None:
    http, *_ = client
    body, result_id = kis_batch(http)
    batch = body["result_batch"]
    assert batch["task"] == "kis"
    assert batch["runtime_generation"] == body["generation"]
    assert batch["row_count"] == len(body["results"])
    assert all(row["row_id"] for row in batch["rows"])
    assert batch["manual_edit_count"] == 0


def test_preflight_reports_a_valid_batch(client) -> None:
    http, *_ = client
    _, result_id = kis_batch(http)
    payload = http.post("/api/submission/preflight", json={"result_id": result_id}).get_json()
    assert payload["valid"] is True
    assert payload["task"] == "kis"
    assert payload["row_count"] >= 1
    assert payload["errors"] == []
    # "Valid" is about FORMAT and says so.
    assert "format" in payload["note"].lower()


def test_preflight_reports_an_invalid_batch_without_an_error_status(client) -> None:
    http, *_ = client
    body, result_id = kis_batch(http)
    # Q&A on this fixture abstains (mock backend, no captions), so it is not exportable.
    qa = http.post(
        "/api/video/vqa", json={"question": "What colour?", "event": "a person"}
    ).get_json()
    payload = http.post(
        "/api/submission/preflight", json={"result_id": qa["result_batch"]["result_id"]}
    )
    assert payload.status_code == 200, "an invalid batch is a normal answer, not an error"
    data = payload.get_json()
    assert data["valid"] is False
    assert data["errors"]


def test_preflight_on_an_unknown_batch_is_404(client) -> None:
    http, *_ = client
    response = http.post("/api/submission/preflight", json={"result_id": "rb_missing"})
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "UNKNOWN_RESULT_BATCH"


# --------------------------------------------------------------------- export


def test_export_writes_the_csv_and_its_sidecar(client) -> None:
    http, _, _, submissions = client
    _, result_id = kis_batch(http)
    response = http.post("/api/submission/save", json={"result_id": result_id, "name": "kis"})
    assert response.status_code == 200
    payload = response.get_json()
    csv_path = Path(payload["path"])
    report_path = Path(payload["report_path"])
    assert csv_path.is_file() and report_path.is_file()
    rows = read_submission_csv(csv_path)
    assert rows and all(len(row) == 2 for row in rows)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["task"] == "kis"
    assert report["runtime_generation"] == http.get("/api/health").get_json()["runtime"]["generation"]
    assert report["result_id"] == result_id


def test_export_refuses_a_non_submittable_qa_batch(client) -> None:
    http, *_ = client
    qa = http.post(
        "/api/video/vqa", json={"question": "What colour?", "event": "a person"}
    ).get_json()
    response = http.post(
        "/api/submission/save",
        json={"result_id": qa["result_batch"]["result_id"], "name": "qa"},
    )
    assert response.status_code == 422
    payload = response.get_json()
    assert payload["validation"]["valid"] is False
    # No file was produced for a refused export.
    assert not payload.get("path")


def test_a_deliberate_manual_answer_makes_a_qa_batch_exportable(client) -> None:
    http, _, _, submissions = client
    qa = http.post(
        "/api/video/vqa", json={"question": "What colour?", "event": "a person"}
    ).get_json()
    result_id = qa["result_batch"]["result_id"]
    rows = qa["result_batch"]["rows"]
    for row in rows:
        edit = http.post(
            f"/api/results/{result_id}/edit",
            json={"row_id": row["row_id"], "field": "answer", "value": "đỏ"},
        )
        assert edit.status_code == 200
    saved = http.post("/api/submission/save", json={"result_id": result_id, "name": "qa"})
    assert saved.status_code == 200
    exported = read_submission_csv(Path(saved.get_json()["path"]))
    assert all(row[2] == "đỏ" for row in exported)


def test_an_invalid_submission_name_is_rejected(client) -> None:
    http, *_ = client
    _, result_id = kis_batch(http)
    for name in ("../escape", "a/b", "with space", "..\\win"):
        response = http.post(
            "/api/submission/save", json={"result_id": result_id, "name": name}
        )
        assert response.status_code == 400
        assert response.get_json()["error_code"] == "INVALID_SUBMISSION_NAME"


def test_export_of_an_unknown_batch_is_404(client) -> None:
    http, *_ = client
    response = http.post("/api/submission/save", json={"result_id": "rb_nope"})
    assert response.status_code == 404


# ---------------------------------------------------------------- generation


def test_a_data_root_switch_makes_an_old_batch_unexportable(client) -> None:
    http, _, root_b, _ = client
    _, result_id = kis_batch(http)
    assert http.post("/api/submission/preflight", json={"result_id": result_id}).get_json()["valid"]

    assert http.post("/api/video/index_folder", json={"path": str(root_b)}).status_code == 200

    stale = http.post("/api/submission/save", json={"result_id": result_id, "name": "kis"})
    assert stale.status_code == 409
    assert stale.get_json()["error_code"] == "STALE_RESULT_GENERATION"
    check = http.post("/api/submission/preflight", json={"result_id": result_id}).get_json()
    assert check["valid"] is False
    assert [issue["code"] for issue in check["errors"]] == ["STALE_RESULT_GENERATION"]


def test_a_batch_from_the_new_generation_exports_normally(client) -> None:
    http, _, root_b, _ = client
    assert http.post("/api/video/index_folder", json={"path": str(root_b)}).status_code == 200
    _, result_id = kis_batch(http)
    assert http.post(
        "/api/submission/save", json={"result_id": result_id, "name": "kis"}
    ).status_code == 200


def test_the_batch_endpoint_reports_staleness(client) -> None:
    http, _, root_b, _ = client
    _, result_id = kis_batch(http)
    before = http.get(f"/api/results/{result_id}").get_json()
    assert before["stale"] is False
    assert http.post("/api/video/index_folder", json={"path": str(root_b)}).status_code == 200
    payload = http.get(f"/api/results/{result_id}").get_json()
    assert payload["stale"] is True
    assert payload["active_generation"] == before["active_generation"] + 1


# -------------------------------------------------------------------- edits


def test_editing_one_row_over_http_leaves_the_others_alone(client) -> None:
    http, *_ = client
    body, result_id = kis_batch(http)
    rows = body["result_batch"]["rows"]
    assert len(rows) >= 2
    before = [row["submission_frames"] for row in rows]

    edit = http.post(
        f"/api/results/{result_id}/edit",
        json={"row_id": rows[0]["row_id"], "field": "frame", "value": "4242"},
    )
    assert edit.status_code == 200
    after = [row["submission_frames"] for row in edit.get_json()["result_batch"]["rows"]]
    assert after[0] == ["4242"]
    assert after[1:] == before[1:]


def test_a_kis_edit_cannot_touch_a_trake_batch(client) -> None:
    http, *_ = client
    _, kis_id = kis_batch(http)
    trake = http.post(
        "/api/video/temporal", json={"events": ["a person", "moves", "leaves"]}
    ).get_json()
    trake_id = trake["result_batch"]["result_id"]
    trake_before = [row["submission_frames"] for row in trake["result_batch"]["rows"]]

    kis_rows = http.get(f"/api/results/{kis_id}").get_json()["result_batch"]["rows"]
    http.post(
        f"/api/results/{kis_id}/edit",
        json={"row_id": kis_rows[0]["row_id"], "field": "frame", "value": "9999"},
    )
    trake_after = [
        row["submission_frames"]
        for row in http.get(f"/api/results/{trake_id}").get_json()["result_batch"]["rows"]
    ]
    assert trake_after == trake_before


def test_a_trake_event_edit_is_event_scoped_over_http(client) -> None:
    http, *_ = client
    trake = http.post(
        "/api/video/temporal", json={"events": ["a person", "moves", "leaves"]}
    ).get_json()
    result_id = trake["result_batch"]["result_id"]
    rows = trake["result_batch"]["rows"]
    before = list(rows[0]["submission_frames"])

    edit = http.post(
        f"/api/results/{result_id}/edit",
        json={"row_id": rows[0]["row_id"], "field": "frame", "value": "1234", "event_index": 1},
    )
    assert edit.status_code == 200
    after = edit.get_json()["result_batch"]["rows"][0]["submission_frames"]
    assert after[1] == "1234"
    assert after[0] == before[0] and after[2] == before[2]
    assert len(after) == 3


@pytest.mark.parametrize("value", ["-1", "abc", ""])
def test_an_invalid_edit_is_rejected_by_the_backend(client, value) -> None:
    http, *_ = client
    body, result_id = kis_batch(http)
    row_id = body["result_batch"]["rows"][0]["row_id"]
    response = http.post(
        f"/api/results/{result_id}/edit",
        json={"row_id": row_id, "field": "frame", "value": value},
    )
    assert response.status_code == 400
    assert response.get_json()["error_code"] in {"INVALID_FRAME_ID", "NEGATIVE_FRAME_ID"}


def test_reset_restores_a_row_and_the_batch(client) -> None:
    http, *_ = client
    body, result_id = kis_batch(http)
    rows = body["result_batch"]["rows"]
    original = rows[0]["submission_frames"]
    http.post(
        f"/api/results/{result_id}/edit",
        json={"row_id": rows[0]["row_id"], "field": "frame", "value": "5555"},
    )
    restored = http.post(
        f"/api/results/{result_id}/reset", json={"row_id": rows[0]["row_id"]}
    ).get_json()
    assert restored["result_batch"]["rows"][0]["submission_frames"] == original
    assert restored["result_batch"]["manual_edit_count"] == 0

    http.post(
        f"/api/results/{result_id}/edit",
        json={"row_id": rows[0]["row_id"], "field": "frame", "value": "6666"},
    )
    whole = http.post(f"/api/results/{result_id}/reset", json={}).get_json()
    assert whole["result_batch"]["manual_edit_count"] == 0
    assert whole["result_batch"]["rows"][0]["submission_frames"] == original


def test_an_edit_survives_into_the_exported_csv(client) -> None:
    http, *_ = client
    body, result_id = kis_batch(http)
    row_id = body["result_batch"]["rows"][0]["row_id"]
    http.post(
        f"/api/results/{result_id}/edit",
        json={"row_id": row_id, "field": "frame", "value": "31337"},
    )
    saved = http.post("/api/submission/save", json={"result_id": result_id, "name": "kis"})
    assert saved.status_code == 200
    exported = read_submission_csv(Path(saved.get_json()["path"]))
    assert exported[0][1] == "31337"
    report = json.loads(Path(saved.get_json()["report_path"]).read_text(encoding="utf-8"))
    assert report["manual_edit_count"] == 1


# ------------------------------------------------------------------ payloads


def test_result_payloads_expose_no_filesystem_paths(client) -> None:
    http, root_a, _, _ = client
    body, result_id = kis_batch(http)
    assert str(root_a) not in json.dumps(body)
    batch = http.get(f"/api/results/{result_id}").get_json()
    assert str(root_a) not in json.dumps(batch)


def test_health_reports_channel_availability_for_the_ui(client) -> None:
    http, *_ = client
    channels = http.get("/api/health").get_json()["retrieval_channels"]
    assert channels["clip"]["available"] is True
    # The UI disables a control whose source is empty rather than showing a dead switch.
    assert channels["ocr"]["available"] is False
    assert channels["ocr"]["reason"] == "no_populated_source_data"

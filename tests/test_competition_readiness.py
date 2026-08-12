"""Phase 11: the readiness preflight and the CLI/API surfaces that expose it.

Readiness is STRUCTURAL. It answers "can this system run and emit well-formed
submissions", never "are the answers right". The distinction is load-bearing in these
tests: a missing optional capability must degrade to a warning that still names the gap,
and a broken dataset, cache or validator must fail outright.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

import ui.app as appmod
from aic2026.cli import main as cli_main
from aic2026.config import config_to_dict
from aic2026.system_profile import (
    NOT_READY,
    OPTIONAL_CHANNELS,
    READY,
    READY_WITH_WARNINGS,
    REQUIRED_CHANNELS,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    evaluate_readiness,
)
from tests.release_support import (
    FakeEngine,
    build_engine,
    channel_status,
    make_config,
    make_data,
    qa_status,
)


@pytest.fixture()
def config(tmp_path):
    return make_config(make_data(tmp_path / "data"), tmp_path / "cache")


def status_of(report, name: str) -> str:
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"no check named {name!r} in {[c.name for c in report.checks]}")


def ready_engine(config):
    """An engine with every channel usable and a visual Q&A backend: the only READY shape."""
    return FakeEngine(
        config,
        channels=channel_status(clip=True, optional=True),
        qa=qa_status(visual=True, backend_type="vlm"),
    )


# ------------------------------------------------------------------ classification


def test_engineless_check_is_ready_with_warnings(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    report = evaluate_readiness(config)
    assert report.status == READY_WITH_WARNINGS
    assert report.ready is True
    assert status_of(report, "engine") == STATUS_WARN
    assert not report.failures


def test_warnings_never_block_and_failures_always_do(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    warned = evaluate_readiness(config, engine=FakeEngine(config))
    assert warned.status == READY_WITH_WARNINGS and warned.ready is True
    broken = evaluate_readiness(
        replace(config, cache=replace(config.cache, allow_stale_cache=True)), engine=FakeEngine(config)
    )
    assert broken.status == NOT_READY and broken.ready is False


def test_a_fully_capable_system_is_plain_ready(tmp_path) -> None:
    """READY is reachable — the verdict is not warning-padded into meaninglessness."""
    engine, config, _ = build_engine(tmp_path)
    config = replace(
        config, refinement=replace(config.refinement, scorer_device="cuda")
    )
    report = evaluate_readiness(config, engine=ready_engine(config))
    assert report.status == READY, [c.to_dict() for c in report.checks if c.status != STATUS_PASS]
    assert report.exit_code() == 0


@pytest.mark.parametrize(
    ("status", "code"), [(READY, 0), (READY_WITH_WARNINGS, 1), (NOT_READY, 2)]
)
def test_exit_codes_are_stable(tmp_path, status, code) -> None:
    from aic2026.system_profile import ReadinessReport

    assert ReadinessReport(status=status).exit_code() == code


# ------------------------------------------------------------------------ failures


def test_missing_cache_makes_the_system_not_ready(tmp_path) -> None:
    config = make_config(make_data(tmp_path / "data"), tmp_path / "absent_cache")
    report = evaluate_readiness(config)
    assert report.status == NOT_READY
    assert status_of(report, "cache") == STATUS_FAIL
    assert "cache" in {item.name for item in report.failures}


def test_stale_cache_is_a_failure_not_a_warning(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    # Rebuilding the config with different build options makes the on-disk cache stale.
    stale = replace(config, dataset=replace(config.dataset, load_objects=True))
    report = evaluate_readiness(stale)
    assert report.status == NOT_READY
    assert status_of(report, "cache") == STATUS_FAIL


def test_allow_stale_cache_is_refused_outright(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    permissive = replace(config, cache=replace(config.cache, allow_stale_cache=True))
    report = evaluate_readiness(permissive)
    assert status_of(report, "cache_policy") == STATUS_FAIL
    assert report.status == NOT_READY


def test_missing_data_root_fails(tmp_path) -> None:
    config = make_config(tmp_path / "gone", tmp_path / "cache")
    report = evaluate_readiness(config)
    assert status_of(report, "data_root") == STATUS_FAIL
    assert status_of(report, "dataset_scope") == STATUS_FAIL
    assert report.status == NOT_READY


def test_empty_scope_fails_because_there_is_nothing_to_search(tmp_path) -> None:
    root = make_data(tmp_path / "data")
    config = make_config(
        root, tmp_path / "cache", dataset={"scope": {"include_patterns": ["ZZZ_*"]}}
    )
    report = evaluate_readiness(config)
    assert status_of(report, "dataset_scope") == STATUS_FAIL


def test_required_channel_failure_is_fatal(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    broken = FakeEngine(config, channels=channel_status(clip=False, reason="no vectors"))
    report = evaluate_readiness(config, engine=broken)
    assert status_of(report, "channel_clip") == STATUS_FAIL
    assert report.status == NOT_READY
    assert "no vectors" in dict((c.name, c.message) for c in report.checks)["channel_clip"]


@pytest.mark.parametrize("name", OPTIONAL_CHANNELS)
def test_optional_channel_absence_is_only_a_warning(tmp_path, name) -> None:
    engine, config, _ = build_engine(tmp_path)
    report = evaluate_readiness(config, engine=FakeEngine(config))
    assert status_of(report, f"channel_{name}") == STATUS_WARN
    assert report.ready is True


def test_required_channels_are_exactly_what_the_tasks_need() -> None:
    assert REQUIRED_CHANNELS == ("clip",)
    assert set(REQUIRED_CHANNELS) & set(OPTIONAL_CHANNELS) == set()


def test_missing_frame_provider_fails(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    report = evaluate_readiness(config, engine=FakeEngine(config, frame_provider=None))
    assert status_of(report, "frame_provider") == STATUS_FAIL


def test_engine_built_for_another_dataset_fails(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    other = make_config(make_data(tmp_path / "other"), tmp_path / "cache")
    report = evaluate_readiness(config, engine=FakeEngine(other))
    assert status_of(report, "runtime_identity") == STATUS_FAIL
    assert report.status == NOT_READY


# ------------------------------------------------------------------------ warnings


def test_missing_visual_qa_backend_warns_and_does_not_claim_capability(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    report = evaluate_readiness(config, engine=FakeEngine(config, qa=qa_status(visual=False)))
    message = dict((c.name, c.message) for c in report.checks)["qa_backend"]
    assert status_of(report, "qa_backend") == STATUS_WARN
    assert "not exportable" in message
    assert report.profile.qa["visual_capable"] is False
    assert report.profile.qa["production_ready"] is False
    assert report.ready is True


def test_cpu_refinement_is_reported_as_a_cost_warning(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    report = evaluate_readiness(
        config, engine=FakeEngine(config, qa=qa_status(visual=True))
    )
    assert status_of(report, "refinement_device") == STATUS_WARN


def test_submission_validator_is_probed_every_run(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    report = evaluate_readiness(config)
    assert status_of(report, "submission_validator") == STATUS_PASS


# -------------------------------------------------------------------- report shape


def test_report_dict_is_json_safe_and_carries_the_structural_disclaimer(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    payload = evaluate_readiness(config, engine=FakeEngine(config)).to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert payload["status"] in {READY, READY_WITH_WARNINGS, NOT_READY}
    assert set(payload) >= {"status", "ready", "checks", "failures", "warnings", "profile", "note"}
    assert "no AIC ground truth" in payload["note"]
    assert "STRUCTURAL" in payload["note"]


def test_report_never_claims_accuracy(tmp_path) -> None:
    engine, config, _ = build_engine(tmp_path)
    text = json.dumps(evaluate_readiness(config, engine=FakeEngine(config)).to_dict()).lower()
    for forbidden in ("accuracy", "recall@", "r@1", "final score", "sota", "state of the art"):
        assert forbidden not in text


def test_supplied_profile_is_reused_rather_than_rebuilt(tmp_path) -> None:
    from aic2026.system_profile import build_system_profile

    engine, config, _ = build_engine(tmp_path)
    profile = build_system_profile(config, config_path="pinned.yaml")
    report = evaluate_readiness(config, profile=profile)
    assert report.profile is profile
    assert report.profile.config_path == "pinned.yaml"


# ---------------------------------------------------------------------------- CLI


def write_config(path, config):
    import yaml

    path.write_text(
        yaml.safe_dump({"aic2026": config_to_dict(config)}, sort_keys=False), encoding="utf-8"
    )
    return path


def test_cli_version_flag(capsys) -> None:
    from aic2026.version import PROJECT_VERSION

    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--version"])
    assert excinfo.value.code == 0
    assert PROJECT_VERSION in capsys.readouterr().out


def test_cli_competition_check_emits_json_and_a_warning_exit_code(tmp_path, capsys) -> None:
    engine, config, _ = build_engine(tmp_path)
    path = write_config(tmp_path / "cfg.yaml", config)
    out_path = tmp_path / "readiness.json"
    code = cli_main(
        ["--config", str(path), "competition-check", "--no-load-engine", "--output", str(out_path)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == READY_WITH_WARNINGS
    assert json.loads(out_path.read_text(encoding="utf-8")) == payload


def test_cli_competition_check_returns_two_when_not_ready(tmp_path, capsys) -> None:
    config = make_config(make_data(tmp_path / "data"), tmp_path / "absent")
    path = write_config(tmp_path / "cfg.yaml", config)
    code = cli_main(["--config", str(path), "competition-check", "--no-load-engine"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == NOT_READY


def test_cli_system_profile_writes_identity(tmp_path, capsys) -> None:
    engine, config, _ = build_engine(tmp_path)
    path = write_config(tmp_path / "cfg.yaml", config)
    out_path = tmp_path / "profile.json"
    code = cli_main(["--config", str(path), "system-profile", "--output", str(out_path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["config_hash"] and payload["project_version"]
    assert json.loads(out_path.read_text(encoding="utf-8")) == payload


def test_cli_serve_refuses_to_start_when_not_ready(tmp_path, capsys) -> None:
    config = make_config(tmp_path / "missing_root", tmp_path / "absent")
    path = write_config(tmp_path / "cfg.yaml", config)
    code = cli_main(["--config", str(path), "serve", "--no-activate"])
    assert code == 2
    assert "refusing to serve" in capsys.readouterr().err


# ---------------------------------------------------------------------------- API


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(appmod, "AIC_CACHE_DIR", tmp_path / "index")
    monkeypatch.setattr(appmod, "SUBMISSION_DIR", tmp_path / "submissions")
    app = appmod.create_app()
    app.testing = True
    return app.test_client()


def test_readiness_endpoint_reports_not_ready_with_503(client) -> None:
    response = client.get("/api/readiness")
    assert response.status_code in (200, 503)
    body = response.get_json()
    assert body["status"] in {READY, READY_WITH_WARNINGS, NOT_READY}
    assert (response.status_code == 503) == (body["status"] == NOT_READY)
    assert "checks" in body and body["checks"]


def test_health_carries_the_system_profile_and_submission_contract(client) -> None:
    body = client.get("/api/health").get_json()
    assert body["ok"] is True
    assert body["system"]["project_version"]
    assert body["system"]["submission_validation_version"] >= 1
    assert set(body["submission"]) >= {"tasks", "max_rows", "validation_version"}

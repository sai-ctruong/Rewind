"""Phase 11: the release configuration is one fixed, safe, parseable artifact.

`configs/competition.yaml` is what a competition run is identified by. These tests pin
the settings that would silently change what a submission means — the safety switches,
the frame policy, the cache identity — and check that the release config goes through
exactly the same validated loader as every other config.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aic2026.config import config_hash, config_to_dict, load_app_config
from aic2026.system_profile import build_system_profile

ROOT = Path(__file__).resolve().parent.parent
COMPETITION = ROOT / "configs" / "competition.yaml"
SETTINGS = ROOT / "configs" / "settings.yaml"


@pytest.fixture(scope="module")
def config():
    return load_app_config(str(COMPETITION))


def test_release_config_exists_and_parses(config) -> None:
    assert COMPETITION.is_file()
    assert config_hash(config)


def test_release_config_uses_the_same_loader_as_settings() -> None:
    """No release-only parser and no release-only field."""
    release_keys = set(config_to_dict(load_app_config(str(COMPETITION))))
    settings_keys = set(config_to_dict(load_app_config(str(SETTINGS))))
    assert release_keys == settings_keys


# ------------------------------------------------------------------ safety switches


def test_production_mode_is_on(config) -> None:
    assert config.runtime.production_mode is True


def test_hashing_fallback_encoder_is_forbidden(config) -> None:
    """A run must never retrieve with meaningless vectors."""
    assert config.encoder.allow_hashing_fallback is False


def test_stale_caches_are_rejected(config) -> None:
    assert config.cache.allow_stale_cache is False
    assert config.cache.validate_data_signature is True


def test_submitted_frame_policy_is_the_official_mapped_frame(config) -> None:
    """`preserve_coarse` keeps the submitted frame the BTC-mapped frame_idx.

    Refinement may still move the *visual* frame it looked at; what it must not do is
    change the number written into the CSV while the organisers' frame semantics for an
    arbitrary decoded frame are unconfirmed.
    """
    assert config.refinement.frame_output_policy == "preserve_coarse"


def test_submission_caps_match_the_official_row_limit(config) -> None:
    from aic2026.submission_validation import MAX_SUBMISSION_ROWS

    assert config.submission.max_predictions == MAX_SUBMISSION_ROWS
    assert config.submission.max_answers == MAX_SUBMISSION_ROWS
    assert config.ranking.final_top_k == MAX_SUBMISSION_ROWS


# ------------------------------------------------------------------ cache identity


def test_release_config_points_at_the_multichannel_cache(config) -> None:
    assert Path(config.dataset.cache_dir).name == "aic2026_index_channels"
    # The object and metadata channels have no source data unless these are set at BUILD
    # time; the cache fingerprint covers both, so a mismatch is caught rather than
    # silently producing empty channels.
    assert config.dataset.load_objects is True
    assert config.dataset.include_media_text is True


def test_scope_is_resolved_from_disk(config) -> None:
    """`existing_videos` selects only videos that really have an MP4 plus map+CLIP."""
    assert config.dataset.scope.mode == "existing_videos"


def test_cache_fingerprint_is_stable_across_loads() -> None:
    first = build_system_profile(load_app_config(str(COMPETITION)))
    second = build_system_profile(load_app_config(str(COMPETITION)))
    assert first.config_hash == second.config_hash
    assert first.cache_fingerprint == second.cache_fingerprint


# ------------------------------------------------------------------- capabilities


def test_optional_text_channels_stay_enabled_so_they_report_honestly(config) -> None:
    """Disabling OCR/ASR/caption would hide that their source data is empty.

    Enabled-but-unavailable is the truthful state: the channel measures its own source
    and reports `available: false`. Disabling it would make the same absence look like a
    deliberate configuration choice.
    """
    channels = config.retrieval_channels
    for name in ("clip", "bm25", "objects", "metadata", "ocr", "asr", "caption"):
        assert getattr(channels, f"{name}_enabled") is True


def test_expensive_refinement_is_off_by_default(config) -> None:
    """Off for latency on CPU, not because it was found to hurt quality."""
    assert config.refinement.enabled is False
    assert config.trake.refinement_enabled is False


def test_qa_backend_is_optional_so_kis_and_trake_still_run(config) -> None:
    assert config.qa.backend_required is False


def test_trake_uses_beam_dp_not_a_claim_of_exact_dp(config) -> None:
    assert config.trake.alignment_method == "beam_dp"
    assert config.trake.k_best_per_video >= 1


# ------------------------------------------------------------------------ hygiene


def test_release_config_contains_no_secrets() -> None:
    raw = COMPETITION.read_text(encoding="utf-8").lower()
    for needle in ("api_key", "api-key", "sk-", "secret", "password", "token:", "bearer"):
        assert needle not in raw


def test_release_config_hardcodes_no_machine_specific_path() -> None:
    raw = COMPETITION.read_text(encoding="utf-8")
    assert ":\\" not in raw and ":/" not in raw.replace("http://", "").replace("https://", "")
    assert "/home/" not in raw and "C:" not in raw


def test_release_config_is_valid_yaml_with_a_single_root() -> None:
    payload = yaml.safe_load(COMPETITION.read_text(encoding="utf-8"))
    assert list(payload) == ["aic2026"]


def test_release_config_documents_that_nothing_was_tuned_for_accuracy() -> None:
    raw = COMPETITION.read_text(encoding="utf-8")
    assert "no AIC ground truth" in raw


def test_cli_accepts_the_release_config(tmp_path, capsys) -> None:
    from aic2026.cli import main as cli_main

    code = cli_main(["--config", str(COMPETITION), "show-config"])
    assert code == 0
    payload = capsys.readouterr().out
    assert "config_hash" in payload

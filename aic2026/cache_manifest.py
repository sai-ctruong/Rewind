"""Cache manifest, fingerprint, and validation helpers for AIC 2026."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ingestion.schemas import AIC_RECORD_SCHEMA_VERSION

from .config import AppConfig, config_hash

CACHE_MANIFEST_SCHEMA_VERSION = 1
CACHE_MANIFEST_FILENAME = "cache_manifest.json"


class CacheManifestError(RuntimeError):
    """Raised when a cache manifest or a referenced cache file is corrupt."""


class LegacyCacheError(CacheManifestError):
    """Raised when a cache predates manifests and fallback is not allowed."""


class StaleCacheError(CacheManifestError):
    """Raised when a cache manifest does not match current build inputs."""


@dataclass(frozen=True)
class CacheMismatch:
    field: str
    expected: Any
    actual: Any
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheValidationResult:
    valid: bool
    legacy: bool = False
    stale: bool = False
    corrupt: bool = False
    hard_mismatches: tuple[CacheMismatch, ...] = ()
    warnings: tuple[CacheMismatch, ...] = ()
    manifest_path: str | None = None

    @property
    def status(self) -> str:
        if self.corrupt:
            return "corrupt"
        if self.legacy:
            return "legacy"
        if self.stale:
            return "stale"
        return "valid" if self.valid else "missing"

    def to_dict(self) -> dict[str, Any]:
        if self.legacy:
            stale_reason = "cache manifest is missing"
        elif self.hard_mismatches:
            stale_reason = "; ".join(item.message for item in self.hard_mismatches)
        else:
            stale_reason = None
        return {
            "status": self.status,
            "valid": self.valid,
            "legacy": self.legacy,
            "stale": self.stale,
            "corrupt": self.corrupt,
            "stale_reason": stale_reason,
            "hard_mismatches": [item.to_dict() for item in self.hard_mismatches],
            "mismatches": [item.to_dict() for item in self.hard_mismatches],
            "warnings": [item.to_dict() for item in self.warnings],
            "manifest_path": self.manifest_path,
        }


@dataclass(frozen=True)
class CacheBuildOptions:
    """Only options and dataset facts that can change the persisted cache."""

    data_root: str
    config_hash: str
    data_signature: str
    video_ids_hash: str
    video_count: int
    frame_count: int
    feature_dim: int
    feature_dtype: str
    load_objects: bool
    include_media_text: bool
    include_ocr: bool
    include_asr: bool
    include_captions: bool
    index_kind: str
    index_params: dict[str, Any]
    encoder_feature_space: str
    record_schema_version: int

    def fingerprint_payload(self) -> dict[str, Any]:
        # Dataset identity is validated separately through signatures and counts.
        return {
            "data_root": self.data_root,
            "load_objects": self.load_objects,
            "include_media_text": self.include_media_text,
            "include_ocr": self.include_ocr,
            "include_asr": self.include_asr,
            "include_captions": self.include_captions,
            "index_kind": self.index_kind,
            "index_params": self.index_params,
            "feature_dim": self.feature_dim,
            "encoder_feature_space": self.encoder_feature_space,
            "record_schema_version": self.record_schema_version,
        }


@dataclass(frozen=True)
class CacheManifest:
    schema_version: int
    created_at_utc: str
    code_version: str | None
    config_hash: str
    cache_fingerprint: str
    data_root: str
    data_signature: str
    video_ids_hash: str
    video_count: int
    frame_count: int
    feature_dim: int
    feature_dtype: str
    load_objects: bool
    include_media_text: bool
    include_ocr: bool
    include_asr: bool
    include_captions: bool
    index_kind: str
    index_params: dict[str, Any]
    encoder_feature_space: str
    record_schema_version: int
    files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheManifest":
        required = set(cls.__dataclass_fields__)
        missing = sorted(required - set(data))
        if missing:
            raise CacheManifestError(
                "Cache manifest is missing required field(s): " + ", ".join(missing)
            )
        try:
            files = data.get("files", {})
            if not isinstance(files, Mapping):
                raise TypeError("files must be an object")
            index_params = data["index_params"]
            if not isinstance(index_params, Mapping):
                raise TypeError("index_params must be an object")
            return cls(
                schema_version=int(data["schema_version"]),
                created_at_utc=str(data["created_at_utc"]),
                code_version=None if data["code_version"] is None else str(data["code_version"]),
                config_hash=str(data["config_hash"]),
                cache_fingerprint=str(data["cache_fingerprint"]),
                data_root=str(data["data_root"]),
                data_signature=str(data["data_signature"]),
                video_ids_hash=str(data["video_ids_hash"]),
                video_count=int(data["video_count"]),
                frame_count=int(data["frame_count"]),
                feature_dim=int(data["feature_dim"]),
                feature_dtype=str(data["feature_dtype"]),
                load_objects=bool(data["load_objects"]),
                include_media_text=bool(data["include_media_text"]),
                include_ocr=bool(data["include_ocr"]),
                include_asr=bool(data["include_asr"]),
                include_captions=bool(data["include_captions"]),
                index_kind=str(data["index_kind"]),
                index_params=dict(index_params),
                encoder_feature_space=str(data["encoder_feature_space"]),
                record_schema_version=int(data["record_schema_version"]),
                files={str(key): str(value) for key, value in files.items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheManifestError(f"Invalid cache manifest field: {exc}") from exc


def canonical_data_root(path: str | Path) -> str:
    text = Path(path).expanduser().resolve(strict=False).as_posix()
    return text.casefold() if os.name == "nt" else text


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_video_ids(video_ids: Iterable[str]) -> str:
    return _sha256_json(sorted(set(str(item) for item in video_ids)))


def _file_fact(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    try:
        stat = path.stat()
    except OSError:
        return {"path": relative, "missing": True}
    return {"path": relative, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def build_data_signature(
    data_root: str | Path,
    *,
    video_ids: Sequence[str],
    include_objects: bool,
    include_media_text: bool,
    include_keyframes: bool = False,
    record_schema_version: int = AIC_RECORD_SCHEMA_VERSION,
) -> str:
    """Hash only deterministic file metadata; large media/file content is never read."""

    root = Path(data_root).expanduser().resolve(strict=False)
    facts: list[dict[str, Any]] = []
    for video_id in sorted(set(str(item) for item in video_ids)):
        facts.append(_file_fact(root, root / "map-keyframes" / f"{video_id}.csv"))
        facts.append(_file_fact(root, root / "clip-features-32" / f"{video_id}.npy"))
        if include_media_text:
            facts.append(_file_fact(root, root / "media-info" / f"{video_id}.json"))
        if include_objects:
            object_dir = root / "objects" / video_id
            if object_dir.exists():
                facts.extend(
                    _file_fact(root, path)
                    for path in sorted(object_dir.rglob("*.json"), key=lambda item: item.as_posix())
                )
            else:
                facts.append(_file_fact(root, object_dir))
        if include_keyframes:
            frame_dir = root / "keyframes" / video_id
            if frame_dir.exists():
                facts.extend(
                    _file_fact(root, path)
                    for path in sorted(
                        (path for path in frame_dir.rglob("*") if path.is_file()),
                        key=lambda item: item.as_posix(),
                    )
                )
            else:
                facts.append(_file_fact(root, frame_dir))
    facts.sort(key=lambda item: item["path"])
    return _sha256_json(
        {
            "data_root": canonical_data_root(root),
            "video_ids": sorted(set(str(item) for item in video_ids)),
            "record_schema_version": int(record_schema_version),
            "files": facts,
        }
    )


def inspect_data_inputs(
    data_root: str | Path,
    *,
    limit_videos: int | None = None,
    limit_frames_per_video: int | None = None,
) -> dict[str, Any]:
    """Collect inexpensive cache facts using CSV rows and memory-mapped NPY headers."""

    root = Path(data_root)
    feature_dir = root / "clip-features-32"
    video_ids = sorted(path.stem for path in feature_dir.glob("*.npy")) if feature_dir.exists() else []
    if limit_videos is not None:
        video_ids = video_ids[: max(0, int(limit_videos))]
    frame_count = 0
    dimensions: set[int] = set()
    dtypes: set[str] = set()
    for video_id in video_ids:
        feature_path = feature_dir / f"{video_id}.npy"
        map_path = root / "map-keyframes" / f"{video_id}.csv"
        if not feature_path.exists() or not map_path.exists():
            continue
        try:
            features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            feature_rows = int(features.shape[0]) if features.ndim >= 1 else 0
            if features.ndim == 2:
                dimensions.add(int(features.shape[1]))
            dtypes.add(str(features.dtype))
        except (OSError, ValueError, EOFError):
            continue
        try:
            with map_path.open("r", encoding="utf-8-sig", newline="") as handle:
                map_rows = sum(1 for _ in csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error):
            map_rows = 0
        if map_rows == feature_rows:
            rows_to_count = map_rows
            frame_limit = (
                None
                if limit_frames_per_video is None
                else max(0, int(limit_frames_per_video))
            )
            if frame_limit is not None and rows_to_count > frame_limit:
                rows_to_count = frame_limit
            frame_count += rows_to_count
    feature_dim = next(iter(dimensions)) if len(dimensions) == 1 else 0
    if not dtypes:
        feature_dtype = "unknown"
    elif len(dtypes) == 1:
        feature_dtype = next(iter(dtypes))
    else:
        feature_dtype = "mixed:" + ",".join(sorted(dtypes))
    return {
        "video_ids": video_ids,
        "video_count": len(video_ids),
        "frame_count": frame_count,
        "feature_dim": feature_dim,
        "feature_dtype": feature_dtype,
    }


def encoder_feature_space(config: AppConfig) -> str:
    return (
        f"{config.encoder.model_name}|dim={int(config.encoder.feature_dim)}"
        f"|normalize={str(bool(config.encoder.normalize)).lower()}"
    )


def _index_params(
    *,
    index_kind: str,
    verify_keyframes: bool,
    limit_videos: int | None,
    limit_frames_per_video: int | None,
    expected_feature_dim: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "verify_keyframes": bool(verify_keyframes),
        "limit_videos": limit_videos,
        "limit_frames_per_video": limit_frames_per_video,
        "expected_feature_dim": expected_feature_dim,
    }
    if index_kind == "hnsw":
        params.update({"hnsw_m": 32, "ef_construction": 200, "ef_search": 2048})
    return params


def cache_build_options_from_config(
    app_config: AppConfig,
    *,
    data_root: str | Path | None = None,
    load_objects: bool | None = None,
    include_media_text: bool | None = None,
    verify_keyframes: bool | None = None,
    index_kind: str | None = None,
    limit_videos: int | None = None,
    limit_frames_per_video: int | None = None,
) -> CacheBuildOptions:
    root = data_root if data_root is not None else app_config.dataset.root
    load_objects = app_config.dataset.load_objects if load_objects is None else bool(load_objects)
    include_media_text = (
        app_config.dataset.include_media_text if include_media_text is None else bool(include_media_text)
    )
    verify_keyframes = (
        app_config.dataset.verify_keyframes if verify_keyframes is None else bool(verify_keyframes)
    )
    index_kind = str(index_kind or app_config.dataset.index_kind)
    inspected = inspect_data_inputs(
        root,
        limit_videos=limit_videos,
        limit_frames_per_video=limit_frames_per_video,
    )
    video_ids = inspected["video_ids"]
    return CacheBuildOptions(
        data_root=canonical_data_root(root),
        config_hash=config_hash(app_config),
        data_signature=build_data_signature(
            root,
            video_ids=video_ids,
            include_objects=load_objects,
            include_media_text=include_media_text,
            include_keyframes=verify_keyframes,
        ),
        video_ids_hash=hash_video_ids(video_ids),
        video_count=int(inspected["video_count"]),
        frame_count=int(inspected["frame_count"]),
        feature_dim=int(app_config.encoder.feature_dim),
        feature_dtype=str(inspected["feature_dtype"]),
        load_objects=load_objects,
        include_media_text=include_media_text,
        include_ocr=False,
        include_asr=False,
        include_captions=False,
        index_kind=index_kind,
        index_params=_index_params(
            index_kind=index_kind,
            verify_keyframes=verify_keyframes,
            limit_videos=limit_videos,
            limit_frames_per_video=limit_frames_per_video,
            expected_feature_dim=app_config.dataset.validation.expected_feature_dim,
        ),
        encoder_feature_space=encoder_feature_space(app_config),
        record_schema_version=AIC_RECORD_SCHEMA_VERSION,
    )


def cache_fingerprint(
    app_config: AppConfig | CacheBuildOptions,
    *,
    data_root: str | Path | None = None,
) -> str:
    if isinstance(app_config, CacheBuildOptions):
        return _sha256_json(app_config.fingerprint_payload())
    root = data_root if data_root is not None else app_config.dataset.root
    payload = {
        "data_root": canonical_data_root(root),
        "load_objects": bool(app_config.dataset.load_objects),
        "include_media_text": bool(app_config.dataset.include_media_text),
        "include_ocr": False,
        "include_asr": False,
        "include_captions": False,
        "index_kind": str(app_config.dataset.index_kind),
        "index_params": _index_params(
            index_kind=str(app_config.dataset.index_kind),
            verify_keyframes=bool(app_config.dataset.verify_keyframes),
            limit_videos=None,
            limit_frames_per_video=None,
            expected_feature_dim=app_config.dataset.validation.expected_feature_dim,
        ),
        "feature_dim": int(app_config.encoder.feature_dim),
        "encoder_feature_space": encoder_feature_space(app_config),
        "record_schema_version": AIC_RECORD_SCHEMA_VERSION,
    }
    return _sha256_json(payload)


def detect_code_version(repo_root: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=None if repo_root is None else str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except Exception:
        return None


def build_cache_manifest(
    *,
    app_config: AppConfig,
    data_root: str | Path,
    video_ids: Sequence[str],
    video_count: int,
    frame_count: int,
    feature_dim: int,
    feature_dtype: str,
    index_kind: str,
    index_params: Mapping[str, Any],
    record_schema_version: int,
    files: Mapping[str, str],
    code_version: str | None = None,
    load_objects: bool | None = None,
    include_media_text: bool | None = None,
) -> CacheManifest:
    load_objects = app_config.dataset.load_objects if load_objects is None else bool(load_objects)
    include_media_text = (
        app_config.dataset.include_media_text if include_media_text is None else bool(include_media_text)
    )
    signature = build_data_signature(
        data_root,
        video_ids=video_ids,
        include_objects=load_objects,
        include_media_text=include_media_text,
        include_keyframes=bool(index_params.get("verify_keyframes", False)),
        record_schema_version=record_schema_version,
    )
    options = CacheBuildOptions(
        data_root=canonical_data_root(data_root),
        config_hash=config_hash(app_config),
        data_signature=signature,
        video_ids_hash=hash_video_ids(video_ids),
        video_count=int(video_count),
        frame_count=int(frame_count),
        feature_dim=int(feature_dim),
        feature_dtype=str(feature_dtype),
        load_objects=load_objects,
        include_media_text=include_media_text,
        include_ocr=False,
        include_asr=False,
        include_captions=False,
        index_kind=str(index_kind),
        index_params=dict(index_params),
        encoder_feature_space=encoder_feature_space(app_config),
        record_schema_version=int(record_schema_version),
    )
    return CacheManifest(
        schema_version=CACHE_MANIFEST_SCHEMA_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        code_version=code_version if code_version is not None else detect_code_version(),
        config_hash=options.config_hash,
        cache_fingerprint=cache_fingerprint(options),
        data_root=options.data_root,
        data_signature=options.data_signature,
        video_ids_hash=options.video_ids_hash,
        video_count=options.video_count,
        frame_count=options.frame_count,
        feature_dim=options.feature_dim,
        feature_dtype=options.feature_dtype,
        load_objects=options.load_objects,
        include_media_text=options.include_media_text,
        include_ocr=options.include_ocr,
        include_asr=options.include_asr,
        include_captions=options.include_captions,
        index_kind=options.index_kind,
        index_params=options.index_params,
        encoder_feature_space=options.encoder_feature_space,
        record_schema_version=options.record_schema_version,
        files={str(key): str(value).replace("\\", "/") for key, value in files.items()},
    )


def write_cache_manifest_atomic(manifest: CacheManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_cache_manifest(path: str | Path) -> CacheManifest:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheManifestError(f"Cannot read cache manifest {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise CacheManifestError(f"Cache manifest {path} must contain a JSON object")
    return CacheManifest.from_dict(data)


def _mismatch(field: str, expected: Any, actual: Any, severity: str = "hard") -> CacheMismatch:
    return CacheMismatch(
        field=field,
        expected=expected,
        actual=actual,
        severity=severity,
        message=f"Cache field {field!r} mismatch: expected {expected!r}, actual {actual!r}.",
    )


def validate_cache_manifest(
    manifest: CacheManifest,
    expected: CacheBuildOptions,
    *,
    current_data_signature: str | None = None,
    current_video_ids_hash: str | None = None,
    current_code_version: str | None = None,
    code_version_policy: str = "warn",
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> CacheValidationResult:
    hard: list[CacheMismatch] = []
    warnings: list[CacheMismatch] = []
    comparisons = {
        "schema_version": (CACHE_MANIFEST_SCHEMA_VERSION, manifest.schema_version),
        "data_root": (expected.data_root, manifest.data_root),
        "video_ids_hash": (
            current_video_ids_hash or expected.video_ids_hash,
            manifest.video_ids_hash,
        ),
        "video_count": (expected.video_count, manifest.video_count),
        "frame_count": (expected.frame_count, manifest.frame_count),
        "feature_dim": (expected.feature_dim, manifest.feature_dim),
        "feature_dtype": (expected.feature_dtype, manifest.feature_dtype),
        "load_objects": (expected.load_objects, manifest.load_objects),
        "include_media_text": (expected.include_media_text, manifest.include_media_text),
        "include_ocr": (expected.include_ocr, manifest.include_ocr),
        "include_asr": (expected.include_asr, manifest.include_asr),
        "include_captions": (expected.include_captions, manifest.include_captions),
        "index_kind": (expected.index_kind, manifest.index_kind),
        "index_params": (expected.index_params, manifest.index_params),
        "encoder_feature_space": (expected.encoder_feature_space, manifest.encoder_feature_space),
        "record_schema_version": (expected.record_schema_version, manifest.record_schema_version),
        "cache_fingerprint": (cache_fingerprint(expected), manifest.cache_fingerprint),
    }
    if current_data_signature is not None:
        comparisons["data_signature"] = (current_data_signature, manifest.data_signature)
    for field_name, (wanted, actual) in comparisons.items():
        if wanted != actual:
            hard.append(_mismatch(field_name, wanted, actual))
    if expected.config_hash != manifest.config_hash:
        warnings.append(_mismatch("config_hash", expected.config_hash, manifest.config_hash, "warning"))
    if current_code_version and manifest.code_version and current_code_version != manifest.code_version:
        severity = "warning" if code_version_policy == "warn" else code_version_policy
        item = _mismatch("code_version", current_code_version, manifest.code_version, severity)
        if code_version_policy == "reject":
            hard.append(item)
        elif code_version_policy == "warn":
            warnings.append(item)
    if cache_dir is not None:
        root = Path(cache_dir).resolve(strict=False)
        for file_key, relative in manifest.files.items():
            candidate = (root / relative).resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError:
                hard.append(_mismatch(f"files.{file_key}", "path inside cache", relative))
                continue
            if not candidate.is_file():
                hard.append(_mismatch(f"files.{file_key}", "existing file", relative))
    return CacheValidationResult(
        valid=not hard,
        stale=bool(hard),
        hard_mismatches=tuple(hard),
        warnings=tuple(warnings),
        manifest_path=None if manifest_path is None else str(manifest_path),
    )


def inspect_cache(
    cache_dir: str | Path,
    expected: CacheBuildOptions,
    *,
    validate_data_signature: bool = True,
    code_version_policy: str = "warn",
    current_code_version: str | None = None,
) -> dict[str, Any]:
    root = Path(cache_dir)
    manifest_path = root / CACHE_MANIFEST_FILENAME
    entry_path = root / "entry" / "entry.pkl"
    exists = root.exists() and any((entry_path.exists(), manifest_path.exists()))
    base: dict[str, Any] = {
        "exists": exists,
        "manifest_exists": manifest_path.is_file(),
        "entry_exists": entry_path.is_file(),
        "manifest_path": str(manifest_path),
        "manifest": None,
    }
    if not manifest_path.is_file():
        legacy = entry_path.is_file()
        result = CacheValidationResult(
            valid=False,
            legacy=legacy,
            stale=legacy,
            manifest_path=str(manifest_path),
        )
        return {**base, **result.to_dict()}
    try:
        manifest = read_cache_manifest(manifest_path)
    except CacheManifestError as exc:
        mismatch = CacheMismatch("manifest", "valid JSON and schema", None, "hard", str(exc))
        result = CacheValidationResult(
            valid=False,
            stale=True,
            corrupt=True,
            hard_mismatches=(mismatch,),
            manifest_path=str(manifest_path),
        )
        return {**base, **result.to_dict(), "error": str(exc)}
    if manifest.schema_version != CACHE_MANIFEST_SCHEMA_VERSION:
        mismatch = _mismatch(
            "schema_version", CACHE_MANIFEST_SCHEMA_VERSION, manifest.schema_version
        )
        result = CacheValidationResult(
            valid=False,
            stale=True,
            corrupt=True,
            hard_mismatches=(mismatch,),
            manifest_path=str(manifest_path),
        )
        return {**base, **result.to_dict(), "manifest": manifest.to_dict()}
    result = validate_cache_manifest(
        manifest,
        expected,
        current_data_signature=expected.data_signature if validate_data_signature else None,
        current_video_ids_hash=expected.video_ids_hash,
        current_code_version=current_code_version,
        code_version_policy=code_version_policy,
        cache_dir=root,
        manifest_path=manifest_path,
    )
    hard = list(result.hard_mismatches)
    required_files = {"entry", "index", "clip_index", "config_snapshot", "dataset_stats", "manifest"}
    if manifest.record_schema_version >= AIC_RECORD_SCHEMA_VERSION:
        required_files.add("dataset_report")
    for file_key in sorted(required_files - set(manifest.files)):
        hard.append(
            CacheMismatch(
                field=f"files.{file_key}",
                expected="manifest file reference",
                actual=None,
                severity="hard",
                message=f"Cache manifest is missing required file reference {file_key!r}.",
            )
        )
    corrupt = any(item.field.startswith("files.") for item in hard)
    for file_key in ("config_snapshot", "dataset_stats", "dataset_report"):
        relative = manifest.files.get(file_key)
        if not relative:
            continue
        snapshot_path = root / relative
        if not snapshot_path.is_file():
            continue
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(snapshot, Mapping):
                raise TypeError("JSON root must be an object")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            corrupt = True
            hard.append(
                CacheMismatch(
                    field=f"files.{file_key}",
                    expected="valid JSON object",
                    actual=relative,
                    severity="hard",
                    message=f"Invalid {file_key} file {snapshot_path}: {exc}",
                )
            )
    if corrupt:
        result = CacheValidationResult(
            valid=False,
            stale=True,
            corrupt=True,
            hard_mismatches=tuple(hard),
            warnings=result.warnings,
            manifest_path=str(manifest_path),
        )
    return {**base, **result.to_dict(), "manifest": manifest.to_dict()}


def mismatch_message(result: CacheValidationResult | Mapping[str, Any]) -> str:
    mismatches = (
        result.hard_mismatches
        if isinstance(result, CacheValidationResult)
        else result.get("hard_mismatches", [])
    )
    fields = [item.field if isinstance(item, CacheMismatch) else str(item.get("field")) for item in mismatches]
    return ", ".join(fields) if fields else "unknown cache mismatch"


__all__ = [
    "CACHE_MANIFEST_FILENAME",
    "CACHE_MANIFEST_SCHEMA_VERSION",
    "CacheBuildOptions",
    "CacheManifest",
    "CacheManifestError",
    "CacheMismatch",
    "CacheValidationResult",
    "LegacyCacheError",
    "StaleCacheError",
    "build_cache_manifest",
    "build_data_signature",
    "cache_build_options_from_config",
    "cache_fingerprint",
    "canonical_data_root",
    "detect_code_version",
    "hash_video_ids",
    "inspect_cache",
    "inspect_data_inputs",
    "mismatch_message",
    "read_cache_manifest",
    "validate_cache_manifest",
    "write_cache_manifest_atomic",
]

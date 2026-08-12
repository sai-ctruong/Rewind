"""Project version and build identity.

The version is what a submission run is identified by, so it is a plain constant rather
than something derived at import time. The git commit is looked up lazily and degrades to
`None` outside a checkout — a release identity that crashes on a source tarball is worse
than one that admits it does not know.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

# Phase 11 is the final implementation phase, so the version tracks it directly rather
# than pretending to a 1.0 the competition has not yet validated.
VERSION = "0.11.0"
RELEASE_NAME = "aic2026"
PROJECT_VERSION = f"{VERSION}-{RELEASE_NAME}"
RELEASE_TAG = "aic2026-competition-ready"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit(short: bool = True) -> Optional[str]:
    """Current commit, or None when git is unavailable or this is not a checkout."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        output = subprocess.run(
            args,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode != 0:
        return None
    value = output.stdout.strip()
    return value or None


def git_is_dirty() -> Optional[bool]:
    """Whether the working tree has uncommitted changes; None if unknown."""
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if output.returncode != 0:
        return None
    return bool(output.stdout.strip())


__all__ = [
    "PROJECT_VERSION",
    "RELEASE_NAME",
    "RELEASE_TAG",
    "VERSION",
    "git_commit",
    "git_is_dirty",
]

"""Persist the two existing fitted estimators; loading never calls fit()."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib

from . import model

PACKAGES = ("scikit-learn", "numpy", "scipy", "pandas", "joblib")


def environment() -> dict[str, str]:
    return {"python": ".".join(platform.python_version_tuple()[:2]),
            **{name: version(name) for name in PACKAGES}}


def save(path: Path, stuff: tuple, full: tuple, metadata: dict) -> dict:
    """Save an explicitly requested, one-time fit without replacing an old model."""
    manifest = path.with_suffix(".json")
    if path.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite the frozen model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"stuff": stuff, "full": full}
    joblib.dump(bundle, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {**metadata, "format_version": 1, "environment": environment(),
            "sha256": digest, "model_version": f"statcast-2025-{digest[:12]}",
            "features": {"stuff": stuff[1], "full": full[1]}}
    manifest.write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return meta


def load(path: Path) -> tuple[dict, dict]:
    """Load only the trusted repo artifact, with hash and compatibility checks.

    joblib uses pickle: do not pass untrusted or externally supplied files.
    """
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["format_version"] != 1:
        raise ValueError("Unsupported model bundle format")
    if meta["environment"] != environment():
        raise ValueError("Model environment mismatch; use Python 3.12 and requirements-model.txt")
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Model artifact SHA-256 does not match its manifest")
    bundle = joblib.load(path)
    for name, include_location in (("stuff", False), ("full", True)):
        if bundle[name][1] != model.feature_set(include_location):
            raise ValueError(f"Unexpected {name} feature ordering")
        if meta["features"][name] != bundle[name][1]:
            raise ValueError(f"Manifest feature ordering differs for {name}")
    return bundle, meta

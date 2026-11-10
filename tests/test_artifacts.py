"""Serialization must preserve real model predictions and reject bad artifacts."""
import json

import numpy as np
import pytest

from pitchquality import artifacts, model
from test_model import synth


@pytest.fixture(scope="module")
def fitted():
    train = synth(n=1200)
    stuff = model.fit(train, False)
    full = model.fit(train, True)
    return stuff, full, synth(n=100, seed=29)


def test_roundtrip_preserves_predictions_and_feature_order(tmp_path, fitted, monkeypatch):
    stuff, full, test = fitted
    path = tmp_path / "model.joblib"
    manifest = artifacts.save(path, stuff, full, {"train_auc": 0.8})
    monkeypatch.setattr(model, "fit", lambda *a, **kw: pytest.fail("loading must not fit"))
    bundle, loaded_meta = artifacts.load(path)
    assert loaded_meta == manifest
    for name, original in (("stuff", stuff), ("full", full)):
        assert bundle[name][1] == original[1]
        np.testing.assert_array_equal(model.predict(bundle[name][0], test, bundle[name][1]),
                                      model.predict(original[0], test, original[1]))
    with pytest.raises(FileExistsError):
        artifacts.save(path, stuff, full, {})


def test_corruption_and_version_mismatch_fail_before_unpickling(tmp_path, fitted, monkeypatch):
    stuff, full, _ = fitted
    path = tmp_path / "model.joblib"
    artifacts.save(path, stuff, full, {})
    monkeypatch.setattr(artifacts.joblib, "load", lambda *a: pytest.fail("must validate before unpickling"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        artifacts.load(path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["environment"]["scikit-learn"] = "incompatible"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="environment mismatch"):
        artifacts.load(path)

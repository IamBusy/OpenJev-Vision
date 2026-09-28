import json
from pathlib import Path

import pytest


def test_world_rendering_replays_without_credentials_or_provider_calls(monkeypatch, tmp_path):
    from openjev.data_v03 import build_worlds, render_worlds

    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts/v03-render").exists():
        pytest.skip("Saved rendering responses are local artifacts")
    cache = root / "artifacts/v03-render"
    root = tmp_path
    (root / "artifacts").mkdir()
    (root / "artifacts/v03-render").symlink_to(cache, target_is_directory=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay tried to use credentials or a model API")

    monkeypatch.setattr("openjev.data_v03.credentials", forbidden)
    monkeypatch.setattr("openjev.data_v03.call_teacher", forbidden)
    worlds = render_worlds(root, build_worlds(), replay=True)
    assert len(worlds) == 120
    assert all(isinstance(w["text"], str) and w["text"] for w in worlds)
    assert json.loads((root / "reports/v03/rendering.json").read_text())["accepted"] == 27

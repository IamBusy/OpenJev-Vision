from pathlib import Path

import pytest

from openjev.branch_model import BranchDecision
from openjev.io import sha256, write_json


def test_hub_loader_pins_base_and_loads_extra_head_and_calibration(tmp_path, monkeypatch):
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    write_json(
        checkpoint / "openjev_config.json",
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "model_revision": "pinned-base-revision",
        },
    )
    write_json(checkpoint / "calibration-v03.json", {"choice": {"temperature": 2.0}})
    (checkpoint / "head.safetensors").write_bytes(b"test fixture")
    write_json(
        checkpoint / "MANIFEST.json",
        {"files": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()}},
    )
    calls = []

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(checkpoint if len(calls) == 1 else tmp_path / "base")

    class Recorder(BranchDecision):
        def __init__(self, root, **kwargs):
            self.arguments = kwargs
            self.default_temperatures = None

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    model = Recorder.from_pretrained(
        "owner/model", revision="pinned-model-revision", local_files_only=True
    )
    assert calls[0][1]["revision"] == "pinned-model-revision"
    assert calls[1][1]["revision"] == "pinned-base-revision"
    assert all(call[1]["local_files_only"] for call in calls)
    assert model.arguments["checkpoint"] == checkpoint
    assert Path(model.arguments["base_model_path"]) == tmp_path / "base"
    assert model.default_temperatures["choice"]["temperature"] == 2.0
    # Missing custom head must fail before downloading/loading the backbone.
    (checkpoint / "head.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        Recorder.from_pretrained(checkpoint, base_model_path=tmp_path / "base")

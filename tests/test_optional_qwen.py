import builtins

import pytest

from openjev.branch_model import BranchDecision
from openjev.qwen import QwenDecision


@pytest.mark.parametrize("model_class", [BranchDecision, QwenDecision])
def test_missing_qwen_extra_reports_install_command_before_loading_weights(
    model_class, tmp_path, monkeypatch
):
    original_import = builtins.__import__

    def without_peft(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ModuleNotFoundError("No module named 'peft'", name="peft")
        return original_import(name, *args, **kwargs)

    # Exercise this even in the full-dependency job, without uninstalling anything.
    monkeypatch.setattr(builtins, "__import__", without_peft)
    with pytest.raises(ModuleNotFoundError, match="uv sync.*--extra qwen") as error:
        model_class(tmp_path, config={})
    assert error.value.name == "peft"

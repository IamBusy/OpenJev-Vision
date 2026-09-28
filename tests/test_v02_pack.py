import json
from collections import defaultdict
from pathlib import Path

import pytest

from openjev.io import read_records, sha256


def test_v02_frozen_groups_and_paired_reference():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/v02/processed"
    if not (data / "manifest.json").exists():
        pytest.skip("V0.2 local data has not been prepared")
    manifest = json.loads((data / "manifest.json").read_text())
    groups = defaultdict(set)
    all_rows = {}
    for split in ["train", "dev", "calibration", "test"]:
        path = data / f"{split}.jsonl"
        assert sha256(path) == manifest["splits"][split]["sha256"]
        rows = read_records(path)
        all_rows[split] = rows
        for r in rows:
            groups[r.group_id].add(split)
            assert r.provenance["qwen_input_tokens"] <= 768
            assert r.provenance["minilm_query_tokens"] <= 192
            assert r.source != "deepseek_synthetic"
    assert all(len(splits) == 1 for splits in groups.values())
    reference = read_records(data / "reference.jsonl")
    assert len(reference) == 64
    assert {r.id for r in reference} <= {r.id for r in all_rows["test"]}
    assert {r.group_id for r in reference}.isdisjoint({r.group_id for r in all_rows["train"]})
    assert len(all_rows["test"]) == 448
    assert {"arc_easy", "boolq", "rubric_en", "rubric_zh"} <= {r.task for r in all_rows["test"]}


def test_qwen_prompt_ignores_private_keys_and_enforces_label_budget():
    from openjev.qwen import prompt
    from openjev.schema import Question

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return json.dumps(messages)

    q = Question(
        type="choice",
        instructions="Choose.",
        criteria={"SECRET_INTERNAL_ID": "First option.", "ANOTHER_PRIVATE_ID": "Second option."},
    )
    rendered = prompt(Tokenizer(), "Input state", q)
    assert "SECRET_INTERNAL_ID" not in rendered
    assert "ANOTHER_PRIVATE_ID" not in rendered
    assert "A: First option." in rendered
    too_many = Question(
        type="choice", instructions="Choose.", criteria={str(i): f"Option {i}" for i in range(27)}
    )
    with pytest.raises(ValueError, match="26"):
        prompt(Tokenizer(), "state", too_many)

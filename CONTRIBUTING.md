# Contributing

OpenJev is a small research project. Issues and pull requests are welcome for
reproducibility failures, incorrect metrics, contract bugs and measured model
improvements. Include a minimal example, version, device and expected behavior.
Do not include API keys, private data or full model-service traces.

For the full contributor environment, install with
`uv sync --frozen --extra vision --extra dev --extra qwen`. Vision/data-only work
can use `uv sync --frozen --extra vision --extra dev`; Qwen model construction
requires the additional `qwen` extra. CI tests both installations.
Before a pull request:

```bash
uv run --no-sync ruff check src tests scripts marketing
uv run --no-sync ruff format --check src tests scripts marketing
uv run --no-sync pytest -q
uv run --no-sync python scripts/check_repository.py
uv build
```

Use focused tests for behavior changes. Keep expensive integration tests optional
and clearly skipped when artifacts are missing. Do not add network calls or paid
provider calls to tests. Keep generated weights, raw data and credentials out of
Git. Update the dependency lock when changing package requirements.

Research changes should predeclare data splits and selection criteria, retain
source/license provenance, and report regressions as well as improvements. Use
new holdouts for decisions made after inspecting previous tests. Compare the same
inputs, precision and hardware when making latency claims. Distinguish an
architecture change from a change to data or training budget.

Describe the concrete problem, final behavior and relevant validation in your
pull request. Contributions are provided under the repository's Apache-2.0
license; third-party material must retain its own applicable terms and attribution.
Be respectful, precise and open to correction in discussions.

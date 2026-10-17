# CLAUDE.md

Course project (AI4015, G2-09): predict SWE agent run success from OpenHands execution traces
(SWE-Gym trajectories). See `docs/architecture.md` for the design.

## Environment
- Python 3.11 venv at `.venv` (created with `uv venv --python 3.11`). Windows host.
- Install: `uv pip install --python .venv -r requirements.txt`
- Run all: `.venv/Scripts/python -m src.run_baseline` (add `--with-llm` for the LLM judge)
- Tests: `.venv/Scripts/python -m pytest -q`

## Rules
- Never use `test_result.report`, `test_result.test_output`, or `apply_patch_output` as features: they leak the label.
- Don't load the whole `messages` column. Stream row groups via `src/data/parser.iter_traces`.
- All CV uses the fold assignment in `data/eval_splits.parquet` (GroupKFold by `instance_id`).
- Module boundaries pass the Pydantic models in `src/schema.py`.
- `data/` and `.env` are gitignored. `results/*.json|png` are committed.
- Hardware: RTX 4050 6 GB, 16 GB RAM, no paid APIs.

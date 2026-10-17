# Architecture — Phase 1 Baseline

Project G2-09 — *Observability and Trust in Agentic AI for Software Engineering* (AI4015 Agentic AI).
Base paper: SWE-Gym (Pan et al., ICML 2025).

## Problem

Given the **execution trace** of a software-engineering agent (OpenHands, solving a SWE-Gym task),
predict whether the run **resolved** the task, i.e. whether its final patch passes the hidden tests.
The output is a calibrated probability `p(resolved | trace)`. A good predictor can serve as a trust
signal, rank candidate runs (best-of-n), or flag runs that need human review.

## Data

`SWE-Gym/OpenHands-Sampled-Trajectories`: 6,055 runs, about 300 MB of parquet in 3 shards.

| column | use |
|---|---|
| `instance_id` | task id, used as the **group key** for cross-validation |
| `run_id` | run id |
| `resolved` | **label** |
| `messages` | OpenHands trajectory in OpenAI chat format (system / user / assistant with `tool_calls` / tool observations) |
| `tools` | tool specs (`execute_bash`, `str_replace_editor`, `finish`) |
| `test_result.git_patch` | the agent's final patch, available at prediction time, so it can be used |
| `test_result.report`, `test_output` | **label leakage, never used as features** |

## Pipeline

```
 HF Hub ──download──► data/raw/*.parquet
                          │
          src/data/loader.py   (pyarrow, light columns only)
                          │──► data/runs_meta.parquet  + dataset stats
                          │
          src/data/parser.py   (streams row groups; messages never fully in RAM)
                          │──► schema.Trace (Pydantic)  ──► src/features.py
                          │                                   │
                          │                       data/features.parquet (schema.RunFeatures)
                          ▼                                   │
          src/eval/splits.py   GroupKFold(instance_id) ──► data/eval_splits.parquet
                                                              │
             ┌────────────────────────────┬──────────────────┼───────────────────────┐
             ▼                            ▼                  ▼                       ▼
  baselines/run_length.py     baselines/tabular.py   baselines/llm_judge.py   baselines/verifier_stub.py
  (#steps only)               (LogReg, GBM)          (50-run sample,          (SWE-Gym 32B verifier,
                                                      llm/provider.py)         Phase 2 on Kaggle)
             └────────────────────────────┴──────────────────┴───────────────────────┘
                                          │ schema.Prediction rows
                                          ▼
                          src/eval/metrics.py + report.py
                 pooled AUROC · within-task AUROC · Brier · ECE · reliability plot
                                          │
                                 results/*.json, results/*.png
```

`python -m src.run_baseline` runs every stage. Each stage reuses its cached output in `data/`
unless `--refresh` is passed.

## State / context structure (contracts in `src/schema.py`)

| model | role |
|---|---|
| `Step` | one agent turn: `index`, `role`, `action_type` (`view`/`edit`/`create`/`bash`/`finish`/`think`/`message`/`observation`), `tool_name`, `command`, `path`, `exit_code`, `is_error`, `tokens` (estimate) |
| `Trace` | `instance_id`, `run_id`, `resolved` (label, optional at inference), `steps: list[Step]`, `final_patch` |
| `RunFeatures` | the fixed-width feature vector extracted from a `Trace` |
| `JudgeVerdict` | structured LLM-judge output: `{p_success: float, rationale: str}`, validated from JSON |
| `Prediction` | `run_id`, `instance_id`, `method`, `fold`, `p_success`, `label` |
| `EvalReport` | metrics for one method, serialised to `results/metrics.json` |

Every module boundary passes these models (or DataFrames whose columns are exactly their fields),
so a stage can be swapped without touching its neighbours.

## Tool interfaces

There are two senses of "tool" here:

1. **Agent tools observed in traces.** The parser maps each OpenHands tool to an `action_type`:
   `execute_bash` → `bash` (with `exit_code` parsed from `[Command finished with exit code N]`),
   `str_replace_editor` → `view | create | edit` (by `command`), `finish` → `finish`.
   Errors are detected from `ERROR:` editor observations, non-zero exit codes, and runtime failures.
2. **Model/provider interface used by this system.** `src/llm/provider.py` defines an abstract
   `LLMProvider.complete(messages, json_mode) -> str` with two implementations:
   - `OllamaProvider`: local, default `qwen2.5-coder:7b` (fits in 6 GB VRAM at Q4)
   - `OpenAICompatProvider`: Groq or OpenRouter free tiers, with keys read from `.env`

   `get_provider()` picks the backend from `LLM_PROVIDER`. The SWE-Gym verifier sits behind a
   `Verifier` protocol (`baselines/verifier_stub.py`), so the Phase 2 Kaggle run can plug in.

## Features (Phase 1)

`n_steps`, `total_tokens`, `n_errors`, `n_repeated_commands`, `n_distinct_tools`,
`n_edits_without_view`, `ran_tests`, `patch_lines` (+ `patch_files`, `patch_empty`).
Token counts are estimated as `chars / 4` because the dataset has no usage metadata.

## Evaluation protocol

- **Splits:** `GroupKFold(n_splits=5)` on `instance_id`, so no task appears in both train and test.
  All methods use the same fold assignment.
- **Pooled AUROC:** over all out-of-fold predictions.
- **Within-task AUROC** (arXiv 2610.05572): computed only over *mixed* tasks (those with both
  resolved and unresolved runs). Each task's AUROC is weighted by `n_pass × n_fail`, which equals
  the pairwise accuracy over all same-task (pass, fail) pairs. This measures whether the predictor
  can **rank runs of the same task**, which is what best-of-n selection needs and what pooled AUROC
  conflates with task difficulty.
- **Calibration:** Brier score, 10-bin expected calibration error, reliability diagram.
- **Initial evaluation set:** the out-of-fold predictions on all 6,055 runs, plus a fixed 50-run
  stratified sample (seeded, drawn from mixed tasks) for the LLM judge. Its ids are stored in
  `data/eval_splits.parquet` and `results/llm_judge_sample.json` so later phases reuse the same set.

## Constraints

Laptop: RTX 4050 (6 GB), 16 GB RAM, zero budget. The parser streams parquet row groups and
reads only the columns it needs. The LLM judge truncates traces to a fixed character budget.
The 32B verifier is out of scope locally and will run on Kaggle in Phase 2.

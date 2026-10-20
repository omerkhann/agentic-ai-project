### all: 6055 runs, base rate 0.081, 188 mixed tasks / 1394 same-task pairs

| method | pooled AUROC | within-task AUROC | Brier | ECE |
|---|---|---|---|---|
| length_only | 0.702 | 0.809 | 0.0724 | 0.0188 |
| prompt_only | 0.640 | 0.500 | 0.0734 | 0.0196 |
| telemetry_lr | 0.839 | 0.854 | 0.0654 | 0.0049 |
| telemetry_gbm | 0.839 | 0.844 | 0.0670 | 0.0202 |

### judge_sample: 50 runs, base rate 0.500, 25 mixed tasks / 25 same-task pairs

| method | pooled AUROC | within-task AUROC | Brier | ECE |
|---|---|---|---|---|
| length_only | 0.752 | 0.720 | 0.3954 | 0.4206 |
| prompt_only | 0.500 | 0.500 | 0.4218 | 0.4120 |
| telemetry_lr | 0.789 | 0.800 | 0.3727 | 0.3989 |
| telemetry_gbm | 0.808 | 0.800 | 0.3597 | 0.3921 |
| llm_judge | 0.750 | 0.740 | 0.2594 | 0.2410 |

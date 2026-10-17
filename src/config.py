"""Central paths and constants. Everything else imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"

HF_DATASET = "SWE-Gym/OpenHands-Sampled-Trajectories"
HF_SHARDS = [f"data/train.raw-0000{i}-of-00003.parquet" for i in range(3)]

META_PATH = DATA_DIR / "runs_meta.parquet"          # light columns + labels
FEATURES_PATH = DATA_DIR / "features.parquet"       # telemetry features per run
SPLITS_PATH = DATA_DIR / "eval_splits.parquet"      # fold assignment + LLM-judge sample

N_FOLDS = 5
SEED = 42
LLM_JUDGE_SAMPLE = 50

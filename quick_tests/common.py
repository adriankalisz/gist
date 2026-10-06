import csv
import math
import os
import subprocess
from pathlib import Path

import torch
from datasets import load_dataset

# Same model and generation settings as server/worker/app/summarizer.py, so model-only
# numbers are comparable with the worker's
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
GENERATE_KWARGS = {"max_length": 142, "min_length": 56}

# The worker rejects texts longer than this (SummarizeRequest.text max_length)
MAX_DOCUMENT_CHARS = 10000

RESULTS_DIR = Path(__file__).parent / "results"


# Returns the first num_docs CNN/DailyMail test articles the worker accepts, as (article, highlights) pairs.
# Every script uses this, so all experiments see the same Documents in the same order.
def load_documents(num_docs: int) -> list[tuple[str, str]]:
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:2000]")
    documents = [(row["article"], row["highlights"]) for row in dataset if len(row["article"]) <= MAX_DOCUMENT_CHARS]
    if len(documents) < num_docs:
        raise ValueError(f"only {len(documents)} test articles fit within {MAX_DOCUMENT_CHARS} characters")
    return documents[:num_docs]


def parse_ints(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Environment columns written on every CSV row, so results from different machines/runs stay identifiable
def environment(commit: str | None = None) -> dict:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "commit": commit or git_commit(),
    }


# Writes rows to a CSV as they arrive (flushed per row), so a crash mid-run keeps everything measured so far
class CsvWriter:
    def __init__(self, path: str | os.PathLike, env: dict):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.env = env
        self.file = open(path, "w", newline="")
        self.writer = None

    def write(self, row: dict):
        row = {**row, **self.env}
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
        print(f"Wrote {self.path}")


# Nearest-rank percentile
def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]

"""AI-based meta scheduler that selects the best classical algorithm
based on a trained scikit-learn model.

The model is trained to predict which classical scheduler (fifo, sjf, priority, rr)
will yield the lowest average waiting time for a given workload. During runtime,
we compute workload-level features, ask the model for the best algorithm, and then
run that scheduler to produce the Gantt chart.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from typing import List, Tuple

import joblib

from .process import Process
from . import classical

GanttChart = List[Tuple[int, int, int]]

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "model.pkl"

# Feature calculation helpers --------------------------------------------------

_FEATURE_ORDER = [
    "n_proc",
    "avg_burst",
    "std_burst",
    "avg_arrival_gap",
]


def _workload_features(processes: List[Process]) -> list[float]:
    bursts = [p.burst_time for p in processes]
    arrivals = [p.arrival_time for p in processes]
    arrivals.sort()
    arrival_gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))] or [0]

    feats = {
        "n_proc": len(processes),
        "avg_burst": mean(bursts),
        "std_burst": stdev(bursts) if len(bursts) > 1 else 0.0,
        "avg_arrival_gap": mean(arrival_gaps),
    }
    return [feats[k] for k in _FEATURE_ORDER]


# Scheduler --------------------------------------------------------------------

_ALGO_MAP = {
    "fifo": classical.fifo,
    "sjf": classical.sjf,
    "priority": classical.priority_scheduling,
    "rr": classical.round_robin,
}


def meta_scheduler(
    processes: List[Process],
    *,
    model_path: str | Path | None = None,
    quantum: int = 4,
) -> GanttChart:
    """Select best classical algorithm using trained model and run it.

    Parameters
    ----------
    processes : list[Process]
        Workload to schedule.
    model_path : str | Path | None, default None
        Path to a joblib-saved scikit-learn classifier. If None, looks for
        `data/model.pkl` at repo root.
    quantum : int, default 4
        Time-slice to use when the predicted algorithm is Round-Robin.
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please train one via scripts/train_model.py"
        )

    clf = joblib.load(path)
    feat_vec = [_workload_features(processes)]
    pred_algo: str = clf.predict(feat_vec)[0]

    if pred_algo not in _ALGO_MAP:
        raise ValueError(f"Model predicted unknown algorithm '{pred_algo}'.")

    if pred_algo == "rr":
        return _ALGO_MAP[pred_algo](processes, quantum=quantum)
    return _ALGO_MAP[pred_algo](processes)

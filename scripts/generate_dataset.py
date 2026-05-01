#!/usr/bin/env python
"""Generate a CSV dataset for AI scheduler training.

Each row represents one simulated workload (set of processes). Features are aggregate
statistics of the workload; label is the classical algorithm that achieved the best
average waiting time.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import List

import sys
from pathlib import Path

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scheduler.process import ProcessGenerator, Process
from scheduler.simulator import run_simulation

_FEATURES = [
    "n_proc",
    "avg_burst",
    "std_burst",
    "avg_arrival_gap",
]
_LABEL = "best_algo"

_ALGOS = ["fifo", "sjf", "priority", "rr"]


def workload_features(processes: List[Process]) -> dict[str, float]:
    bursts = [p.burst_time for p in processes]
    arrivals = [p.arrival_time for p in processes]
    arrivals.sort()
    arrival_gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))] or [0]

    return {
        "n_proc": len(processes),
        "avg_burst": mean(bursts),
        "std_burst": stdev(bursts) if len(bursts) > 1 else 0.0,
        "avg_arrival_gap": mean(arrival_gaps),
    }


def pick_best_algo(processes: List[Process]) -> str:
    best_algo = None
    best_wait = float("inf")
    for algo in _ALGOS:
        kwargs = {"quantum": 4} if algo == "rr" else {}
        result = run_simulation(processes, algorithm=algo, **kwargs)
        if result.avg_waiting < best_wait:
            best_wait = result.avg_waiting
            best_algo = algo
    return best_algo  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset for AI scheduler")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of workloads to simulate")
    parser.add_argument("--out", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    gen = ProcessGenerator(seed=args.seed)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FEATURES + [_LABEL])
        writer.writeheader()
        for i in range(args.n_samples):
            n_proc = gen._rng.randint(20, 200)
            processes = gen.generate(n_proc)

            feats = workload_features(processes)
            label = pick_best_algo(processes)
            row = {**feats, _LABEL: label}
            writer.writerow(row)
            if (i + 1) % 100 == 0:
                print(f"Generated {i+1}/{args.n_samples}")

    print(f"Dataset saved to {args.out}")


if __name__ == "__main__":
    main()

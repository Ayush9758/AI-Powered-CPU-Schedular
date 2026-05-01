#!/usr/bin/env python
"""Compare AI scheduler against classical algorithms.

Example:
    python scripts/compare.py --model_path data/model.pkl --n_runs 100
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict

import sys
from pathlib import Path

# Ensure project root importable
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scheduler.process import ProcessGenerator
from scheduler.simulator import run_simulation

METRICS = ["avg_waiting", "avg_turnaround", "throughput"]
ALGORITHMS = ["fifo", "sjf", "priority", "rr", "ai"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheduler performance comparison")
    parser.add_argument("--model_path", type=Path, default=Path("data/model.pkl"))
    parser.add_argument("--n_runs", type=int, default=100, help="Number of random workloads")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = ProcessGenerator(seed=args.seed)

    # Accumulate metrics per algorithm
    acc: Dict[str, Dict[str, list[float]]] = {
        alg: {m: [] for m in METRICS} for alg in ALGORITHMS
    }

    for _ in range(args.n_runs):
        n_proc = gen._rng.randint(30, 150)
        procs = gen.generate(n_proc)

        for alg in ALGORITHMS:
            kwargs = {"quantum": 4} if alg == "rr" else {}
            if alg == "ai":
                kwargs["model_path"] = args.model_path
            res = run_simulation(procs, algorithm=alg, **kwargs)
            acc[alg]["avg_waiting"].append(res.avg_waiting)
            acc[alg]["avg_turnaround"].append(res.avg_turnaround)
            acc[alg]["throughput"].append(res.throughput)

    # Compute mean metrics
    print("\n=== Average Metrics over", args.n_runs, "runs ===")
    header = f"{'Algorithm':<10}  {'Wait':>8}  {'Turnaround':>11}  {'Throughput':>10}"
    print(header)
    print("-" * len(header))
    for alg in ALGORITHMS:
        w = mean(acc[alg]["avg_waiting"])
        ta = mean(acc[alg]["avg_turnaround"])
        thr = mean(acc[alg]["throughput"])
        print(f"{alg:<10}  {w:8.2f}  {ta:11.2f}  {thr:10.2f}")


if __name__ == "__main__":
    main()

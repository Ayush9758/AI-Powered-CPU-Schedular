#!/usr/bin/env python
"""CLI to run CPU scheduling simulation using classical algorithms.

Usage example:
    python scripts/run_simulation.py --algorithm rr --quantum 4 --n_procs 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scheduler.process import ProcessGenerator
from scheduler.simulator import run_simulation, list_algorithms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU Scheduling Simulation")
    parser.add_argument(
        "--algorithm",
        choices=list_algorithms(),
        default="fifo",
        help="Scheduling algorithm to use.",
    )
    parser.add_argument("--n_procs", type=int, default=50, help="Number of processes to generate")
    parser.add_argument("--quantum", type=int, default=4, help="Time quantum for Round Robin")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to save Gantt chart (pid,start,end)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gen = ProcessGenerator(seed=args.seed)
    processes = gen.generate(args.n_procs)

    algo_kwargs = {"quantum": args.quantum} if args.algorithm == "rr" else {}
    result = run_simulation(processes, algorithm=args.algorithm, **algo_kwargs)

    print("Simulation Summary:")
    print(result.summary())

    if args.output:
        import csv

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pid", "start", "end"])
            writer.writerows(result.chart)
        print(f"Gantt chart saved to {args.output}")


if __name__ == "__main__":
    main()

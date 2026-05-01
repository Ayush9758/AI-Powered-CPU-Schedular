"""Discrete-event CPU scheduling simulator.

Core entry point:
    run_simulation(processes, algorithm="fifo", **algo_kwargs)

Returns a SimulationResult containing the Gantt chart and key metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .process import Process
from . import classical  # noqa: F401 (expose classical schedulers)
from .ai import meta_scheduler as ai_scheduler
from . import metrics

GanttChart = List[Tuple[int, int, int]]  # (pid, start, end)


@dataclass
class SimulationResult:
    chart: GanttChart
    avg_waiting: float
    avg_turnaround: float
    throughput: float

    def summary(self) -> str:
        return (
            f"Avg waiting: {self.avg_waiting:.2f}, "
            f"Avg turnaround: {self.avg_turnaround:.2f}, "
            f"Throughput: {self.throughput:.2f} procs/unit"
        )


_ALGO_REGISTRY: Dict[str, Callable[..., GanttChart]] = {
    "ai": ai_scheduler,
    "fifo": classical.fifo,
    "sjf": classical.sjf,
    "priority": classical.priority_scheduling,
    "rr": classical.round_robin,
}


def list_algorithms() -> List[str]:
    return list(_ALGO_REGISTRY.keys())


def run_simulation(
    processes: List[Process],
    algorithm: str = "fifo",
    **algo_kwargs,
) -> SimulationResult:
    """Run the chosen scheduling algorithm and compute metrics."""
    alg = _ALGO_REGISTRY.get(algorithm.lower())
    if alg is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Valid: {list_algorithms()}")

    chart = alg(processes, **algo_kwargs) if algo_kwargs else alg(processes)

    avg_wait = metrics.average_waiting_time(processes, chart)
    avg_ta = metrics.average_turnaround_time(processes, chart)
    thr = metrics.throughput(chart)

    return SimulationResult(chart=chart, avg_waiting=avg_wait, avg_turnaround=avg_ta, throughput=thr)

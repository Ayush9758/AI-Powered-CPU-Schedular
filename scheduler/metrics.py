"""Metric calculation utilities for CPU scheduling simulations."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .process import Process

GanttChart = List[Tuple[int, int, int]]  # (pid, start, end)

__all__ = [
    "compute_completion_times",
    "average_waiting_time",
    "average_turnaround_time",
    "throughput",
]


def compute_completion_times(chart: GanttChart) -> Dict[int, int]:
    """Return completion time for each process from a Gantt chart."""
    completion: Dict[int, int] = {}
    for pid, _start, end in chart:
        completion[pid] = end  # last occurrence will be final completion
    return completion


def average_waiting_time(processes: List[Process], chart: GanttChart) -> float:
    completion = compute_completion_times(chart)
    waiting_sum = 0.0
    for p in processes:
        waiting_sum += completion[p.pid] - p.arrival_time - p.burst_time
    return waiting_sum / len(processes)


def average_turnaround_time(processes: List[Process], chart: GanttChart) -> float:
    completion = compute_completion_times(chart)
    ta_sum = 0.0
    for p in processes:
        ta_sum += completion[p.pid] - p.arrival_time
    return ta_sum / len(processes)


def throughput(chart: GanttChart) -> float:
    """Processes completed per unit time."""
    if not chart:
        return 0.0
    total_time = chart[-1][2]  # end time of last slice
    n_proc = len({pid for pid, _, _ in chart})
    return n_proc / total_time if total_time > 0 else 0.0

"""Process data model and random process generator utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

__all__ = ["Process", "ProcessGenerator"]


@dataclass(order=True)
class Process:
    sort_index: int = field(init=False, repr=False)
    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 0

    def __post_init__(self):
        # Used for default sorting by arrival_time then pid
        self.sort_index = (self.arrival_time, self.pid)


class ProcessGenerator:
    """Generates synthetic processes for simulation."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self, n: int, max_arrival: int = 100, max_burst: int = 20, max_priority: int = 10) -> List[Process]:
        processes: List[Process] = []
        for pid in range(1, n + 1):
            arrival = self._rng.randint(0, max_arrival)
            burst = self._rng.randint(1, max_burst)
            priority = self._rng.randint(0, max_priority)
            processes.append(Process(pid=pid, arrival_time=arrival, burst_time=burst, priority=priority))
        # Sort by arrival for convenience
        processes.sort(key=lambda p: (p.arrival_time, p.pid))
        return processes

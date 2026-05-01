"""Classical CPU scheduling algorithms for benchmarking.
Each algorithm returns a list of tuples (pid, start_time, end_time) representing
when each process ran on the CPU (Gantt chart representation).
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

from .process import Process

GanttChart = List[Tuple[int, int, int]]  # (pid, start, end)


def fifo(processes: List[Process]) -> GanttChart:
    """First In First Out (FCFS) non-preemptive scheduling."""
    time = 0
    chart: GanttChart = []
    for p in sorted(processes, key=lambda x: (x.arrival_time, x.pid)):
        if time < p.arrival_time:
            time = p.arrival_time  # CPU idle
        start = time
        end = start + p.burst_time
        chart.append((p.pid, start, end))
        time = end
    return chart


def sjf(processes: List[Process]) -> GanttChart:
    """Shortest Job First (non-preemptive)."""
    chart: GanttChart = []
    time = 0
    ready: List[Process] = []
    i = 0
    processes = sorted(processes, key=lambda p: (p.arrival_time, p.pid))
    n = len(processes)

    while len(chart) < n:
        # Add newly arrived processes to ready queue
        while i < n and processes[i].arrival_time <= time:
            ready.append(processes[i])
            i += 1
        if not ready:
            # No process is ready, jump to next arrival
            time = processes[i].arrival_time
            continue
        # Choose process with smallest burst
        ready.sort(key=lambda p: (p.burst_time, p.arrival_time, p.pid))
        p = ready.pop(0)
        start = time
        end = start + p.burst_time
        chart.append((p.pid, start, end))
        time = end
    return chart


def priority_scheduling(processes: List[Process]) -> GanttChart:
    """Priority scheduling (lower number ⇒ higher priority). Non-preemptive."""
    chart: GanttChart = []
    time = 0
    ready: List[Process] = []
    i = 0
    processes = sorted(processes, key=lambda p: (p.arrival_time, p.pid))
    n = len(processes)

    while len(chart) < n:
        while i < n and processes[i].arrival_time <= time:
            ready.append(processes[i])
            i += 1
        if not ready:
            time = processes[i].arrival_time
            continue
        ready.sort(key=lambda p: (p.priority, p.arrival_time, p.pid))
        p = ready.pop(0)
        start = time
        end = start + p.burst_time
        chart.append((p.pid, start, end))
        time = end
    return chart


def round_robin(processes: List[Process], quantum: int = 4) -> GanttChart:
    """Round Robin (preemptive) scheduling. Handles CPU idle gaps correctly."""
    # Sort by arrival for deterministic order
    processes = sorted(processes, key=lambda p: p.arrival_time)
    n = len(processes)
    time = 0
    i = 0  # index of next process that has not yet arrived
    queue: deque[Process] = deque()
    remaining = {p.pid: p.burst_time for p in processes}
    chart: GanttChart = []

    def add_arrivals(current_time: int):
        nonlocal i
        while i < n and processes[i].arrival_time <= current_time:
            queue.append(processes[i])
            i += 1

    # Simulation loop
    while queue or i < n:
        # If no process is ready, fast-forward time to next arrival
        if not queue:
            next_arrival = processes[i].arrival_time
            time = max(time, next_arrival)
            add_arrivals(time)
            continue

        p = queue.popleft()
        # Ensure CPU time starts not earlier than process arrival
        if time < p.arrival_time:
            time = p.arrival_time
        start = time
        run = min(quantum, remaining[p.pid])
        time += run
        remaining[p.pid] -= run
        chart.append((p.pid, start, time))

        # Add any processes that arrived during this time slice
        add_arrivals(time)

        # If the current process still has remaining burst, re-enqueue it
        if remaining[p.pid] > 0:
            queue.append(p)

    return chart

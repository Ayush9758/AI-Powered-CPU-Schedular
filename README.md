# AI-Powered Adaptive CPU Scheduler

This project implements an AI-driven CPU scheduler that learns workload patterns and dynamically selects the best process to run, improving waiting time, turnaround time, and throughput compared with classical algorithms.

## Directory Layout

```
scheduler/            # Core library code
    classical.py      # FIFO, SJF, Priority, RR implementations
    ai.py             # AI-based scheduler wrapper
    simulator.py      # Discrete-event simulation engine
    process.py        # Process data structure & generator
    metrics.py        # Metric collection utilities

notebooks/            # Optional Jupyter notebooks for exploration
scripts/              # CLI entry points (train, run-sim, dashboard)
data/                 # Logged training data & models
```

## Prerequisites

```
python>=3.9
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Generate a sample workload and run baseline algorithms:

```bash
python scripts/run_simulation.py --algorithm fifo --n_procs 1000
```

2. Train the AI model:

```bash
python scripts/train_model.py --dataset data/sim_logs.csv --model_path data/model.pkl
```

3. Evaluate AI vs. classical:

```bash
python scripts/compare.py --model_path data/model.pkl --n_procs 1000
```

4. Launch the Streamlit dashboard:

```bash
streamlit run scripts/dashboard.py
```

## Team Avengers
* Team Lead: <name>
* Member 2: <name>
* Member 3: <name>
* Member 4: <name>

---
Made with ❤️  for OS Project 2025

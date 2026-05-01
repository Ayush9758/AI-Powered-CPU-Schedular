#!/usr/bin/env python
"""Interactive Streamlit dashboard to showcase the AI‐powered CPU scheduler.

Run with:
    streamlit run scripts/dashboard.py

Features
--------
• Sidebar controls to configure a random workload (#processes, seed).
• Choose which algorithms to display (AI + classical).
• Gantt chart visualisation (Altair) of CPU time slices.
• Metrics table (avg waiting, turnaround, throughput).
• Bar chart comparing metrics across algorithms.
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

# Ensure project root importable when launched via `streamlit run`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scheduler.process import ProcessGenerator  # noqa: E402
from scheduler.simulator import run_simulation, list_algorithms  # noqa: E402

GanttChart = List[Tuple[int, int, int]]  # (pid, start, end)

ALT_THEMES = {
    "light": "none",
    "dark": "none",
}


def gantt_to_dataframe(chart: GanttChart, algorithm: str) -> pd.DataFrame:
    """Convert Gantt chart list to pandas DataFrame for plotting."""
    return pd.DataFrame(
        {
            "pid": [pid for pid, *_ in chart],
            "start": [start for _pid, start, _ in chart],
            "end": [end for _pid, _start, end in chart],
            "algorithm": algorithm,
        }
    )


def plot_gantt(df: pd.DataFrame, title: str) -> alt.Chart:
    """Return an Altair Gantt chart."""
    df = df.copy()
    df["duration"] = df["end"] - df["start"]
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="start:Q",
            x2="end:Q",
            y=alt.Y("pid:O", title="Process ID"),
            color=alt.Color("pid:O", legend=None),
            tooltip=["pid", "start", "end", "duration"],
        )
        .properties(title=title, height=400)
    )
    return chart


def plot_metric_bars(metric_dict: Dict[str, Dict[str, float]], metric: str) -> alt.Chart:
    data = pd.DataFrame(
        {
            "algorithm": list(metric_dict.keys()),
            metric: [metric_dict[alg][metric] for alg in metric_dict],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x="algorithm:O",
            y=f"{metric}:Q",
            color="algorithm:O",
            tooltip=["algorithm", metric],
        )
        .properties(height=300)
    )


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AI CPU Scheduler Dashboard", layout="wide")
st.title("🧠⚡ AI-Powered CPU Scheduler Demo")
st.markdown(
    "This interactive dashboard simulates random CPU workloads, compares classic "
    "scheduling algorithms with the AI-selected scheduler, and visualises results "
    "via Gantt charts and metrics."
)

with st.sidebar:
    st.header("⚙️ Simulation Settings")
    n_procs = st.slider("Number of processes", min_value=20, max_value=200, value=80)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    selected_algorithms = st.multiselect(
        "Algorithms to show",
        options=list_algorithms(),
        default=["ai", "sjf", "fifo"],
        help="AI meta-scheduler plus any classical ones you want to compare.",
    )
    quantum = st.slider("Round-Robin quantum (if selected)", 1, 10, 4)

    run_button = st.button("🚀 Run Simulation")

# When button pressed, perform simulation ---------------------------------------------------------

if run_button:
    gen = ProcessGenerator(seed=seed)
    processes = gen.generate(n_procs)

    results: Dict[str, Dict[str, float]] = {}
    all_gantt_segments: List[pd.DataFrame] = []

    for alg in selected_algorithms:
        kwargs = {"quantum": quantum} if alg == "rr" else {}
        if alg == "ai":
            # Simulator implicitly loads model from data/model.pkl
            pass
        sim_res = run_simulation(processes, algorithm=alg, **kwargs)

        # Metrics summary
        results[alg] = {
            "avg_waiting": sim_res.avg_waiting,
            "avg_turnaround": sim_res.avg_turnaround,
            "throughput": sim_res.throughput,
        }

        # Convert Gantt to DF for plotting
        all_gantt_segments.append(gantt_to_dataframe(sim_res.chart, alg))

    # Concatenate for easier multi-algorithm plotting
    df_gantt = pd.concat(all_gantt_segments, ignore_index=True)

    # Tabs for visuals ---------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Metrics", "📈 Gantt Chart"])

    with tab1:
        st.subheader("Average Metrics")
        st.dataframe(pd.DataFrame(results).T.style.format("{:.2f}"))

        st.markdown("### Metric Comparison")
        col1, col2, col3 = st.columns(3)
        col1.altair_chart(plot_metric_bars(results, "avg_waiting"), use_container_width=True)
        col2.altair_chart(plot_metric_bars(results, "avg_turnaround"), use_container_width=True)
        col3.altair_chart(plot_metric_bars(results, "throughput"), use_container_width=True)

    with tab2:
        # Show separate gantt per algorithm for clarity
        for alg in selected_algorithms:
            df_alg = df_gantt[df_gantt["algorithm"] == alg]
            st.markdown(f"#### {alg.upper()} Gantt Chart")
            st.altair_chart(plot_gantt(df_alg, title=f"{alg.upper()}"), use_container_width=True)

else:
    st.info("Configure parameters in the sidebar and click **Run Simulation** ☝️ to begin.")

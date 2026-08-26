"""Reusable Streamlit render functions for the QNAS dashboard: the queue
table, the live log tail, and the Plotly evolution/retrain charts.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qnas_dashboard import log_parsers

try:
    import generate_infographic as gi
except Exception:  # pragma: no cover - keeps the dashboard usable if the
    # (heavy, torch/cuda-importing) generate_infographic module fails to load.
    gi = None

STATUS_COLORS = {
    "queued": "#94a3b8",
    "running": "#3b82f6",
    "done": "#22c55e",
    "failed": "#ef4444",
    "stopped": "#f59e0b",
    "cancelled": "#a1a1aa",
}


def status_badge(status):
    color = STATUS_COLORS.get(status, "#94a3b8")
    return f'<span class="status-badge" style="background:{color}22;color:{color};">{status}</span>'


def inject_css():
    """One-time page styling: relies on inherited/currentColor and alpha-blended
    accent colors so it holds up in both light and dark Streamlit themes.
    """
    st.markdown("""
    <style>
    .block-container {
        padding-top: clamp(1rem, 2vw, 2rem);
        padding-bottom: clamp(1.5rem, 3vw, 3rem);
        padding-left: clamp(1rem, 4vw, 3rem);
        padding-right: clamp(1rem, 4vw, 3rem);
        max-width: 1400px;
    }

    .page-title {
        font-size: clamp(1.35rem, 1rem + 1.5vw, 1.9rem);
        font-weight: 700; margin: 0.2rem 0 1.2rem 0;
        letter-spacing: -0.01em;
    }

    /* Metric tiles: shrink label/value text at narrow widths instead of
    overflowing or truncating when several sit in one row (e.g. inside the
    dialog, which is much narrower than the full page). */
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        font-size: clamp(0.72rem, 0.6rem + 0.4vw, 0.85rem);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 0.9rem + 0.8vw, 1.5rem);
        overflow-wrap: anywhere;
    }

    /* Streamlit only stacks st.columns at a fixed browser-viewport
    breakpoint, not a container one - inside the (narrower) dialog that
    leaves columns cramped rather than stacked even on a wide screen. Query
    the page/dialog width itself (a container query has to look at an
    ancestor, not the horizontal block it restyles) and stack columns once
    that ancestor gets tight. */
    .block-container, div[data-testid="stDialog"] { container-type: inline-size; }
    @container (max-width: 460px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            width: 100% !important;
        }
    }

    .status-badge {
        padding: 3px 12px; border-radius: 999px; font-weight: 600;
        font-size: 0.8em; letter-spacing: 0.02em; text-transform: uppercase;
        white-space: nowrap;
    }

    .qc-row {
        display: flex; align-items: center; justify-content: space-between;
        padding: 3px 0;
    }
    .qc-count { font-weight: 600; opacity: 0.85; }

    .page-indicator {
        text-align: center; padding-top: 0.4rem; opacity: 0.75; font-size: 0.9em;
    }

    div[data-testid="stMetric"] {
        background: color-mix(in srgb, currentColor 5%, transparent);
        border-radius: 10px; padding: 0.7rem 0.9rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    button[kind="secondary"] { border-radius: 8px; }

    div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }

    /* Job list rows: st.button itself is the whole clickable row (a real,
    guaranteed-clickable widget - no absolutely-positioned overlay hacks,
    which turned out to look fine but silently swallowed clicks in the real
    browser). The rich look comes from the Markdown st.button supports
    natively in its label (bold / inline code / emoji), plus CSS on the
    button element scoped via the key's st-key- class. */
    div[class*="st-key-job_row_"] button {
        text-align: left; justify-content: flex-start;
        white-space: normal; word-break: break-word; line-height: 1.6;
        background: transparent; border: 1px solid transparent;
        border-radius: 10px; padding: 10px 16px; width: 100%;
        margin-bottom: 4px; font-size: 0.95em;
    }
    div[class*="st-key-job_row_"] button:hover {
        background: color-mix(in srgb, currentColor 6%, transparent);
        border-color: color-mix(in srgb, currentColor 12%, transparent);
        color: inherit;
    }
    div[class*="st-key-job_row_"] button p { margin: 0; }
    div[class*="st-key-job_row_"] button code {
        background: color-mix(in srgb, currentColor 8%, transparent);
        font-size: 0.92em;
    }
    </style>
    """, unsafe_allow_html=True)


MODE_ICONS = {
    "evolve": "🧬",
    "retrain": "🔁",
    "pipeline": "🔗",
}


def _row_label(job):
    icon = MODE_ICONS.get(job["mode"], "⚙️")
    dot = STATUS_DOTS.get(job["status"], "⚪")
    config_name = Path(job["config_path"]).name
    started = (job["started_at"] or "").replace("T", " ")[:16]

    top = f"{icon} **Job #{job['id']}**  ·  {job['mode']}  ·  {dot} {job['status']}"
    meta = f"`{config_name}` → `{job['experiment_path']}`"
    if started:
        meta += f"  ·  started {started}"
    return f"{top}\n\n{meta}"


STATUS_DOTS = {
    "queued": "⚪",
    "running": "🔵",
    "done": "🟢",
    "failed": "🔴",
    "stopped": "🟠",
    "cancelled": "⚫",
}


def render_queue_rows(jobs):
    """Renders every job as one full-width clickable row - no checkbox
    column, no fixed-width columns (those don't survive responsive layouts).
    Each row is a single real st.button (guaranteed clickable everywhere,
    unlike a CSS overlay trick) whose label uses st.button's native Markdown
    support (bold / inline code / emoji) for a two-line, card-like look.
    Returns the id of the job whose row was clicked *this run*, or None.
    """
    if not jobs:
        st.info("Queue is empty.")
        return None

    clicked_id = None
    for job in jobs:
        if st.button(_row_label(job), key=f"job_row_{job['id']}", width='stretch'):
            clicked_id = job["id"]
    return clicked_id


def render_log_tail(log_path, state_key, max_chars=40000, height=420):
    if not log_path or not Path(log_path).exists():
        st.caption("No log file yet.")
        return

    text_key = f"{state_key}_text"
    offset_key = f"{state_key}_offset"
    if text_key not in st.session_state:
        st.session_state[text_key] = ""
        st.session_state[offset_key] = 0

    new_text, new_offset = log_parsers._read_new_text(log_path, st.session_state[offset_key])
    if new_text:
        st.session_state[text_key] += new_text
        st.session_state[offset_key] = new_offset
        if len(st.session_state[text_key]) > max_chars:
            st.session_state[text_key] = st.session_state[text_key][-max_chars:]

    st.text_area(
        "log", value=st.session_state[text_key], height=height,
        label_visibility="collapsed", key=f"{state_key}_area",
    )


def _accumulate(state_key, parse_fn, path):
    off_key = f"{state_key}_offset"
    data_key = f"{state_key}_data"
    if off_key not in st.session_state:
        st.session_state[off_key] = 0
        st.session_state[data_key] = []
    new_records, new_offset = parse_fn(path, st.session_state[off_key])
    st.session_state[off_key] = new_offset
    st.session_state[data_key].extend(new_records)
    return st.session_state[data_key]


def render_evolution_chart(log_qnas_path, state_key):
    if not log_qnas_path or not Path(log_qnas_path).exists():
        st.caption("No log_QNAS.txt yet - evolution hasn't produced a generation summary.")
        return

    records = _accumulate(state_key, log_parsers.parse_qnas_generations, log_qnas_path)
    if not records:
        st.caption("Waiting for the first generation to finish...")
        return

    gens = [r["generation"] for r in records]
    best = [r["best_fitness"] for r in records]
    stages = [r["stage"] for r in records]

    fig = go.Figure()
    for r in records:
        fig.add_trace(go.Box(
            x=[r["generation"]] * len(r["fitnesses"]), y=r["fitnesses"],
            marker_color="#94a3b8", showlegend=False, boxpoints=False,
            width=0.6, line_width=1,
        ))
    fig.add_trace(go.Scatter(
        x=gens, y=best, mode="lines+markers", name="Best fitness",
        line=dict(color="#3b82f6", width=3), marker=dict(size=6),
    ))

    stage_changes = [gens[0]] + [g for g, s, ps in zip(gens[1:], stages[1:], stages[:-1]) if s != ps]
    for g in stage_changes[1:]:
        fig.add_vline(x=g - 0.5, line_dash="dot", line_color="#f59e0b", opacity=0.6)

    fig.update_layout(
        template="plotly_white", height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Generation", yaxis_title="Fitness",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch', key=f"{state_key}_fig")

    latest = records[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generation", latest["generation"])
    c2.metric("Stage", latest["stage"])
    c3.metric("Best fitness", f"{max(best):.2f}")
    c4.metric("Architectures found", sum(r["discovered"] for r in records))


def render_retrain_chart(retrain_log_path, state_key):
    if not retrain_log_path or not Path(retrain_log_path).exists():
        st.caption("No retrain.log yet.")
        return

    off_key = f"{state_key}_offset"
    epochs_key = f"{state_key}_epochs"
    tests_key = f"{state_key}_tests"
    if off_key not in st.session_state:
        st.session_state[off_key] = 0
        st.session_state[epochs_key] = []
        st.session_state[tests_key] = []

    new_epochs, new_tests, new_offset = log_parsers.parse_retrain_epochs(
        retrain_log_path, st.session_state[off_key])
    st.session_state[off_key] = new_offset
    st.session_state[epochs_key].extend(new_epochs)
    st.session_state[tests_key].extend(new_tests)

    epochs = st.session_state[epochs_key]
    if not epochs:
        st.caption("Waiting for the first retrain checkpoint...")
        return

    df = pd.DataFrame(epochs)
    fig = go.Figure()
    for exp_name, sub in df.groupby("experiment"):
        label = Path(exp_name).name
        fig.add_trace(go.Scatter(x=sub["epoch"], y=sub["val_acc"], mode="lines+markers",
                                  name=f"{label} val acc"))
    fig.update_layout(
        template="plotly_white", height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Epoch", yaxis_title="Validation accuracy (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch', key=f"{state_key}_acc_fig")

    fig2 = go.Figure()
    for exp_name, sub in df.groupby("experiment"):
        label = Path(exp_name).name
        fig2.add_trace(go.Scatter(x=sub["epoch"], y=sub["train_loss"], mode="lines",
                                   name=f"{label} train loss", line=dict(dash="dot")))
        fig2.add_trace(go.Scatter(x=sub["epoch"], y=sub["val_loss"], mode="lines",
                                   name=f"{label} val loss"))
    fig2.update_layout(
        template="plotly_white", height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Epoch", yaxis_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig2, width='stretch', key=f"{state_key}_loss_fig")

    if st.session_state[tests_key]:
        st.dataframe(pd.DataFrame(st.session_state[tests_key]), hide_index=True,
                     width='stretch')


def render_train_individuals(train_log_path, state_key, max_records=30):
    if not train_log_path or not Path(train_log_path).exists():
        return
    records = _accumulate(state_key, log_parsers.parse_train_individuals, train_log_path)
    if not records:
        return
    df = pd.DataFrame(records[-max_records:])
    st.dataframe(df, hide_index=True, width='stretch')


def render_search_summary(experiment_path):
    """Renders the same information generate_infographic.py draws into
    infographic.png, but as native Streamlit widgets (metrics/tables) instead
    of a static image.
    """
    if gi is None:
        st.caption("Search summary is unavailable (generate_infographic failed to import).")
        return

    experiment_path = str(experiment_path)
    gen_data = gi.load_generation_data(experiment_path)
    if not gen_data:
        st.caption("No data_QNAS.pkl yet - the search summary appears once generations complete.")
        return

    best_net = gi.load_best_network(experiment_path, gen_data)
    indiv_df = gi.load_individuals(experiment_path)
    stage_ranges = gi.stage_generation_ranges(gen_data)
    stage_nets = gi.load_best_networks_per_stage(gen_data, stage_ranges)
    retrain_runs = gi.load_retrain_runs(experiment_path)
    run_config = gi.load_run_config(experiment_path)
    search_time_s = gi.actual_running_time_seconds(gen_data)
    cache_stats = gi.load_cache_stats(experiment_path, gen_data)
    num_gpus = len((run_config.get("train", {}) or {}).get("available_gpus") or [1])

    has_best_acc = not indiv_df.empty and indiv_df["best_accuracy"].notna().any()
    best_acc = indiv_df["best_accuracy"].max() if has_best_acc else None
    best_params = indiv_df.loc[indiv_df["best_accuracy"].idxmax(), "total_trainable_params"] \
        if has_best_acc else None

    retrain_accs = [r["accuracy"] for r in retrain_runs]
    retrain_durations = [r["duration_s"] for r in retrain_runs if r.get("duration_s") is not None]
    mean_retrain_acc = sum(retrain_accs) / len(retrain_accs) if retrain_accs else None
    mean_retrain_time_s = sum(retrain_durations) / len(retrain_durations) if retrain_durations else None
    gpu_days = (search_time_s * num_gpus / 86400) if search_time_s is not None else None

    # A dialog is much narrower than the full page, so these use a 2-wide
    # grid rather than 4-across - st.columns only auto-stacks at a browser
    # viewport breakpoint, not a container one, so 4 columns stay cramped
    # here even on a wide screen.
    st.markdown("**Run summary**")
    metrics = [
        ("Best acc (search)", f"{best_acc:.2f}" if best_acc is not None else "—"),
        ("Mean acc (retrain)", f"{mean_retrain_acc:.2f}" if mean_retrain_acc is not None else "—"),
        ("Best model size", f"{best_params / 1e6:.2f}M" if best_params else "—"),
        ("Generations run", len(gen_data)),
        ("Search time", gi._format_hours(search_time_s)),
        ("Mean retrain time", gi._format_hours(mean_retrain_time_s)),
        ("GPU-days (search)", f"{gpu_days:.2f}" if gpu_days is not None else "—"),
        ("Individuals evaluated", gi._total_individuals_evaluated(gen_data)),
    ]
    for i in range(0, len(metrics), 2):
        cols = st.columns(2)
        for col, (label, value) in zip(cols, metrics[i:i + 2]):
            col.metric(label, value)

    if cache_stats:
        st.markdown("**Architecture cache**")
        c = st.columns(3)
        c[0].metric("Distinct architectures", cache_stats["distinct"])
        c[1].metric("Cache hits", cache_stats["total_hits"])
        hit_ratio = cache_stats["hit_ratio"]
        c[2].metric("Hit ratio", f"{hit_ratio * 100:.1f}%" if hit_ratio is not None else "—")

    if stage_nets:
        st.markdown("**Best network per progressive stage**")
        rows = [{
            "stage": s["stage_idx"], "generations": f"{s['gen_start']}-{s['gen_end']}",
            "best generation": s["generation"], "fitness": round(s["best_accuracy"], 2),
            "net_list": " → ".join(s["net_list"]),
        } for s in stage_nets]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    elif best_net and best_net.get("net_list"):
        st.markdown("**Best network found**")
        gen, idx = best_net.get("best_so_far_id", (None, None))
        fitness = best_net.get("best_so_far")
        st.caption(f"generation {gen}, individual {idx}"
                   + (f" — fitness {fitness:.2f}" if fitness is not None else ""))
        st.code(" → ".join(best_net["net_list"]))

    if retrain_runs:
        st.markdown("**Search vs. retrain accuracy**")
        rows = [{"run": "best in search",
                  "accuracy": round(best_acc, 2) if best_acc is not None else None,
                  "duration": "—"}]
        for r in retrain_runs:
            rows.append({
                "run": r["label"], "accuracy": round(r["accuracy"], 2),
                "duration": gi._format_hours(r.get("duration_s")),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

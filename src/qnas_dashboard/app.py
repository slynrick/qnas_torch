"""Streamlit dashboard for the qnas_torch experiment queue: live queue state,
per-job log tailing, and iterative evolution/retrain progress charts.

Run with: scripts/run_dashboard.sh  (wraps `streamlit run src/qnas_dashboard/app.py`)
"""

from pathlib import Path

import streamlit as st

from qnas_dashboard import components, data

st.set_page_config(page_title="QNAS Queue Dashboard", page_icon="🧬", layout="wide")
components.inject_css()

with st.sidebar:
    st.markdown("### 🧬 QNAS Dashboard")
    REFRESH_SECONDS = st.slider("Auto-refresh (s)", 3, 30, 5)
    st.divider()
    st.markdown("#### Worker")


@st.fragment(run_every=REFRESH_SECONDS)
def worker_panel():
    worker, current = data.get_worker()
    alive = bool(worker and worker.get("alive"))
    with st.sidebar:
        dot = "🟢" if alive else "🔴"
        st.markdown(f"{dot} **{'Running' if alive else 'Stopped'}**"
                     + (f" · pid {worker['pid']}" if alive else ""))
        if current:
            st.caption(f"Working on #{current['id']} · {current['mode']} · "
                       f"{Path(current['config_path']).name}")

        col1, col2 = st.columns(2)
        if col1.button("▶ Start", disabled=alive, width='stretch'):
            ok, msg = data.start_worker()
            (st.success if ok else st.error)(msg)
            st.rerun(scope="app")
        if col2.button("■ Stop", disabled=not alive, width='stretch'):
            ok, msg = data.stop_worker()
            (st.success if ok else st.error)(msg)
            st.rerun(scope="app")

        counts = data.status_counts()
        st.divider()
        st.markdown("#### Queue counts")
        for status in ("queued", "running", "done", "failed", "stopped", "cancelled"):
            n = counts.get(status, 0)
            st.markdown(
                f'<div class="qc-row">{components.status_badge(status)}'
                f'<span class="qc-count">{n}</span></div>',
                unsafe_allow_html=True,
            )


worker_panel()

st.markdown('<div class="page-title">🧬 QNAS Experiment Queue</div>', unsafe_allow_html=True)


@st.dialog("Job details", width="large")
def show_job_dialog(job_id):
    job = data.get_job(job_id)
    if job is None:
        st.warning("This job no longer exists.")
        return

    header_cols = st.columns([3, 2], vertical_alignment="center")
    header_cols[0].markdown(f"#### Job #{job['id']} — {job['mode']}")
    header_cols[1].markdown(
        f'<div style="text-align:right">{components.status_badge(job["status"])}</div>',
        unsafe_allow_html=True,
    )

    # A dialog is much narrower than the full page, so metadata/actions use a
    # 2-wide grid here rather than 4-across - st.columns only auto-stacks at
    # a *browser-viewport* breakpoint, not a container one, so 4 columns
    # packed into a dialog stay cramped even on a wide screen.
    meta_cols = st.columns(2)
    meta_cols[0].markdown(f"**Config**<br>{Path(job['config_path']).name}", unsafe_allow_html=True)
    meta_cols[1].markdown(f"**Started**<br>{job['started_at'] or '—'}", unsafe_allow_html=True)
    meta_cols = st.columns(2)
    meta_cols[0].markdown(f"**Experiment path**<br>{job['experiment_path']}", unsafe_allow_html=True)
    meta_cols[1].markdown(f"**Finished**<br>{job['finished_at'] or '—'}", unsafe_allow_html=True)
    if job["status"] == "failed" and job.get("error_message"):
        st.error(job["error_message"])
    if job.get("exit_code") is not None:
        st.caption(f"Exit code: {job['exit_code']}")

    action_cols = st.columns(2)
    if action_cols[0].button("Cancel (queued only)", disabled=job["status"] != "queued",
                              width='stretch'):
        ok, msg = data.cancel_job(job["id"])
        (st.success if ok else st.error)(msg)
        st.rerun()
    if action_cols[1].button("Retry", disabled=job["status"] not in ("failed", "stopped", "cancelled"),
                              width='stretch'):
        ok, msg = data.retry_job(job["id"])
        (st.success if ok else st.error)(msg)
        st.rerun()
    action_cols = st.columns(2)
    if action_cols[0].button("Remove", disabled=job["status"] == "running", width='stretch'):
        ok, msg = data.remove_job(job["id"])
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()
    if action_cols[1].button("Stop worker (kills this job)", disabled=job["status"] != "running",
                              width='stretch'):
        ok, msg = data.stop_worker()
        (st.success if ok else st.error)(msg)
        st.rerun()

    experiment_path = Path(job["experiment_path"])
    if not experiment_path.is_absolute():
        from qnas_queue.db import PROJECT_ROOT
        experiment_path = PROJECT_ROOT / experiment_path

    tabs = st.tabs(["📜 Live log", "🧬 Evolution progress", "📈 Retrain progress", "📊 Search summary"])

    with tabs[0]:

        @st.fragment(run_every=REFRESH_SECONDS)
        def log_tab():
            components.render_log_tail(job["log_path"], state_key=f"job_{job['id']}_log")

        log_tab()

    with tabs[1]:

        @st.fragment(run_every=REFRESH_SECONDS)
        def evolution_tab():
            log_qnas_path = experiment_path / "log_QNAS.txt"
            components.render_evolution_chart(str(log_qnas_path), state_key=f"job_{job['id']}_evo")
            st.markdown("**Current generation individuals**")
            components.render_train_individuals(
                str(experiment_path / "train.log"), state_key=f"job_{job['id']}_train")

        evolution_tab()

    with tabs[2]:

        @st.fragment(run_every=REFRESH_SECONDS)
        def retrain_tab():
            components.render_retrain_chart(
                str(experiment_path / "retrain.log"), state_key=f"job_{job['id']}_retrain")

        retrain_tab()

    with tabs[3]:

        @st.fragment(run_every=REFRESH_SECONDS)
        def summary_tab():
            components.render_search_summary(experiment_path)
            params_path = experiment_path / "log_params_evolution.txt"
            with st.expander("Raw run config (log_params_evolution.txt)"):
                if params_path.exists():
                    st.text(params_path.read_text())
                else:
                    st.caption("No log_params_evolution.txt yet.")

        summary_tab()


@st.fragment(run_every=REFRESH_SECONDS)
def queue_section():
    jobs = data.list_jobs()
    with st.container(border=True):
        st.markdown("#### Jobs")
        clicked_id = components.render_queue_rows(jobs)
        if clicked_id is not None:
            show_job_dialog(clicked_id)


queue_section()

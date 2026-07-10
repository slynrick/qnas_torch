""" Generate a single-page infographic summarizing a Q-NAS search (and, if present,
    a retrain phase) run: fitness over generations, the best network found, and
    architecture-cache reuse rates.
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

from util import load_pkl, load_yaml

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+")
_HHMM_RE = re.compile(r"(\d+)h:?(\d+)m")

# --- palette -----------------------------------------------------------
BG = "#0f172a"
PANEL_BG = "#16213c"
INK = "#e8ecf5"
MUTED = "#8b96b3"
GRID = "#2a3557"
ACCENT = "#5b8cff"
ACCENT_2 = "#5be0c0"
ACCENT_3 = "#ffb454"
BAD = "#ff6b6b"

OP_COLORS = {
    "conv": "#5b8cff",
    "pool": "#5be0c0",
    "no_op": "#3a4568",
}


def _op_color(op_name: str) -> str:
    if op_name == "no_op":
        return OP_COLORS["no_op"]
    if "pool" in op_name:
        return OP_COLORS["pool"]
    return OP_COLORS["conv"]


def _style_axes(ax, title=None):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)


def load_generation_data(experiment_path):
    data_path = os.path.join(experiment_path, "data_QNAS.pkl")
    if not os.path.exists(data_path):
        return None
    return load_pkl(data_path)


def load_best_network(experiment_path, gen_data):
    """ The best individual is already a concrete, discrete network (no MixedOp
    "collapsing" step needed) - its net_list lives in its own training_params.txt,
    at the folder pointed to by the last generation's best_so_far_id.
    """
    if not gen_data:
        return None
    last_gen = max(gen_data.keys())
    best_so_far_id = gen_data[last_gen].get("best_so_far_id")
    if not best_so_far_id:
        return None
    gen, idx = best_so_far_id
    params_path = os.path.join(experiment_path, f"{gen}_{idx}", "training_params.txt")
    if not os.path.exists(params_path):
        return None
    params = load_yaml(params_path)
    net_list = params.get("net_list")
    if not net_list:
        return None
    return {
        "net_list": net_list,
        "best_so_far": gen_data[last_gen].get("best_so_far"),
        "best_so_far_id": best_so_far_id,
    }


def load_individuals(experiment_path):
    rows = []
    for params_file in sorted(glob.glob(os.path.join(experiment_path, "*_*", "training_params.txt"))):
        try:
            params = load_yaml(params_file)
        except Exception:
            continue
        rows.append({
            "generation": int(params.get("generation", -1)),
            "individual": int(params.get("individual", -1)),
            "best_accuracy": params.get("best_accuracy"),
            "best_validation_loss": params.get("best_validation_loss"),
            "total_trainable_params": params.get("total_trainable_params"),
            "total_flops": params.get("total_flops"),
            "cuda_inference_time": params.get("cuda_inference_time"),
            "training_time": params.get("training_time"),
            "weight_reuse_applied": bool(params.get("weight_reuse_applied", False)),
        })
    return pd.DataFrame(rows)


def load_retrain_results(experiment_path):
    for path in glob.glob(os.path.join(experiment_path, "retrain_results_*.txt")):
        with open(path) as f:
            try:
                return json.load(f)
            except Exception:
                continue
    return None


def _parse_hhmm_seconds(time_str):
    if not time_str:
        return None
    m = _HHMM_RE.match(time_str)
    if not m:
        return None
    hours, minutes = int(m.group(1)), int(m.group(2))
    return hours * 3600 + minutes * 60


def search_wall_time_seconds(experiment_path, gen_data):
    """Real wall-clock duration of the search, from the first log line's
    timestamp to the last generation's recorded time (both in log_QNAS.txt /
    data_QNAS.pkl). Falls back to first-generation-to-last-generation if the
    log file is missing, since that's still an observed timestamp span.
    """
    if not gen_data:
        return None

    start_dt = None
    log_path = os.path.join(experiment_path, "log_QNAS.txt")
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                first_line = f.readline()
            m = _TIMESTAMP_RE.search(first_line)
            if m:
                start_dt = datetime.fromisoformat(m.group(0))
        except Exception:
            start_dt = None

    last_gen = max(gen_data.keys())
    end_str = gen_data[last_gen].get("time")
    if not end_str:
        return None
    try:
        end_dt = datetime.fromisoformat(end_str)
    except Exception:
        return None

    if start_dt is None:
        first_gen = min(gen_data.keys())
        first_str = gen_data[first_gen].get("time")
        if not first_str:
            return None
        try:
            start_dt = datetime.fromisoformat(first_str)
        except Exception:
            return None

    return (end_dt - start_dt).total_seconds()


def load_retrain_runs(experiment_path):
    """Per-repetition retrain accuracy/duration, read from retrain_*/ dirs
    (best_accuracy.txt + retraining_params.txt's t0/t1). Falls back to the
    legacy aggregate retrain_results_*.txt format if no per-run dirs exist.
    """
    runs = []
    for acc_path in sorted(glob.glob(os.path.join(experiment_path, "retrain_*", "best_accuracy.txt"))):
        run_dir = os.path.dirname(acc_path)
        try:
            accuracy = load_yaml(acc_path).get("best_accuracy")
        except Exception:
            accuracy = None
        if accuracy is None:
            continue

        duration_s = None
        params_path = os.path.join(run_dir, "retraining_params.txt")
        if os.path.exists(params_path):
            try:
                params = load_yaml(params_path)
                t0, t1 = params.get("t0"), params.get("t1")
                if t0 is not None and t1 is not None:
                    duration_s = float(t1) - float(t0)
            except Exception:
                duration_s = None

        runs.append({
            "label": os.path.basename(run_dir),
            "accuracy": float(accuracy),
            "duration_s": duration_s,
        })

    if runs:
        return runs

    aggregate = load_retrain_results(experiment_path)
    if not aggregate:
        return []
    for run_name, res in aggregate.items():
        acc = res.get("test_accuracy", res.get("best_accuracy"))
        if acc is None:
            continue
        runs.append({
            "label": run_name,
            "accuracy": float(acc),
            "duration_s": _parse_hhmm_seconds(res.get("time")),
        })
    return runs


def _format_hours(seconds):
    if seconds is None or seconds != seconds:
        return "—"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def load_run_config(experiment_path):
    log_path = os.path.join(experiment_path, "log_params_evolution.txt")
    if not os.path.exists(log_path):
        return {}
    try:
        return load_yaml(log_path)
    except Exception:
        return {}


# --- panels --------------------------------------------------------------

def _total_individuals_evaluated(gen_data):
    """Cumulative count of individuals evaluated, from data_QNAS.pkl's running total_eval.

    Per-individual directories on disk can be pruned by the search (e.g. after weight
    reuse/collapse), so this must come from the generation log, not a glob of dirs.
    """
    if not gen_data:
        return 0
    last_gen = max(gen_data.keys())
    return int(gen_data[last_gen].get("total_eval", 0))


def draw_header(fig, gs_row, experiment_path, gen_data, indiv_df, run_config, search_time_s):
    ax = fig.add_subplot(gs_row)
    ax.set_facecolor(BG)
    ax.axis("off")

    name = os.path.basename(os.path.normpath(experiment_path))
    n_generations = len(gen_data) if gen_data else 0
    n_individuals = _total_individuals_evaluated(gen_data) or len(indiv_df)

    if search_time_s is not None:
        time_label = _format_hours(search_time_s)
    else:
        avg_time = indiv_df["training_time"].mean() if not indiv_df.empty else np.nan
        if avg_time == avg_time and n_individuals:
            # Per-individual dirs may have been pruned from disk (e.g. weight reuse
            # cleanup), so scale the observed average up to the full evaluated count.
            time_label = f"~{_format_hours(avg_time * n_individuals)} (estimated)"
        else:
            time_label = "—"

    ax.text(0, 0.75, f"Q-NAS Search Report — {name}", color=INK, fontsize=22,
            fontweight="bold", transform=ax.transAxes, va="top")
    subtitle = f"Generations: {n_generations}   |   Individuals evaluated: {n_individuals}   |   Search time: {time_label}"
    ax.text(0, 0.25, subtitle, color=MUTED, fontsize=11, transform=ax.transAxes, va="top")


def draw_fitness_curve(fig, gs_cell, gen_data):
    ax = fig.add_subplot(gs_cell)
    if not gen_data:
        _style_axes(ax, "Fitness over generations")
        ax.text(0.5, 0.5, "no data_QNAS.pkl found", color=MUTED, ha="center", va="center",
                transform=ax.transAxes)
        return
    _style_axes(ax, "Fitness over generations")

    gens = sorted(gen_data.keys())
    best_so_far = [gen_data[g]["best_so_far"] for g in gens]

    for g in gens:
        fitnesses = np.asarray(gen_data[g]["fitnesses"], dtype=float)
        jitter = (np.random.rand(len(fitnesses)) - 0.5) * 0.3
        ax.scatter(np.full_like(fitnesses, g) + jitter, fitnesses, color=ACCENT_2,
                   alpha=0.5, s=22, zorder=2, label="population" if g == gens[0] else None)

    ax.plot(gens, best_so_far, color=ACCENT, linewidth=2.4, marker="o", markersize=5,
            zorder=3, label="best so far")
    ax.set_xlabel("generation", color=MUTED, fontsize=9)
    ax.set_ylabel("fitness", color=MUTED, fontsize=9)
    ax.set_xticks(gens)
    legend = ax.legend(loc="lower right", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(MUTED)


def draw_pareto_scatter(fig, gs_cell, indiv_df):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Accuracy vs. model size")
    if indiv_df.empty or indiv_df["total_trainable_params"].isna().all():
        ax.text(0.5, 0.5, "no per-individual data found", color=MUTED, ha="center", va="center",
                transform=ax.transAxes)
        return

    df = indiv_df.dropna(subset=["total_trainable_params", "best_accuracy"])
    sizes = 40 + 200 * (df["cuda_inference_time"].fillna(0) / max(df["cuda_inference_time"].max(), 1))
    ax.scatter(df["total_trainable_params"] / 1e6, df["best_accuracy"], s=sizes,
               c=ACCENT, alpha=0.75, edgecolors=BG, linewidths=0.5)
    ax.set_xlabel("params (M)", color=MUTED, fontsize=9)
    ax.set_ylabel("best accuracy / fitness", color=MUTED, fontsize=9)


def draw_genome_strip(fig, gs_cell, best_net):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Best network found")
    ax.set_xticks([])
    ax.set_yticks([])
    if not best_net or not best_net.get("net_list"):
        ax.text(0.5, 0.5, "no best network found", color=MUTED,
                ha="center", va="center", transform=ax.transAxes)
        return

    ops = best_net["net_list"]
    n = len(ops)
    box_w = 1.0 / n
    for i, op in enumerate(ops):
        x0 = i * box_w
        color = _op_color(op)
        ax.add_patch(FancyBboxPatch((x0 + box_w * 0.06, 0.25), box_w * 0.88, 0.5,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     transform=ax.transAxes, facecolor=color,
                                     edgecolor=BG, linewidth=1.5))
        label = op.replace("_", "\n")
        ax.text(x0 + box_w / 2, 0.5, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.5, color=BG, fontweight="bold")

    fitness = best_net.get("best_so_far")
    gen, idx = best_net.get("best_so_far_id", (None, None))
    subtitle = f"gen {gen}, individual {idx} — fitness {fitness:.2f}" if fitness is not None else ""
    ax.text(0.5, 0.05, subtitle, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=MUTED)


def draw_timing_panel(fig, gs_cell, search_time_s, retrain_runs):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Search vs. retrain time")

    durations = [r["duration_s"] for r in retrain_runs if r.get("duration_s") is not None]
    mean_retrain_s = float(np.mean(durations)) if durations else None

    if search_time_s is None and mean_retrain_s is None:
        ax.text(0.5, 0.5, "no timing data found", color=MUTED, ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    labels, values_h, colors = [], [], []
    if search_time_s is not None:
        labels.append("search")
        values_h.append(search_time_s / 3600)
        colors.append(ACCENT)
    if mean_retrain_s is not None:
        labels.append(f"retrain (n={len(durations)})")
        values_h.append(mean_retrain_s / 3600)
        colors.append(ACCENT_2)

    bars = ax.barh(labels, values_h, color=colors, height=0.5)
    ax.set_xlabel("hours", color=MUTED, fontsize=9)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    # Widen the left margin within this panel's own cell so long y-tick labels
    # (e.g. "retrain (n=2)") aren't clipped by the figure edge.
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.035, pos.y0, pos.width - 0.035, pos.height])
    for bar, val in zip(bars, values_h):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"  {val:.2f}h", va="center",
                ha="left", color=INK, fontsize=9)


def draw_stage_timeline(fig, gs_cell, gen_data):
    """Plot progressive-stage growth over the search: node count and op-menu size per
    generation (only populated when the run used QNAS.progressive_stages).
    """
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Progressive stage growth")

    gens = sorted(gen_data.keys()) if gen_data else []
    gens_with_stage = [g for g in gens if "fn_list" in gen_data[g]]
    if not gens_with_stage:
        ax.text(0.5, 0.5, "no progressive_stages configured", color=MUTED,
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    num_nodes = [gen_data[g]["num_net_nodes"] for g in gens_with_stage]
    num_ops = [len(gen_data[g]["fn_list"]) for g in gens_with_stage]

    ax2 = ax.twinx()
    ax2.tick_params(colors=MUTED, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.plot(gens_with_stage, num_nodes, color=ACCENT_3, marker="o", markersize=5,
            linewidth=2.2, label="num nodes")
    ax2.plot(gens_with_stage, num_ops, color=ACCENT, marker="s", markersize=4,
            linewidth=1.6, linestyle="--", label="num ops")
    ax.set_xlabel("generation", color=MUTED, fontsize=9)
    ax.set_ylabel("num nodes", color=ACCENT_3, fontsize=9)
    ax2.set_ylabel("num ops", color=ACCENT, fontsize=9)


def draw_retrain_panel(fig, gs_cell, retrain_runs, indiv_df):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Accuracy: search vs. retrain")
    ax.set_xticks([])
    if not retrain_runs:
        ax.text(0.5, 0.5, "no retrain results found", color=MUTED, ha="center", va="center",
                transform=ax.transAxes)
        ax.set_yticks([])
        return

    search_best = indiv_df["best_accuracy"].max() if not indiv_df.empty else np.nan
    search_best_params = indiv_df.loc[indiv_df["best_accuracy"].idxmax(), "total_trainable_params"] \
        if not indiv_df.empty and indiv_df["best_accuracy"].notna().any() else None

    retrain_accs = [r["accuracy"] for r in retrain_runs]

    labels = ["best in search"] + [f"retrain {i+1}" for i in range(len(retrain_accs))]
    values = [search_best] + retrain_accs
    params_labels = [search_best_params] + [None] * len(retrain_accs)
    colors = [ACCENT] + [ACCENT_2] * len(retrain_accs)
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    for bar, val, params in zip(bars, values, params_labels):
        if val != val:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center",
                va="bottom", color=INK, fontsize=8)
        # Param count of the collapsed/retrained architecture, under the accuracy label.
        if params:
            ax.text(bar.get_x() + bar.get_width() / 2, val * 0.5,
                    f"{params/1e6:.2f}M params", ha="center", va="center",
                    color=BG, fontsize=7.5, fontweight="bold")


def draw_stat_tiles(fig, gs_row, indiv_df, gen_data, search_time_s, retrain_runs, num_gpus):
    ax = fig.add_subplot(gs_row)
    ax.set_facecolor(BG)
    ax.axis("off")

    best_acc = indiv_df["best_accuracy"].max() if not indiv_df.empty else None
    best_params = indiv_df.loc[indiv_df["best_accuracy"].idxmax(), "total_trainable_params"] \
        if not indiv_df.empty and indiv_df["best_accuracy"].notna().any() else None
    n_generations = len(gen_data) if gen_data else 0
    # GPU-days = wall-clock search time * number of physical GPUs used. Threads
    # sharing one GPU don't add throughput, so this is wall time * len(available_gpus),
    # not a sum of per-individual durations (those threads contend for the same
    # GPU(s), and only a handful of individual dirs typically survive on disk to
    # average over, making a per-individual extrapolation unreliable).
    gpu_days = (search_time_s * num_gpus / 86400) if search_time_s is not None else None

    retrain_accs = [r["accuracy"] for r in retrain_runs]
    retrain_durations = [r["duration_s"] for r in retrain_runs if r.get("duration_s") is not None]
    mean_retrain_acc = float(np.mean(retrain_accs)) if retrain_accs else None
    mean_retrain_time_s = float(np.mean(retrain_durations)) if retrain_durations else None

    tiles = [
        ("Best acc (search)", f"{best_acc:.2f}" if best_acc is not None else "—"),
        ("Mean acc (retrain)", f"{mean_retrain_acc:.2f}" if mean_retrain_acc is not None else "—"),
        ("Best model size", f"{best_params/1e6:.2f}M" if best_params else "—"),
        ("Search time", _format_hours(search_time_s)),
        ("Mean retrain time", _format_hours(mean_retrain_time_s)),
        ("GPU-days (search)", f"{gpu_days:.2f}" if gpu_days is not None else "—"),
        ("Generations run", f"{n_generations}"),
    ]

    n = len(tiles)
    tile_w = 1.0 / n
    for i, (label, value) in enumerate(tiles):
        x0 = i * tile_w
        ax.add_patch(FancyBboxPatch((x0 + tile_w * 0.06, 0.05), tile_w * 0.88, 0.9,
                                     boxstyle="round,pad=0.01,rounding_size=0.03",
                                     transform=ax.transAxes, facecolor=PANEL_BG,
                                     edgecolor=GRID, linewidth=1))
        ax.text(x0 + tile_w / 2, 0.62, value, transform=ax.transAxes, ha="center", va="center",
                fontsize=17, color=ACCENT_2, fontweight="bold")
        ax.text(x0 + tile_w / 2, 0.25, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=MUTED)


def generate_infographic(experiment_path, output_path):
    gen_data = load_generation_data(experiment_path)
    best_net = load_best_network(experiment_path, gen_data)
    indiv_df = load_individuals(experiment_path)
    retrain_runs = load_retrain_runs(experiment_path)
    run_config = load_run_config(experiment_path)
    search_time_s = search_wall_time_seconds(experiment_path, gen_data)
    num_gpus = len(run_config.get("train", {}).get("available_gpus") or [1])

    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    gs = gridspec.GridSpec(
        6, 2, figure=fig,
        height_ratios=[0.55, 1.3, 1.3, 1.3, 1.3, 0.5],
        hspace=0.55, wspace=0.25,
        left=0.05, right=0.94, top=0.97, bottom=0.03,
    )

    draw_header(fig, gs[0, :], experiment_path, gen_data, indiv_df, run_config, search_time_s)
    draw_fitness_curve(fig, gs[1, 0], gen_data)
    draw_pareto_scatter(fig, gs[1, 1], indiv_df)
    draw_genome_strip(fig, gs[2, :], best_net)
    draw_timing_panel(fig, gs[3, 0], search_time_s, retrain_runs)
    draw_stage_timeline(fig, gs[3, 1], gen_data)
    draw_retrain_panel(fig, gs[4, :], retrain_runs, indiv_df)
    draw_stat_tiles(fig, gs[5, :], indiv_df, gen_data, search_time_s, retrain_runs, num_gpus)

    fig.savefig(output_path, dpi=180, facecolor=BG)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, required=True,
                        help="Directory with the Q-NAS search (and optionally retrain) artifacts.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path. Defaults to <experiment_path>/infographic.png.")
    args = parser.parse_args()

    out = args.output or os.path.join(args.experiment_path, "infographic.png")
    result_path = generate_infographic(args.experiment_path, out)
    print(f"Infographic saved to {result_path}")

""" Generate a single-page infographic summarizing a Q-NAS search (and, if present,
    a retrain phase) run: fitness over generations, the best network found, and
    weight-bank / transfer-learning reuse rates.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

from util import load_pkl, load_yaml

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


def load_best_network(experiment_path):
    best_path = os.path.join(experiment_path, "best_network_collapsed.pkl")
    if not os.path.exists(best_path):
        return None
    return load_pkl(best_path)


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


def load_weight_bank(experiment_path):
    index_path = os.path.join(experiment_path, "weight_bank", "index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        return json.load(f)


def load_retrain_results(experiment_path):
    for path in glob.glob(os.path.join(experiment_path, "retrain_results_*.txt")):
        with open(path) as f:
            try:
                return json.load(f)
            except Exception:
                continue
    return None


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


def draw_header(fig, gs_row, experiment_path, gen_data, indiv_df, run_config):
    ax = fig.add_subplot(gs_row)
    ax.set_facecolor(BG)
    ax.axis("off")

    name = os.path.basename(os.path.normpath(experiment_path))
    qnas_cfg = run_config.get("QNAS", {})
    mixedop = qnas_cfg.get("mixedop_mode", False)
    n_generations = len(gen_data) if gen_data else 0
    n_individuals = _total_individuals_evaluated(gen_data) or len(indiv_df)

    avg_time = indiv_df["training_time"].mean() if not indiv_df.empty else np.nan
    if avg_time == avg_time and n_individuals:
        # Per-individual dirs may have been pruned from disk (e.g. weight reuse
        # cleanup), so scale the observed average up to the full evaluated count.
        total_time_h = avg_time * n_individuals / 3600
        time_label = f"~{total_time_h:.1f}h (estimated)"
    else:
        time_label = "—"

    ax.text(0, 0.75, f"Q-NAS Search Report — {name}", color=INK, fontsize=22,
            fontweight="bold", transform=ax.transAxes, va="top")
    mode_label = "MixedOperation (DARTS-style)" if mixedop else "Discrete"
    subtitle = f"Mode: {mode_label}   |   Generations: {n_generations}   |   Individuals evaluated: {n_individuals}   |   Total training time: {time_label}"
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
    colors = [ACCENT_2 if reused else ACCENT for reused in df["weight_reuse_applied"]]
    ax.scatter(df["total_trainable_params"] / 1e6, df["best_accuracy"], s=sizes,
               c=colors, alpha=0.75, edgecolors=BG, linewidths=0.5)
    ax.set_xlabel("params (M)", color=MUTED, fontsize=9)
    ax.set_ylabel("best accuracy / fitness", color=MUTED, fontsize=9)


def draw_genome_strip(fig, gs_cell, best_net):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Best network found")
    ax.set_xticks([])
    ax.set_yticks([])
    if not best_net or not best_net.get("net_list"):
        ax.text(0.5, 0.5, "no best_network_collapsed.pkl found", color=MUTED,
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


def draw_weight_reuse_panel(fig, gs_cell, indiv_df, weight_bank):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Weight reuse / transfer learning")
    ax.set_xticks([])
    ax.set_yticks([])

    if indiv_df.empty:
        ax.text(0.5, 0.5, "no per-individual data found", color=MUTED, ha="center", va="center",
                transform=ax.transAxes)
        return

    n_total = len(indiv_df)
    n_reused = int(indiv_df["weight_reuse_applied"].sum())
    hit_rate = n_reused / n_total if n_total else 0.0

    # Use explicit data coordinates for both the pie and the text so they share
    # one coordinate system (ax.pie's `center`/`radius` are data coords, not axes
    # fractions - mixing the two caused the pie and text to overlap previously).
    ax.set_xlim(-1.3, 2.6)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")

    wedge_colors = [ACCENT_2, GRID] if n_reused else [GRID]
    wedge_values = [n_reused, n_total - n_reused] if n_reused else [1]
    ax.pie(wedge_values, colors=wedge_colors, startangle=90,
           wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=2),
           radius=1.15, center=(0, 0))
    ax.text(0, 0, f"{hit_rate*100:.0f}%", ha="center", va="center", fontsize=15,
            color=INK, fontweight="bold")

    bank_size = len(weight_bank) if weight_bank else 0
    reused_time = indiv_df.loc[indiv_df["weight_reuse_applied"], "training_time"].mean()
    scratch_time = indiv_df.loc[~indiv_df["weight_reuse_applied"], "training_time"].mean()
    lines = [f"{n_reused}/{n_total} individuals reused weights", f"weight bank size: {bank_size}"]
    if reused_time == reused_time and scratch_time == scratch_time:
        lines.append(f"avg train time — reused: {reused_time:.0f}s, scratch: {scratch_time:.0f}s")
    elif reused_time == reused_time:
        lines.append(f"avg train time (reused): {reused_time:.0f}s")
    elif scratch_time == scratch_time:
        lines.append(f"avg train time (scratch): {scratch_time:.0f}s")

    ax.text(1.5, 0.4, "\n\n".join(lines), ha="left", va="top", fontsize=9, color=MUTED)


def draw_alpha_trend(fig, gs_cell, gen_data):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Alpha confidence (MixedOp)")

    gens = sorted(gen_data.keys()) if gen_data else []
    gens_with_alpha = [g for g in gens if "alpha_logits" in gen_data[g]]
    if not gens_with_alpha:
        ax.text(0.5, 0.5, "discrete-mode run — no alpha logits", color=MUTED,
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    mean_conf, max_conf = [], []
    for g in gens_with_alpha:
        alphas = np.asarray(gen_data[g]["alpha_logits"], dtype=float)  # [pop, nodes, ops]
        exp = np.exp(alphas - alphas.max(axis=-1, keepdims=True))
        softmax = exp / exp.sum(axis=-1, keepdims=True)
        max_conf.append(softmax.max(axis=-1).mean())
        entropy = -(softmax * np.log(softmax + 1e-12)).sum(axis=-1)
        mean_conf.append(entropy.mean())

    ax2 = ax.twinx()
    ax2.tick_params(colors=MUTED, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.plot(gens_with_alpha, max_conf, color=ACCENT_3, marker="o", markersize=5,
            linewidth=2.2, label="mean max-softmax")
    ax2.plot(gens_with_alpha, mean_conf, color=ACCENT, marker="s", markersize=4,
            linewidth=1.6, linestyle="--", label="mean entropy")
    ax.set_xlabel("generation", color=MUTED, fontsize=9)
    ax.set_ylabel("confidence", color=ACCENT_3, fontsize=9)
    ax2.set_ylabel("entropy", color=ACCENT, fontsize=9)
    ax.set_xticks(gens_with_alpha)


def draw_retrain_panel(fig, gs_cell, retrain_results, indiv_df):
    ax = fig.add_subplot(gs_cell)
    _style_axes(ax, "Retrain comparison")
    ax.set_xticks([])
    if not retrain_results:
        ax.text(0.5, 0.5, "no retrain results found", color=MUTED, ha="center", va="center",
                transform=ax.transAxes)
        ax.set_yticks([])
        return

    search_best = indiv_df["best_accuracy"].max() if not indiv_df.empty else np.nan
    retrain_accs = []
    for run_name, res in retrain_results.items():
        acc = res.get("test_accuracy", res.get("best_accuracy"))
        if acc is not None:
            retrain_accs.append(acc)

    labels = ["best in search"] + [f"retrain {i+1}" for i in range(len(retrain_accs))]
    values = [search_best] + retrain_accs
    colors = [ACCENT] + [ACCENT_2] * len(retrain_accs)
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    for bar, val in zip(bars, values):
        if val == val:
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center",
                    va="bottom", color=INK, fontsize=8)


def draw_stat_tiles(fig, gs_row, indiv_df, weight_bank, gen_data):
    ax = fig.add_subplot(gs_row)
    ax.set_facecolor(BG)
    ax.axis("off")

    best_acc = indiv_df["best_accuracy"].max() if not indiv_df.empty else None
    best_params = indiv_df.loc[indiv_df["best_accuracy"].idxmax(), "total_trainable_params"] \
        if not indiv_df.empty and indiv_df["best_accuracy"].notna().any() else None
    n_individuals = _total_individuals_evaluated(gen_data) or len(indiv_df)
    avg_time = indiv_df["training_time"].mean() if not indiv_df.empty else np.nan
    # Single-GPU-equivalent compute time: sum of each individual's own training
    # time, regardless of how many ran in parallel (threads) - the standard
    # "GPU-days" accounting used to report NAS search cost.
    total_time_s = avg_time * n_individuals if avg_time == avg_time and n_individuals else 0
    total_time_h = total_time_s / 3600
    gpu_days = total_time_s / 86400
    hit_rate = indiv_df["weight_reuse_applied"].mean() * 100 if not indiv_df.empty else 0
    n_generations = len(gen_data) if gen_data else 0

    tiles = [
        ("Best fitness", f"{best_acc:.2f}" if best_acc is not None else "—"),
        ("Best model size", f"{best_params/1e6:.2f}M" if best_params else "—"),
        ("Total train time", f"{total_time_h:.1f}h"),
        ("GPU-days", f"{gpu_days:.2f}"),
        ("Weight-reuse hit rate", f"{hit_rate:.0f}%"),
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
    best_net = load_best_network(experiment_path)
    indiv_df = load_individuals(experiment_path)
    weight_bank = load_weight_bank(experiment_path)
    retrain_results = load_retrain_results(experiment_path)
    run_config = load_run_config(experiment_path)

    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    gs = gridspec.GridSpec(
        6, 2, figure=fig,
        height_ratios=[0.55, 1.3, 1.3, 1.3, 1.3, 0.5],
        hspace=0.55, wspace=0.25,
        left=0.05, right=0.94, top=0.97, bottom=0.03,
    )

    draw_header(fig, gs[0, :], experiment_path, gen_data, indiv_df, run_config)
    draw_fitness_curve(fig, gs[1, 0], gen_data)
    draw_pareto_scatter(fig, gs[1, 1], indiv_df)
    draw_genome_strip(fig, gs[2, :], best_net)
    draw_weight_reuse_panel(fig, gs[3, 0], indiv_df, weight_bank)
    draw_alpha_trend(fig, gs[3, 1], gen_data)
    draw_retrain_panel(fig, gs[4, :], retrain_results, indiv_df)
    draw_stat_tiles(fig, gs[5, :], indiv_df, weight_bank, gen_data)

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

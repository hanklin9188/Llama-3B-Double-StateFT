#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


PAPER = "#f4f0e6"
INK = "#20201d"
RED = "#b64235"
TEAL = "#167d78"
GOLD = "#c38b22"
BLUE = "#3f67a4"
GRID = "#d8d0c0"


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def median_by_step(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        value = number(row[field])
        if math.isfinite(value):
            grouped[int(row["step"])].append(value)
    steps = sorted(grouped)
    return steps, [statistics.median(grouped[step]) for step in steps]


def latest_trainer_state(adapter_dir):
    checkpoints = []
    for directory in Path(adapter_dir).glob("checkpoint-*"):
        suffix = directory.name.rsplit("-", 1)[-1]
        state_path = directory / "trainer_state.json"
        if suffix.isdigit() and state_path.is_file():
            checkpoints.append((int(suffix), state_path))
    if not checkpoints:
        return None
    with max(checkpoints)[1].open(encoding="utf-8") as handle:
        return json.load(handle)


def rank_matrices(rank_rows, runtime_rows, layer_count=28):
    event_steps = sorted({int(row["step"]) for row in runtime_rows})
    rank_by_step = defaultdict(dict)
    for row in rank_rows:
        rank_by_step[int(row["step"])][(int(row["layer"]), row["branch"])] = int(
            row["rank_after"]
        )
    current = {(layer, branch): 64 for layer in range(layer_count) for branch in ("attn", "mlp")}
    matrices = {"attn": [], "mlp": []}
    for step in event_steps:
        current.update(rank_by_step.get(step, {}))
        for branch in matrices:
            matrices[branch].append([current[(layer, branch)] for layer in range(layer_count)])
    return event_steps, {
        branch: np.asarray(values, dtype=float).T for branch, values in matrices.items()
    }


def style_axis(axis):
    axis.set_facecolor(PAPER)
    axis.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(colors=INK, labelsize=9)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.title.set_color(INK)


def panel_label(axis, label):
    axis.text(
        -0.12,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=14,
        fontweight="bold",
        color=INK,
        va="top",
    )


def plot(metrics_dir, output_prefix):
    metrics_dir = Path(metrics_dir).resolve()
    adapter_dir = metrics_dir.parent
    geometry = read_csv(metrics_dir / "branch_geometry_all.csv")
    capacity = read_csv(metrics_dir / "branch_capacity_all.csv")
    rank_rows = read_csv(metrics_dir / "rank_all.csv")
    runtime = read_csv(metrics_dir / "runtime_overhead.csv")
    trainer_state = latest_trainer_state(adapter_dir)
    if not geometry or not runtime:
        raise RuntimeError("Geometry and runtime metrics are required")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )
    figure = plt.figure(figsize=(15.5, 14), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 1.05])

    # A: validation loss
    axis = figure.add_subplot(grid[0, 0])
    style_axis(axis)
    if trainer_state:
        evaluations = [row for row in trainer_state["log_history"] if "eval_loss" in row]
        steps = [int(row["step"]) for row in evaluations]
        losses = [float(row["eval_loss"]) for row in evaluations]
        axis.plot(steps, losses, color=RED, linewidth=2.2)
        best_index = int(np.argmin(losses))
        axis.scatter([steps[best_index]], [losses[best_index]], color=INK, s=38, zorder=4)
        axis.annotate(
            f"best {losses[best_index]:.4f}\nstep {steps[best_index]}",
            (steps[best_index], losses[best_index]),
            xytext=(12, 15),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
        )
    axis.set_title("Validation loss")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Causal-LM loss")
    panel_label(axis, "A")

    # B: representation geometry
    axis = figure.add_subplot(grid[0, 1])
    style_axis(axis)
    input_steps, input_id = median_by_step(geometry, "id_input_lcb")
    output_steps, output_id = median_by_step(geometry, "id_output_median")
    energy_steps, output_energy = median_by_step(geometry, "output_energy")
    axis.plot(input_steps, input_id, color=BLUE, linewidth=2.1, marker="o", ms=3, label="Input ID LCB")
    axis.plot(output_steps, output_id, color=TEAL, linewidth=2.1, marker="o", ms=3, label="Output ID")
    axis.set_title("Branch representation geometry")
    axis.set_xlabel("Allocation step")
    axis.set_ylabel("Median intrinsic dimension")
    energy_axis = axis.twinx()
    energy_axis.plot(
        energy_steps,
        np.asarray(output_energy) * 1000,
        color=GOLD,
        linewidth=1.8,
        linestyle="--",
        label="Output energy x1000",
    )
    energy_axis.set_ylabel("Median output RMS x1000", color=GOLD)
    energy_axis.tick_params(axis="y", colors=GOLD, labelsize=9)
    energy_axis.spines["top"].set_visible(False)
    energy_axis.spines["right"].set_color(GOLD)
    lines = axis.lines + energy_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="best")
    panel_label(axis, "B")

    # C: effective capacity
    axis = figure.add_subplot(grid[1, 0])
    style_axis(axis)
    effective_steps, effective_rank = median_by_step(capacity, "effective_rank")
    saturation_steps, saturation = median_by_step(capacity, "parameter_rank_saturation")
    axis.plot(effective_steps, effective_rank, color=RED, linewidth=2.2, marker="o", ms=3)
    axis.axhline(64, color=INK, linewidth=1, linestyle=":", alpha=0.65, label="Initial active rank 64")
    axis.set_title("Control capacity utilization")
    axis.set_xlabel("Allocation step")
    axis.set_ylabel("Median effective rank", color=RED)
    axis.tick_params(axis="y", colors=RED)
    saturation_axis = axis.twinx()
    saturation_axis.plot(
        saturation_steps,
        np.asarray(saturation) * 100,
        color=TEAL,
        linewidth=2,
        linestyle="--",
    )
    saturation_axis.set_ylabel("Median parameter saturation (%)", color=TEAL)
    saturation_axis.tick_params(axis="y", colors=TEAL, labelsize=9)
    saturation_axis.set_ylim(0, 100)
    saturation_axis.spines["top"].set_visible(False)
    saturation_axis.spines["right"].set_color(TEAL)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    panel_label(axis, "C")

    event_steps, rank_maps = rank_matrices(rank_rows, runtime)
    heatmap_norm = TwoSlopeNorm(vmin=32, vcenter=64, vmax=128)
    heatmap_image = None
    for panel, branch, location in (("D", "attn", grid[1, 1]), ("E", "mlp", grid[2, 0])):
        axis = figure.add_subplot(location)
        axis.set_facecolor(PAPER)
        heatmap_image = axis.imshow(
            rank_maps[branch],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdYlBu_r",
            norm=heatmap_norm,
        )
        tick_indices = np.linspace(0, len(event_steps) - 1, 7, dtype=int)
        axis.set_xticks(tick_indices, [str(event_steps[index]) for index in tick_indices])
        axis.set_yticks(range(0, 28, 3))
        axis.set_title(f"{branch.upper()} branch-rank trajectory")
        axis.set_xlabel("Allocation step")
        axis.set_ylabel("Decoder layer")
        axis.tick_params(labelsize=8, colors=INK)
        panel_label(axis, panel)
    colorbar = figure.colorbar(heatmap_image, ax=figure.axes[-2:], shrink=0.78, pad=0.02)
    colorbar.set_label("Active rank", color=INK)
    colorbar.ax.tick_params(colors=INK, labelsize=8)

    # F: allocation decisions and overhead
    axis = figure.add_subplot(grid[2, 1])
    style_axis(axis)
    allocation_steps = [int(row["step"]) for row in runtime]
    accepted = [int(row["accepted_transfers"]) for row in runtime]
    overhead_minutes = [float(row["wall_time"]) / 60 for row in runtime]
    colors = [TEAL if value else GRID for value in accepted]
    axis.bar(allocation_steps, accepted, width=420, color=colors, edgecolor=INK, linewidth=0.35)
    axis.set_title("Rank-exchange decisions and overhead")
    axis.set_xlabel("Allocation step")
    axis.set_ylabel("Accepted transfers")
    axis.set_ylim(0, max(accepted) + 1)
    overhead_axis = axis.twinx()
    overhead_axis.plot(allocation_steps, overhead_minutes, color=RED, linewidth=1.8, marker=".")
    overhead_axis.set_ylabel("Allocator wall time (minutes)", color=RED)
    overhead_axis.tick_params(axis="y", colors=RED, labelsize=9)
    overhead_axis.spines["top"].set_visible(False)
    overhead_axis.spines["right"].set_color(RED)
    panel_label(axis, "F")

    latest_geometry_step = max(int(row["step"]) for row in geometry)
    latest_rank_step = max((int(row["step"]) for row in rank_rows), default=0)
    figure.suptitle(
        "ID-DR StateFT: Training Geometry and Dynamic Rank Allocation",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.5,
        -0.012,
        f"Llama 3.2 3B | geometry through step {latest_geometry_step} | "
        f"rank allocation frozen after step {latest_rank_step} | global rank budget = 3584",
        ha="center",
        fontsize=9,
        color="#5b574f",
    )

    output_prefix = Path(output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def save_single(figure, output_dir, name):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def single_figure(title, width=8.8, height=5.8):
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    figure.suptitle(title, fontsize=16, fontweight="bold", color=INK)
    return figure, axis


def plot_separate(metrics_dir, output_dir):
    metrics_dir = Path(metrics_dir).resolve()
    adapter_dir = metrics_dir.parent
    geometry = read_csv(metrics_dir / "branch_geometry_all.csv")
    capacity = read_csv(metrics_dir / "branch_capacity_all.csv")
    rank_rows = read_csv(metrics_dir / "rank_all.csv")
    runtime = read_csv(metrics_dir / "runtime_overhead.csv")
    trainer_state = latest_trainer_state(adapter_dir)
    outputs = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )

    figure, axis = single_figure("Validation Loss During ID-DR StateFT")
    style_axis(axis)
    if trainer_state:
        evaluations = [row for row in trainer_state["log_history"] if "eval_loss" in row]
        steps = [int(row["step"]) for row in evaluations]
        losses = [float(row["eval_loss"]) for row in evaluations]
        axis.plot(steps, losses, color=RED, linewidth=2.5)
        best_index = int(np.argmin(losses))
        axis.scatter(steps[best_index], losses[best_index], color=INK, s=48, zorder=4)
        axis.annotate(
            f"Best = {losses[best_index]:.5f} at step {steps[best_index]}",
            (steps[best_index], losses[best_index]),
            xytext=(-175, 25),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9},
        )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Causal-LM validation loss")
    outputs.append(save_single(figure, output_dir, "01_validation_loss"))

    figure, axis = single_figure("Branch Representation Geometry")
    style_axis(axis)
    input_steps, input_id = median_by_step(geometry, "id_input_lcb")
    output_steps, output_id = median_by_step(geometry, "id_output_median")
    energy_steps, output_energy = median_by_step(geometry, "output_energy")
    axis.plot(input_steps, input_id, color=BLUE, linewidth=2.4, marker="o", ms=3.5, label="Input ID LCB")
    axis.plot(output_steps, output_id, color=TEAL, linewidth=2.4, marker="o", ms=3.5, label="Output ID")
    axis.set_xlabel("Allocation step")
    axis.set_ylabel("Median intrinsic dimension")
    second = axis.twinx()
    second.plot(
        energy_steps,
        np.asarray(output_energy) * 1000,
        color=GOLD,
        linewidth=2,
        linestyle="--",
        label="Output energy x1000",
    )
    second.set_ylabel("Median output RMS x1000", color=GOLD)
    second.tick_params(axis="y", colors=GOLD)
    second.spines["top"].set_visible(False)
    second.spines["right"].set_color(GOLD)
    lines = axis.lines + second.lines
    axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    outputs.append(save_single(figure, output_dir, "02_branch_geometry"))

    figure, axis = single_figure("Control Capacity Utilization")
    style_axis(axis)
    effective_steps, effective_rank = median_by_step(capacity, "effective_rank")
    saturation_steps, saturation = median_by_step(capacity, "parameter_rank_saturation")
    axis.plot(effective_steps, effective_rank, color=RED, linewidth=2.5, marker="o", ms=3.5)
    axis.axhline(64, color=INK, linewidth=1.2, linestyle=":", label="Initial active rank = 64")
    axis.set_xlabel("Allocation step")
    axis.set_ylabel("Median effective rank", color=RED)
    axis.tick_params(axis="y", colors=RED)
    second = axis.twinx()
    second.plot(saturation_steps, np.asarray(saturation) * 100, color=TEAL, linewidth=2.3, linestyle="--")
    second.set_ylabel("Median parameter saturation (%)", color=TEAL)
    second.tick_params(axis="y", colors=TEAL)
    second.set_ylim(0, 100)
    second.spines["top"].set_visible(False)
    second.spines["right"].set_color(TEAL)
    axis.legend(frameon=False, loc="upper left")
    outputs.append(save_single(figure, output_dir, "03_capacity_utilization"))

    event_steps, rank_maps = rank_matrices(rank_rows, runtime)
    heatmap_norm = TwoSlopeNorm(vmin=32, vcenter=64, vmax=128)
    for index, branch in enumerate(("attn", "mlp"), start=4):
        figure, axis = single_figure(
            f"{branch.upper()} Branch-Rank Trajectory", width=10.5, height=6.4
        )
        image = axis.imshow(
            rank_maps[branch],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdYlBu_r",
            norm=heatmap_norm,
        )
        tick_indices = np.linspace(0, len(event_steps) - 1, 9, dtype=int)
        axis.set_xticks(tick_indices, [str(event_steps[position]) for position in tick_indices])
        axis.set_yticks(range(28))
        axis.set_xlabel("Allocation step")
        axis.set_ylabel("Decoder layer")
        axis.tick_params(labelsize=8, colors=INK)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9, pad=0.025)
        colorbar.set_label("Active rank")
        outputs.append(save_single(figure, output_dir, f"{index:02d}_{branch}_rank_heatmap"))

    figure, axis = single_figure("Rank-Exchange Decisions and Allocator Cost")
    style_axis(axis)
    allocation_steps = [int(row["step"]) for row in runtime]
    accepted = [int(row["accepted_transfers"]) for row in runtime]
    overhead_minutes = [float(row["wall_time"]) / 60 for row in runtime]
    axis.bar(
        allocation_steps,
        accepted,
        width=420,
        color=[TEAL if value else GRID for value in accepted],
        edgecolor=INK,
        linewidth=0.4,
        label="Accepted transfers",
    )
    axis.set_xlabel("Allocation step")
    axis.set_ylabel("Accepted transfers")
    axis.set_ylim(0, max(accepted) + 1)
    second = axis.twinx()
    second.plot(allocation_steps, overhead_minutes, color=RED, linewidth=2, marker="o", ms=3)
    second.set_ylabel("Allocator wall time (minutes)", color=RED)
    second.tick_params(axis="y", colors=RED)
    second.spines["top"].set_visible(False)
    second.spines["right"].set_color(RED)
    axis.legend(frameon=False, loc="upper right")
    outputs.append(save_single(figure, output_dir, "06_allocation_overhead"))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Plot ID-DR StateFT experiment metrics")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output-prefix")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=["separate", "overview", "both"], default="separate")
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir).expanduser().resolve()
    output_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else metrics_dir / "figures" / "id_dr_experiment_overview"
    )
    if args.mode in {"overview", "both"}:
        png_path, pdf_path = plot(metrics_dir, output_prefix)
        print(f"Overview PNG: {png_path}")
        print(f"Overview PDF: {pdf_path}")
    if args.mode in {"separate", "both"}:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else metrics_dir / "figures" / "separate"
        )
        for png_path, pdf_path in plot_separate(metrics_dir, output_dir):
            print(f"PNG: {png_path}")
            print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()

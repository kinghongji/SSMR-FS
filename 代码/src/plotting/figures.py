from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from config.experiment_config import METHOD_ORDER
from src.utils.random_utils import pad_path


METHOD_LABELS = {
    "简单随机": "SRS",
    "Sobol": "Sobol",
    "OA-LHS": "OA-LHS",
    "MaxPro": "MaxPro",
    "SSMR-FS": "SSMR-FS",
}

COLORS = {
    "storage_records": "#8A8A8A",
    "简单随机": "#4C78A8",
    "Sobol": "#59A14F",
    "OA-LHS": "#F28E2B",
    "MaxPro": "#B279A2",
    "SSMR-FS": "#D62728",
}

LINE_STYLES = {
    "简单随机": "--",
    "Sobol": ":",
    "OA-LHS": "-.",
    "MaxPro": (0, (5, 2)),
    "SSMR-FS": "-",
}

METRIC_PANELS = (
    ("路径覆盖误差", "(a) Path coverage error (PCE) ↓"),
    (
        "关键温区覆盖误差",
        "(b) Key temperature-zone coverage error (KCE) ↓",
    ),
    ("路径间隔", "(c) Path separation (PS) ↑"),
    ("对数特征体积", "(d) Log feature volume (LFV) ↑"),
)

LABEL_STYLES = {
    "简单随机": {
        "coverage": {"offset": (7, 7), "ha": "left", "va": "bottom"},
        "space": {"offset": (7, 5), "ha": "left", "va": "bottom"},
    },
    "Sobol": {
        "coverage": {"offset": (-7, -7), "ha": "right", "va": "top"},
        "space": {"offset": (-7, -5), "ha": "right", "va": "top"},
    },
    "OA-LHS": {
        "coverage": {"offset": (-7, 7), "ha": "right", "va": "bottom"},
        "space": {"offset": (7, 5), "ha": "left", "va": "bottom"},
    },
    "MaxPro": {
        "coverage": {"offset": (7, 7), "ha": "left", "va": "bottom"},
        "space": {"offset": (7, -6), "ha": "left", "va": "top"},
    },
    "SSMR-FS": {
        "coverage": {"offset": (7, 7), "ha": "left", "va": "bottom"},
        "space": {"offset": (-7, 7), "ha": "right", "va": "bottom"},
    },
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.unicode_minus": True,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.0,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def _save(fig: plt.Figure, output_stem: Path, dpi: int) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if output_stem.suffix:
        output_stem = output_stem.with_suffix("")
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _sorted_temperature_matrix(
    records: Sequence[dict], max_days: int
) -> np.ndarray:
    matrix = np.asarray(
        [pad_path(record["path"], max_days=max_days) for record in records],
        dtype=float,
    )
    path_lengths = np.sum(np.isfinite(matrix), axis=1)
    order = np.argsort(path_lengths, kind="stable")
    return matrix[order]


def plot_storage_temperature_heatmap(
    storage_records: Sequence[dict],
    output_stem: Path,
    dpi: int,
    max_days: int = 10,
) -> None:
    _setup_style()
    matrix = _sorted_temperature_matrix(storage_records, max_days)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(9.4, 7.8), constrained_layout=True)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-18,
        vmax=22,
    )
    ax.set_xlabel("Storage duration (d)")
    ax.set_ylabel("Record index (ordered by storage duration)")
    ax.set_xticks(np.arange(max_days), labels=np.arange(1, max_days + 1))
    y_positions = np.linspace(0, matrix.shape[0] - 1, 5, dtype=int)
    ax.set_yticks(y_positions, labels=y_positions + 1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Temperature (°C)")
    _save(fig, output_stem, dpi)


def plot_method_temperature_heatmaps(
    records_by_method: Dict[str, Sequence[dict]],
    output_stem: Path,
    dpi: int,
    max_days: int = 10,
) -> None:
    _setup_style()
    missing = [method for method in METHOD_ORDER if method not in records_by_method]
    if missing:
        raise ValueError(f"绘制五种方法热图时缺少方法：{missing}")

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    fig = plt.figure(figsize=(16.5, 10.2))
    grid = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.09],
        left=0.055,
        right=0.94,
        bottom=0.08,
        top=0.96,
        wspace=0.50,
        hspace=0.38,
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    colorbar_axis = fig.add_subplot(grid[:, 6])
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    image = None
    for ax, method, panel_label in zip(axes, METHOD_ORDER, panel_labels):
        matrix = _sorted_temperature_matrix(records_by_method[method], max_days)
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-18,
            vmax=18,
        )
        ax.set_title(f"{panel_label} {METHOD_LABELS[method]}")
        ax.set_xlabel("Storage duration (d)")
        ax.set_ylabel("Path index (ordered by storage duration)")
        ax.set_xticks(np.arange(max_days), labels=np.arange(1, max_days + 1))
        y_positions = np.linspace(0, matrix.shape[0] - 1, 5, dtype=int)
        ax.set_yticks(y_positions, labels=y_positions + 1)

    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Temperature (°C)")
    _save(fig, output_stem, dpi)


def plot_evaluation_space_distribution(
    reference_coordinates: np.ndarray,
    transformed_by_method: Dict[str, np.ndarray],
    explained_variance_ratio: np.ndarray,
    output_stem: Path,
    dpi: int,
) -> None:
    _setup_style()
    if reference_coordinates.shape[1] < 2:
        raise ValueError("评价PCA至少需要两个主成分才能绘制二维分布图。")
    missing = [method for method in METHOD_ORDER if method not in transformed_by_method]
    if missing:
        raise ValueError(f"绘制统一评价空间时缺少方法：{missing}")

    all_coordinates = np.vstack(
        [reference_coordinates]
        + [transformed_by_method[method] for method in METHOD_ORDER]
    )
    x_all = all_coordinates[:, 0]
    y_all = all_coordinates[:, 1]
    x_margin = max(0.3, 0.05 * (x_all.max() - x_all.min()))
    y_margin = max(0.3, 0.05 * (y_all.max() - y_all.min()))
    x_limits = (x_all.min() - x_margin, x_all.max() + x_margin)
    y_limits = (y_all.min() - y_margin, y_all.max() + y_margin)

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.6), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.865,
        wspace=0.22,
        hspace=0.30,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8.5,
            markerfacecolor=COLORS["storage_records"],
            markeredgecolor="none",
            label="Storage temperature records",
        )
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8.5,
            markerfacecolor=COLORS[method],
            markeredgecolor="none",
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.45,
    )
    panel_titles = ["(a) Storage temperature records"] + [
        f"({label}) {METHOD_LABELS[method]}"
        for label, method in zip(["b", "c", "d", "e", "f"], METHOD_ORDER)
    ]

    for index, (ax, title) in enumerate(zip(axes.flat, panel_titles)):
        ax.scatter(
            reference_coordinates[:, 0],
            reference_coordinates[:, 1],
            s=24,
            color=COLORS["storage_records"],
            alpha=0.34 if index > 0 else 0.68,
            edgecolors="none",
            zorder=2,
        )
        if index > 0:
            method = METHOD_ORDER[index - 1]
            method_values = transformed_by_method[method]
            ax.scatter(
                method_values[:, 0],
                method_values[:, 1],
                s=25,
                color=COLORS[method],
                alpha=0.72,
                edgecolors="none",
                zorder=3,
            )
        ax.set_title(title, pad=10)
        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}%)")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_box_aspect(1)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.axhline(0, color="#C8C8C8", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#C8C8C8", linewidth=0.8, zorder=0)
        ax.grid(color="#E5E5E5", linewidth=0.65, alpha=0.70)

    _save(fig, output_stem, dpi)


def plot_pca_cumulative_variance(
    cumulative_explained_variance: np.ndarray,
    retained_components: int,
    retained_variance: float,
    output_stem: Path,
    dpi: int,
) -> None:
    _setup_style()
    component_number = np.arange(1, len(cumulative_explained_variance) + 1)
    fig, ax = plt.subplots(figsize=(11.8, 6.8), constrained_layout=True)
    ax.fill_between(
        component_number,
        cumulative_explained_variance,
        0,
        color="#4C78A8",
        alpha=0.12,
    )
    ax.plot(
        component_number,
        cumulative_explained_variance,
        color="#2F5597",
        marker="o",
        markersize=6.0,
        linewidth=2.4,
    )
    ax.axhline(0.95, color="#D62728", linestyle="--", linewidth=1.6)
    ax.axvline(retained_components, color="#D62728", linestyle="--", linewidth=1.6)
    ax.scatter(
        [retained_components],
        [retained_variance],
        s=85,
        color="#D62728",
        zorder=4,
    )
    ax.annotate(
        f"{retained_components} PCs retained\n"
        f"Cumulative variance: {retained_variance * 100:.2f}%",
        xy=(retained_components, retained_variance),
        xytext=(retained_components + 1.5, retained_variance - 0.16),
        color="#D62728",
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": "#D62728", "linewidth": 1.4},
    )
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xlim(1, int(component_number.max()))
    ax.set_ylim(0, 1.03)
    ax.set_xticks(component_number)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(color="#E2E2E2", linewidth=0.75, alpha=0.75)
    _save(fig, output_stem, dpi)


def _add_method_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    offset_group: str,
) -> None:
    for method in METHOD_ORDER:
        row = frame.loc[frame["method"] == method].iloc[0]
        x_value = float(row[x_column])
        y_value = float(row[y_column])
        is_ssmr = method == "SSMR-FS"
        ax.scatter(
            [x_value],
            [y_value],
            s=155 if is_ssmr else 105,
            color=COLORS[method],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        label_style = LABEL_STYLES[method][offset_group]
        ax.annotate(
            METHOD_LABELS[method],
            xy=(x_value, y_value),
            xytext=label_style["offset"],
            textcoords="offset points",
            fontsize=12.5,
            fontweight="bold" if is_ssmr else "normal",
            color=COLORS[method],
            ha=label_style["ha"],
            va=label_style["va"],
        )


def _add_limits(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_margin = max(0.02, 0.16 * (x_max - x_min))
    y_margin = max(0.02, 0.16 * (y_max - y_min))
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)


def plot_design_quality_dimensions(
    metrics_frame: pd.DataFrame, output_stem: Path, dpi: int
) -> None:
    _setup_style()
    missing = [method for method in METHOD_ORDER if method not in set(metrics_frame["method"])]
    if missing:
        raise ValueError(f"主比较结果缺少方法：{missing}")
    metrics = metrics_frame.set_index("method").loc[list(METHOD_ORDER)].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.15, top=0.90, wspace=0.30)
    coverage_axis, space_axis = axes

    _add_method_points(
        coverage_axis,
        metrics,
        "路径覆盖误差",
        "关键温区覆盖误差",
        "coverage",
    )
    _add_limits(
        coverage_axis,
        metrics["路径覆盖误差"],
        metrics["关键温区覆盖误差"],
    )
    coverage_axis.set_title("(a) Coverage quality", pad=12)
    coverage_axis.set_xlabel("Path coverage error (PCE; lower is better)")
    coverage_axis.set_ylabel(
        "Key temperature-zone coverage error (KCE; lower is better)"
    )
    coverage_axis.annotate(
        "Preferred direction",
        xy=(0.10, 0.70),
        xytext=(0.31, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        color="#4D4D4D",
        fontsize=12,
        arrowprops={"arrowstyle": "->", "color": "#4D4D4D", "linewidth": 1.4},
    )

    _add_method_points(
        space_axis,
        metrics,
        "路径间隔",
        "对数特征体积",
        "space",
    )
    _add_limits(space_axis, metrics["路径间隔"], metrics["对数特征体积"])
    space_axis.set_title("(b) Path-space quality", pad=12)
    space_axis.set_xlabel("Path separation (PS; higher is better)")
    space_axis.set_ylabel("Log feature volume (LFV; higher is better)")
    space_axis.annotate(
        "Preferred direction",
        xy=(0.73, 0.34),
        xytext=(0.53, 0.14),
        xycoords="axes fraction",
        textcoords="axes fraction",
        color="#4D4D4D",
        fontsize=12,
        arrowprops={"arrowstyle": "->", "color": "#4D4D4D", "linewidth": 1.4},
    )

    for ax in axes:
        ax.grid(color="#DCDCDC", linewidth=0.75, alpha=0.75)
        ax.tick_params(axis="both", pad=6)
    _save(fig, output_stem, dpi)


def _legend_handles() -> list:
    handles = []
    for method in METHOD_ORDER:
        is_ssmr = method == "SSMR-FS"
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[method],
                linestyle=LINE_STYLES[method],
                linewidth=3.0 if is_ssmr else 1.8,
                marker="o" if is_ssmr else None,
                markersize=5.5 if is_ssmr else 0,
                label=METHOD_LABELS[method],
            )
        )
    return handles


def plot_repeated_stability(
    repeat_frame: pd.DataFrame, output_stem: Path, dpi: int
) -> None:
    _setup_style()
    required = {"method", "repeat", *[metric for metric, _ in METRIC_PANELS]}
    missing_columns = sorted(required.difference(repeat_frame.columns))
    if missing_columns:
        raise ValueError(f"30次重复结果缺少列：{missing_columns}")

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.6))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.875,
        wspace=0.23,
        hspace=0.34,
    )
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.8,
    )

    repeat_values = sorted(int(value) for value in repeat_frame["repeat"].unique())
    for ax, (metric, title) in zip(axes.flat, METRIC_PANELS):
        for method in METHOD_ORDER:
            method_frame = repeat_frame[repeat_frame["method"] == method].sort_values(
                "repeat"
            )
            is_ssmr = method == "SSMR-FS"
            ax.plot(
                method_frame["repeat"],
                method_frame[metric],
                color=COLORS[method],
                linestyle=LINE_STYLES[method],
                linewidth=3.0 if is_ssmr else 1.7,
                marker="o" if is_ssmr else None,
                markersize=4.6 if is_ssmr else 0,
                alpha=1.0 if is_ssmr else 0.86,
                zorder=3 if is_ssmr else 2,
            )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Independent run")
        ax.set_ylabel("Metric value")
        ax.set_xlim(min(repeat_values), max(repeat_values))
        if repeat_values == list(range(1, 31)):
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.margins(y=0.12)
        ax.grid(color="#DEDEDE", linewidth=0.75, alpha=0.75)
    _save(fig, output_stem, dpi)


def plot_sample_size_metrics(
    sample_frame: pd.DataFrame, output_stem: Path, dpi: int
) -> None:
    _setup_style()
    required = {"method", "sample_size", *[metric for metric, _ in METRIC_PANELS]}
    missing_columns = sorted(required.difference(sample_frame.columns))
    if missing_columns:
        raise ValueError(f"不同输出数量结果缺少列：{missing_columns}")

    output_sizes = sorted(int(value) for value in sample_frame["sample_size"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.6))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.875,
        wspace=0.23,
        hspace=0.34,
    )
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.8,
    )

    for ax, (metric, title) in zip(axes.flat, METRIC_PANELS):
        for method in METHOD_ORDER:
            method_frame = sample_frame[sample_frame["method"] == method].sort_values(
                "sample_size"
            )
            is_ssmr = method == "SSMR-FS"
            ax.plot(
                method_frame["sample_size"],
                method_frame[metric],
                color=COLORS[method],
                linestyle=LINE_STYLES[method],
                linewidth=3.0 if is_ssmr else 1.7,
                marker="o" if is_ssmr else None,
                markersize=5.5 if is_ssmr else 0,
                alpha=1.0 if is_ssmr else 0.86,
                zorder=3 if is_ssmr else 2,
            )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Number of output paths (N)")
        ax.set_ylabel("Metric value")
        ax.set_xticks(output_sizes)
        ax.margins(x=0.04, y=0.12)
        ax.grid(color="#DEDEDE", linewidth=0.75, alpha=0.75)
    _save(fig, output_stem, dpi)

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import beta

from safe_on_clean_text.conformal.methods import METHOD_LABELS

BACKGROUND = "#ffffff"
EC_DAY = ["#02692b", "#dd8e00", "#723f93", "#db5683", "#4e88cb"]
EC_NIGHT_BG = "#0b1512"
EC_NIGHT = ["#257f3d", "#e1961b", "#85569a", "#d96a8d", "#4998c5"]
GREY = "#7A7A7A"
LIGHT_GREY = "#D9D9D9"
DARK_GREY = "#4A4A4A"
MUTED_RED = "#b4534b"
# Figures are drawn at their printed size (full text width = 6.5 in) and saved at exactly that size, so the font sizes
# below are the printed sizes. Text is Computer Modern (cmr10 + mathtext "cm"); cmr10 has the TeX OT1 layout, so plain
# strings must avoid < > { } _ | \ " and non-ASCII characters (they render as other glyphs): use mathtext instead.
TEXT_WIDTH = 6.5


def ensure_parent_dir(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def figure_style() -> Iterator[None]:
    original = plt.rcParams.copy()
    try:
        plt.rcParams.update(
            {
                "figure.figsize": (TEXT_WIDTH, 3.0),
                "figure.dpi": 140,
                "figure.facecolor": BACKGROUND,
                "figure.constrained_layout.use": True,
                "axes.facecolor": BACKGROUND,
                "savefig.dpi": 300,
                "font.family": "serif",
                "font.serif": ["cmr10"],
                "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True,
                "font.size": 9,
                "figure.titlesize": 9,
                "axes.titlesize": 9,
                "axes.titlepad": 4,
                "axes.labelsize": 9,
                "axes.edgecolor": DARK_GREY,
                "axes.linewidth": 0.8,
                "axes.labelcolor": DARK_GREY,
                "axes.titlecolor": DARK_GREY,
                "xtick.color": DARK_GREY,
                "ytick.color": DARK_GREY,
                "xtick.labelsize": 8.5,
                "ytick.labelsize": 8.5,
                "legend.fontsize": 8.5,
                "legend.frameon": True,
                "legend.fancybox": False,
                "legend.borderpad": 0.4,
                "legend.labelspacing": 0.35,
                "legend.handlelength": 1.8,
                "legend.handletextpad": 0.6,
                "grid.color": LIGHT_GREY,
                "grid.linestyle": "--",
                "grid.linewidth": 0.6,
                "axes.grid": True,
                "axes.axisbelow": True,
            }
        )
        yield
    finally:
        plt.rcParams.update(original)


def apply_common_axis_format(ax: Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.9)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK_GREY)
    ax.spines["bottom"].set_color(DARK_GREY)


# Paper figures: same style; saved as vector PDF and PNG.
PAPER_MODELS = {
    # model: (label, color, line style); encoders solid, zero-shot LLMs dashed
    "tfidf_logreg": ("TF-IDF + LogReg", EC_DAY[2], "-"),
    "distilbert": ("DistilBERT", EC_DAY[0], "-"),
    "roberta": ("RoBERTa", EC_DAY[1], "-"),
    "mentalroberta": ("MentalRoBERTa", EC_DAY[3], "-"),
    "gpt_s_tok_P1": ("GPT-4.1-mini S-tok", EC_DAY[0], "--"),
    "qwen_s_tok_P1": ("Qwen3-4B S-tok", EC_DAY[1], "--"),
    "gemma_s_tok_P1": ("Gemma 4 E4B S-tok", EC_DAY[2], "--"),
}
METHOD_MARKERS = {"M0": "o", "M1": "s", "M3-mix": "^", "M5": "D", "B1": "X"}
CONDITION_LABELS = {"clean": "clean", "head_25": "head 25%", "tail_25": "tail 25%"}


def save_figure(fig: Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_parent_dir(output_path)
    # Saved at the figure size (no tight bounding box), so the file has exactly the printed size.
    # No creation date in the PDF, so rebuilding the assets gives identical files.
    fig.savefig(output_path.with_suffix(".pdf"), facecolor=BACKGROUND, metadata={"CreationDate": None})
    fig.savefig(output_path.with_suffix(".png"), facecolor=BACKGROUND)


def plot_risk_coverage_vs_workload(
    points: pd.DataFrame,
    output_path: str | Path,
    nominal: float = 0.90,
    inset: dict[str, tuple[float, float, float, float]] | None = None,
    curves: pd.DataFrame | None = None,
) -> None:
    """
    Risk-class coverage c1 (median over resplits, 95% interval) against workload per 1,000 screened at the main
    prevalence, one panel per condition; colour / line = model, marker = method (operating points of one model joined).
    points: model, condition, method, c1, c1_q025, c1_q975, workload_per_1000_pi=0.1.
    inset: condition -> (x0, x1, y0, y1) region shown enlarged inside that panel (dense operating points).
    curves (optional): model, condition, method, alpha, metric (c1 / workload_per_1000), median; drawn as thin
    c1-vs-workload lines over the alpha grid for M0 and M1 of every model.
    """

    def draw(ax, condition: str, markersize: float) -> None:
        if curves is not None:
            medians = curves.pivot_table(
                index=["model", "condition", "method", "alpha"], columns="metric", values="median"
            )
            for model, (_, color, style) in PAPER_MODELS.items():
                for method in ["M0", "M1"]:
                    line = medians.loc[(model, condition, method)].sort_index()
                    ax.plot(
                        line["workload_per_1000"],
                        line["c1"],
                        linestyle=style,
                        color=color,
                        linewidth=0.5,
                        alpha=0.5,
                    )
        for model, (label, color, style) in PAPER_MODELS.items():
            part = points[(points["model"] == model) & (points["condition"] == condition)].sort_values(
                "workload_per_1000_pi=0.1"
            )
            if part.empty:
                continue
            ax.plot(
                part["workload_per_1000_pi=0.1"], part["c1"], linestyle=style, color=color, linewidth=1.0, alpha=0.8
            )
            for _, row in part.iterrows():
                ax.errorbar(
                    row["workload_per_1000_pi=0.1"],
                    row["c1"],
                    yerr=[
                        [row["c1"] - row["c1_q025"]],
                        [row["c1_q975"] - row["c1"]],
                    ],
                    marker=METHOD_MARKERS[row["method"]],
                    color=color,
                    markersize=markersize,
                    markeredgecolor=DARK_GREY,
                    markeredgewidth=0.5,
                    elinewidth=0.7,
                    capsize=0,
                    linestyle="none",
                )
        ax.axhline(nominal, color=DARK_GREY, linestyle=":", linewidth=1.0)

    conditions = [c for c in CONDITION_LABELS if c in set(points["condition"])]
    with figure_style():
        fig, axes = plt.subplots(1, len(conditions), figsize=(TEXT_WIDTH, 3.6), sharey=True)
        for ax, condition in zip(axes, conditions):
            draw(ax, condition, markersize=4.5)
            ax.set_title(CONDITION_LABELS[condition])
            ax.set_xlabel(r"workload per 1,000 screened ($\pi = 0.10$)")
            apply_common_axis_format(ax)
            if inset and condition in inset:
                x0, x1, y0, y1 = inset[condition]
                zoom = ax.inset_axes([0.42, 0.07, 0.56, 0.52])
                draw(zoom, condition, markersize=4)
                zoom.set_xlim(x0, x1)
                zoom.set_ylim(y0, y1)
                zoom.tick_params(labelsize=8)
                zoom.set_facecolor(BACKGROUND)
                zoom.grid(True, alpha=0.6)
                ax.indicate_inset_zoom(zoom, edgecolor=DARK_GREY)
        axes[0].set_ylabel(r"risk-class coverage $c_1$")
        model_handles = [Line2D([], [], color=c, linestyle=s, label=name) for name, c, s in PAPER_MODELS.values()]
        method_handles = [
            Line2D([], [], color=DARK_GREY, marker=m, linestyle="none", label=METHOD_LABELS[k])
            for k, m in METHOD_MARKERS.items()
        ]
        # Methods in the empty lower right of the last panel, models below the panels.
        axes[-1].legend(handles=method_handles, loc="lower right", frameon=False)
        fig.legend(handles=model_handles, loc="outside lower center", ncol=4, frameon=False)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_score_mechanism(
    histograms: pd.DataFrame, cutoffs: dict[str, dict[str, float]], output_path: str | Path
) -> None:
    """
    p(stress) distributions of stress and not stress posts per condition (shares per bin), with the p(stress) above
    which each method puts the stress class in the set, labelled with its value at the top of the line.
    cutoffs: condition -> method (M0, M1, M3-mix) -> p cutoff.
    """

    conditions = list(CONDITION_LABELS)
    styles = {"M0": "-", "M1": "--", "M3-mix": ":"}
    classes = [(1, "stress posts", EC_DAY[0]), (0, "not stress posts", EC_DAY[1])]
    with figure_style():
        fig, axes = plt.subplots(1, len(conditions), figsize=(TEXT_WIDTH, 2.4), sharex=True, sharey=True)
        value_labels = []
        for ax, condition in zip(axes, conditions):
            for label, name, color in classes:
                part = histograms[(histograms["condition"] == condition) & (histograms["label"] == label)]
                ax.bar(
                    part["bin_low"],
                    part["share"],
                    width=part["bin_high"] - part["bin_low"],
                    align="edge",
                    color=color,
                    alpha=0.55,
                    edgecolor=color,
                    linewidth=0.3,
                )
            texts = []
            for method, value in cutoffs[condition].items():
                ax.axvline(value, color=DARK_GREY, linestyle=styles[method], linewidth=1.2)
                ha = "left" if value < 0.08 else "right" if value > 0.92 else "center"
                # Just above the top of the line; rows are offsets in points (independent of the panel height).
                text = ax.annotate(
                    f"{value:.2f}",
                    (value, 1.0),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=8,
                    color=DARK_GREY,
                )
                text.set_in_layout(False)
                texts.append(text)
            value_labels.append(texts)
            ax.set_title(CONDITION_LABELS[condition])
            ax.set_xlim(0, 1)
            ax.set_xlabel(r"$p(\text{stress})$")
            apply_common_axis_format(ax)
        axes[0].set_ylabel("share per bin")
        handles = [
            Line2D([], [], color=DARK_GREY, linestyle=styles[m], linewidth=1.2, label=METHOD_LABELS[m]) for m in styles
        ] + [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.55, label=name) for _, name, color in classes]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
        # Cut-off values that would overlap (close cut-offs) are moved up one by one until they no longer overlap.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rows = 1
        for texts in value_labels:
            placed = []
            for text in sorted(texts, key=lambda t: t.xy[0]):
                while any(text.get_window_extent(renderer).overlaps(box) for box in placed):
                    text.xyann = (0, text.xyann[1] + 9.5)
                placed.append(text.get_window_extent(renderer))
                rows = max(rows, round((text.xyann[1] - 2) / 9.5) + 1)
        # Titles sit above the highest row of cut-off values (same height in every panel).
        for ax in axes:
            ax.set_title(ax.get_title(), pad=4 + 9.5 * rows)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_mechanism_schematic(cutoffs: dict[str, float], output_path: str | Path) -> None:
    """
    Conceptual picture of the mechanism: degradation moves at-risk posts from high p(risk) into the band between the
    M0 and M1 risk cut-offs, where M0 still puts risk in the set and M1 does not. cutoffs: M0, M1 -> p cutoff.
    """

    m0, m1 = cutoffs["M0"], cutoffs["M1"]
    x = np.linspace(0, 1, 1001)
    # Schematic densities (beta shapes chosen for illustration), not estimated from data.
    clean = beta.pdf(x, 10, 2)
    degraded = beta.pdf(x, 2.2, 1.6)
    with figure_style():
        fig, ax = plt.subplots(figsize=(6.0, 2.4))
        ax.plot(x, clean, color=EC_DAY[0], linewidth=1.4)
        ax.plot(x, degraded, color=EC_DAY[1], linewidth=1.4, linestyle="--")
        band = (x >= m0) & (x <= m1)
        ax.fill_between(x[band], 0, degraded[band], color=MUTED_RED, alpha=0.45, linewidth=0)
        top = 1.12 * clean.max()
        for value, style, method, ha, offset in [(m0, "-", "M0", "right", -0.012), (m1, "--", "M1", "left", 0.012)]:
            ax.axvline(value, color=DARK_GREY, linestyle=style, linewidth=1.2)
            ax.text(
                value + offset,
                0.97 * top,
                f"{METHOD_LABELS[method]}\n" + rf"$p\geq {value:.2f}$",
                ha=ha,
                va="top",
                fontsize=8.5,
                color=DARK_GREY,
            )
        ax.text(0.86, 1.03 * clean.max(), "clean text", ha="right", va="center", fontsize=8.5, color=EC_DAY[0])
        peak = x[np.argmax(degraded)]
        ax.text(
            0.5,
            beta.pdf(0.5, 2.2, 1.6) + 0.07 * top,
            "degraded text",
            ha="center",
            va="center",
            fontsize=8.5,
            color=EC_DAY[1],
        )
        ax.annotate(
            "band: caught by M0,\nmissed by M1",
            xy=((m0 + m1) / 2, 0.45 * degraded[band].min()),
            xytext=(0.02, 0.45 * top),
            ha="left",
            va="center",
            fontsize=8.5,
            color=MUTED_RED,
            arrowprops={"arrowstyle": "->", "color": MUTED_RED, "linewidth": 0.9, "connectionstyle": "arc3,rad=0.25"},
        )
        start = 0.83
        ax.annotate(
            "",
            xy=(peak + 0.02, 1.08 * degraded.max()),
            xytext=(start, beta.pdf(start, 10, 2)),
            arrowprops={"arrowstyle": "->", "color": GREY, "linewidth": 1.1, "connectionstyle": "arc3,rad=0.3"},
        )
        ax.text(0.68, 0.58 * top, "truncation,\ndeletion, typos", ha="right", va="center", fontsize=8.5, color=GREY)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, top)
        ax.set_xticks([0, m0, m1, 1], ["0", f"{m0:.2f}", f"{m1:.2f}", "1"])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"model score $p(\text{risk})$ of an at-risk post")
        ax.set_ylabel("share of at-risk posts")
        save_figure(fig, output_path)
        plt.close(fig)


def plot_triage_scheme(output_path: str | Path) -> None:
    """
    Figure 1: prediction set -> triage action, and the prevalence reweighting of coverage.
    """

    with figure_style():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.2))
        ax.axis("off")
        boxes = [
            (0.012, 0.50, "conformal prediction\nset for one post", LIGHT_GREY),
            (0.30, 0.84, r"$\{$not risk$\}$ $\rightarrow$ dismiss", EC_DAY[1]),
            (0.30, 0.52, r"$\{$risk$\}$ $\rightarrow$ flag", EC_DAY[0]),
            (0.30, 0.14, r"$\{$risk, not risk$\}$" + "\nor empty set\n" + r"$\rightarrow$ refer to a human", EC_DAY[2]),
        ]
        for x, y, text, color in boxes:
            ax.text(
                x,
                y,
                text,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                color=DARK_GREY,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": color, "alpha": 0.35, "edgecolor": DARK_GREY},
            )
        for y in [0.84, 0.52, 0.14]:
            ax.annotate(
                "",
                xy=(0.29, y),
                xytext=(0.215, 0.50),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": DARK_GREY, "linewidth": 1.0},
            )
        ax.text(
            0.62,
            0.72,
            r"risk-class coverage $c_1$ = share of risk posts"
            "\nwhose set contains the risk label\n\n"
            r"workload per 1,000 = 1,000 $\times$ share"
            "\nnot dismissed; referrals = full or empty sets",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            color=DARK_GREY,
        )
        ax.text(
            0.62,
            0.14,
            r"metric($\pi$) = $\pi\,\cdot$ metric(risk)"
            "\n"
            r"$\quad + \, (1-\pi)\,\cdot$ metric(not risk)"
            "\n"
            r"e.g. coverage($\pi$) = $\pi c_1 + (1-\pi)\, c_0$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            color=DARK_GREY,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": BACKGROUND, "edgecolor": LIGHT_GREY},
        )
        save_figure(fig, output_path)
        plt.close(fig)


def plot_class_coverage(points: pd.DataFrame, output_path: str | Path, title: str = "", nominal: float = 0.90) -> None:
    """
    Figure 2 style: per model (rows) and metric (panels: coverage, c0, c1) the median over resplits with its 95% interval
    for M0 (circle) and M1 (square), on the grey reference band of that metric at the model's sizes.
    points: model, method, metric, median, q025, q975, band_low, band_high. nominal: 1 - alpha (dotted line).
    """

    metrics = [("coverage", "coverage"), ("c0", r"$c_0$ (not risk)"), ("c1", r"$c_1$ (risk)")]
    models = [m for m in PAPER_MODELS if m in set(points["model"])] + sorted(set(points["model"]) - set(PAPER_MODELS))
    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 0.3 * len(models) + 1.3), sharey=True)
        for ax, (metric, label) in zip(axes, metrics):
            for row, model in enumerate(models):
                part = points[(points["model"] == model) & (points["metric"] == metric)]
                band = part.iloc[0]
                ax.fill_betweenx(
                    [row - 0.35, row + 0.35],
                    band["band_low"],
                    band["band_high"],
                    color=LIGHT_GREY,
                    alpha=0.8,
                    linewidth=0,
                )
                for offset, method, marker in [(-0.12, "M0", "o"), (0.12, "M1", "s")]:
                    point = part[part["method"] == method]
                    if point.empty:
                        continue
                    point = point.iloc[0]
                    color = PAPER_MODELS.get(model, (model, GREY, "-"))[1]
                    ax.errorbar(
                        point["median"],
                        row + offset,
                        xerr=[[point["median"] - point["q025"]], [point["q975"] - point["median"]]],
                        marker=marker,
                        color=color,
                        markeredgecolor=DARK_GREY,
                        markeredgewidth=0.5,
                        markersize=4,
                        elinewidth=0.8,
                        linestyle="none",
                    )
            ax.axvline(nominal, color=DARK_GREY, linestyle=":", linewidth=1.0)
            ax.set_title(label)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            apply_common_axis_format(ax)
        axes[1].set_xlabel("coverage (median over resplits, 95% interval)")
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels([PAPER_MODELS.get(m, (m,))[0] for m in models])
        axes[0].invert_yaxis()
        fig.legend(
            handles=[
                Line2D([], [], marker="o", color=DARK_GREY, linestyle="none", label=METHOD_LABELS["M0"]),
                Line2D([], [], marker="s", color=DARK_GREY, linestyle="none", label=METHOD_LABELS["M1"]),
                plt.Rectangle((0, 0), 1, 1, color=LIGHT_GREY, label="reference band (guarantee holds)"),
            ],
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )
        if title:
            fig.suptitle(title)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_prevalence_curves(points: pd.DataFrame, output_path: str | Path) -> None:
    """
    Figure 3: coverage (top) and workload per 1,000 (bottom) against the risk prevalence pi, one column per model;
    lines = scenarios (M0 and M1 calibrated at the original prevalence, M0-LS), 95% interval shaded.
    points: model, scenario, prevalence, metric (coverage / workload_per_1000), median, q025, q975.
    """

    scenarios = [
        ("M0", METHOD_LABELS["M0"], EC_DAY[0], "-"),
        ("M1", METHOD_LABELS["M1"], EC_DAY[1], "--"),
        ("M0-LS", METHOD_LABELS["M0-LS"], EC_DAY[3], ":"),
    ]
    models = [m for m in PAPER_MODELS if m in set(points["model"])]
    # At most 4 model columns per block of two rows (coverage, workload); each metric shares its y axis across blocks.
    columns = min(4, len(models))
    blocks = -(-len(models) // columns)
    with figure_style():
        fig, axes = plt.subplots(
            2 * blocks, columns, figsize=(TEXT_WIDTH, 1.8 * blocks + 0.4), sharex=True, squeeze=False
        )
        for index, ax in enumerate(axes.flat):
            row, column = divmod(index, columns)
            if row // 2 * columns + column >= len(models):
                ax.remove()
                continue
            if (row, column) not in [(0, 0), (1, 0)]:
                ax.sharey(axes[row % 2, 0])
            if column > 0:
                ax.tick_params(labelleft=False)
        for position, model in enumerate(models):
            block, column = divmod(position, columns)
            for row, metric in enumerate(["coverage", "workload_per_1000"]):
                ax = axes[2 * block + row, column]
                for scenario, label, color, style in scenarios:
                    part = points[
                        (points["model"] == model) & (points["scenario"] == scenario) & (points["metric"] == metric)
                    ].sort_values("prevalence")
                    if part.empty:
                        continue
                    ax.plot(
                        part["prevalence"],
                        part["median"],
                        color=color,
                        linestyle=style,
                        linewidth=1.0,
                        marker="o",
                        markersize=2.5,
                        label=label,
                    )
                    ax.fill_between(
                        part["prevalence"], part["q025"], part["q975"], color=color, alpha=0.12, linewidth=0
                    )
                if metric == "coverage":
                    ax.axhline(0.90, color=DARK_GREY, linestyle=":", linewidth=0.9)
                    ax.set_title(PAPER_MODELS[model][0])
                else:
                    ax.tick_params(labelbottom=True)
                    ax.set_xlabel(r"risk prevalence $\pi$")
                apply_common_axis_format(ax)
        for block in range(blocks):
            axes[2 * block, 0].set_ylabel("coverage")
            axes[2 * block + 1, 0].set_ylabel("workload\nper 1,000")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_degradation_curves(
    points: pd.DataFrame, output_path: str | Path, title: str = "", height: float = 4.0
) -> None:
    """
    Figure 4: risk-class coverage c1 (top) and workload per 1,000 at pi = 0.10 (bottom) against degradation strength,
    one column per family (head, tail, del, typo); lines = calibration modes (clean: M0, M1; matched: M3; mix: M3-mix).
    points: family, strength, method, metric (c1 / workload_per_1000), median, q025, q975, band_low, band_high (c1 only).
    """

    methods = [
        ("M0", METHOD_LABELS["M0"], EC_DAY[0], "-"),
        ("M1", METHOD_LABELS["M1"], EC_DAY[1], "-"),
        ("M3", METHOD_LABELS["M3"], EC_DAY[3], "--"),
        ("M3-mix", METHOD_LABELS["M3-mix"], EC_DAY[2], ":"),
    ]
    # (family, panel title, x label); truncation axes run from 100% kept down, so degradation grows to the right.
    families = [
        ("head", "head truncation\n(first words kept)", "share of words\nkept (%)"),
        ("tail", "tail truncation\n(last words kept)", "share of words\nkept (%)"),
        ("del", "word deletion", "word deletion\nprobability (%)"),
        ("typo", "letter typos", "per-letter typo\nprobability (%)"),
    ]
    with figure_style():
        fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH, height), sharey="row")
        for column, (family, family_title, label) in enumerate(families):
            for row, metric in enumerate(["c1", "workload_per_1000"]):
                ax = axes[row, column]
                part = points[(points["family"] == family) & (points["metric"] == metric)]
                if metric == "c1" and not part.empty:
                    band = part.drop_duplicates("strength").sort_values("strength")
                    ax.fill_between(
                        band["strength"], band["band_low"], band["band_high"], color=LIGHT_GREY, linewidth=0, zorder=0.6
                    )
                for method, method_label, color, style in methods:
                    line = part[part["method"] == method].sort_values("strength")
                    ax.plot(
                        line["strength"],
                        line["median"],
                        color=color,
                        linestyle=style,
                        linewidth=1.0,
                        marker="o",
                        markersize=2.5,
                        label=method_label,
                    )
                    ax.fill_between(line["strength"], line["q025"], line["q975"], color=color, alpha=0.06, linewidth=0)
                apply_common_axis_format(ax)
            axes[1, column].set_xlabel(label)
            axes[0, column].set_title(family_title)
            if family in ("head", "tail"):
                for ax in axes[:, column]:
                    ax.set_xticks([100, 75, 50, 25])
                    ax.invert_xaxis()
        axes[0, 0].set_ylabel(r"risk-class coverage $c_1$")
        axes[1, 0].set_ylabel("workload per 1,000\n" + r"($\pi = 0.10$)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(plt.Rectangle((0, 0), 1, 1, color=LIGHT_GREY))
        labels.append(r"$c_1$ reference band (guarantee holds)")
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        if title:
            fig.suptitle(title)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_slack(points: pd.DataFrame, output_path: str | Path) -> None:
    """
    Exploratory (post hoc): x = clean M0 c1 - 0.90 (slack), y = paired head_25 c1(M1) - c1(M0) (median, 95% interval);
    one point per run; colour = model (LLMs hollow), marker = dataset.
    """

    markers = {"dreaddit": "o", "sdcnl": "^"}
    with figure_style():
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for _, row in points.iterrows():
            _, color, style = PAPER_MODELS[row["model"]]
            ax.errorbar(
                row["slack"],
                row["difference_median"],
                yerr=[
                    [row["difference_median"] - row["difference_q025"]],
                    [row["difference_q975"] - row["difference_median"]],
                ],
                marker=markers[row["dataset"]],
                color=color,
                markerfacecolor=color if style == "-" else BACKGROUND,
                markeredgecolor=color if style == "--" else DARK_GREY,
                markeredgewidth=0.9,
                markersize=4.5,
                elinewidth=0.7,
                linestyle="none",
            )
        ax.axhline(0, color=DARK_GREY, linewidth=0.8)
        ax.axvline(0, color=DARK_GREY, linewidth=0.8, linestyle=":")
        ax.set_xlabel(r"slack: clean-data $c_1$ under M0 $-$ 0.90")
        ax.set_ylabel(r"head 25%: $c_1$(M1) $-$ $c_1$(M0), paired")
        apply_common_axis_format(ax)
        handles = [
            Line2D(
                [], [], color=c, marker="o", markerfacecolor=c if s == "-" else BACKGROUND, linestyle="none", label=name
            )
            for m, (name, c, s) in PAPER_MODELS.items()
            if m in set(points["model"])
        ]
        dataset_names = {"dreaddit": "Dreaddit", "sdcnl": "SDCNL"}
        handles += [
            Line2D([], [], color=DARK_GREY, marker=mk, linestyle="none", label=dataset_names[d])
            for d, mk in markers.items()
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_lodo(points: pd.DataFrame, output_path: str | Path) -> None:
    """
    Appendix (H4): per held-out domain, median M0 / M1 coverage (95% interval over calibration bootstraps) on the domain's
    reference band; one panel per model.
    """

    models = [m for m in ["distilbert", "gpt_s_tok_P1", "qwen_s_tok_P1", "gemma_s_tok_P1"] if m in set(points["model"])]
    domains = sorted(points["domain"].unique())
    with figure_style():
        fig, axes = plt.subplots(1, len(models), figsize=(TEXT_WIDTH, 2.6), sharey=True)
        for ax, model in zip(axes, models):
            part = points[points["model"] == model]
            for index, domain in enumerate(domains):
                rows = part[part["domain"] == domain]
                band = rows.iloc[0]
                ax.fill_between(
                    [index - 0.35, index + 0.35],
                    band["band_low"],
                    band["band_high"],
                    color=LIGHT_GREY,
                    alpha=0.8,
                    linewidth=0,
                )
                for offset, method, marker in [(-0.12, "M0", "o"), (0.12, "M1", "s")]:
                    point = rows[rows["method"] == method].iloc[0]
                    ax.errorbar(
                        index + offset,
                        point["median"],
                        yerr=[[point["median"] - point["q025"]], [point["q975"] - point["median"]]],
                        marker=marker,
                        color=PAPER_MODELS[model][1],
                        markeredgecolor=DARK_GREY,
                        markeredgewidth=0.5,
                        markersize=4,
                        elinewidth=0.8,
                        linestyle="none",
                    )
            ax.set_xticks(range(len(domains)))
            ax.set_xticklabels(domains, rotation=45, ha="right", rotation_mode="anchor")
            ax.axhline(0.90, color=DARK_GREY, linestyle=":", linewidth=0.9)
            ax.set_title(PAPER_MODELS[model][0])
            apply_common_axis_format(ax)
            ax.tick_params(axis="x", which="minor", bottom=False)
        axes[0].set_ylabel("coverage of the\nheld-out domain")
        save_figure(fig, output_path)
        plt.close(fig)


def plot_grouped_bars(
    table: pd.DataFrame,
    x: str,
    groups: str,
    value: str,
    low: str | None,
    high: str | None,
    output_path: str | Path,
    ylabel: str,
    title: str = "",
    reference: float | None = None,
) -> None:
    """
    Appendix helper: grouped bars of `value` by `x` with one bar per `groups` level and optional interval whiskers.
    """

    levels = list(dict.fromkeys(table[groups]))
    categories = list(dict.fromkeys(table[x]))
    colors = EC_DAY
    width = 0.8 / len(levels)
    with figure_style():
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.8))
        for index, level in enumerate(levels):
            part = table[table[groups] == level].set_index(x).reindex(categories)
            positions = [i - 0.4 + width * (index + 0.5) for i in range(len(categories))]
            errors = None
            if low and high:
                errors = [part[value] - part[low], part[high] - part[value]]
            ax.bar(
                positions,
                part[value],
                width=width,
                color=colors[index % len(colors)],
                edgecolor=DARK_GREY,
                linewidth=0.5,
                yerr=errors,
                error_kw={"elinewidth": 0.8},
                ecolor=DARK_GREY,
                capsize=2,
                label=str(level),
            )
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel)
        if reference is not None:
            ax.axhline(reference, color=DARK_GREY, linestyle=":", linewidth=0.9)
        if title:
            ax.set_title(title)
        apply_common_axis_format(ax)
        ax.tick_params(axis="x", which="minor", bottom=False)
        fig.legend(loc="outside lower center", ncol=min(len(levels), 2), frameon=False)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_subreddit_prevalence(points: pd.DataFrame, output_path: str | Path) -> None:
    """
    Appendix: subreddit coverage (median over resplits) against the subreddit's stress share for M0 and M1, with the
    reweighting line pi * c1 + (1 - pi) * c0 of the model's overall clean class coverage; marker size ~ subreddit size.
    """

    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.2), sharey=True)
        for ax, method, color in zip(axes, ["M0", "M1"], [EC_DAY[0], EC_DAY[1]]):
            ax.scatter(
                points["stress_share"],
                points[f"{method}_coverage"],
                s=4 + 0.6 * points["n_median"],
                color=color,
                edgecolor=DARK_GREY,
                linewidth=0.5,
                alpha=0.8,
                label=r"subreddit (size $\propto$ test $n$)",
            )
            line = points.sort_values("stress_share")
            ax.plot(
                line["stress_share"],
                line[f"{method}_line"],
                color=DARK_GREY,
                linestyle="--",
                linewidth=1.0,
                label=r"$\pi c_1 + (1-\pi)\, c_0$",
            )
            middle = (points["stress_share"].min() + points["stress_share"].max()) / 2
            for _, row in points.sort_values(f"{method}_coverage", ascending=False).iterrows():
                # Labels of points in the right half go to the left of the point, so they stay inside the panel.
                right = row["stress_share"] > middle
                ax.annotate(
                    row["subreddit"].replace("_", r"$\_$"),
                    (row["stress_share"], row[f"{method}_coverage"]),
                    fontsize=8,
                    color=DARK_GREY,
                    xytext=(-4 if right else 4, 3),
                    ha="right" if right else "left",
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": LIGHT_GREY, "linewidth": 0.6},
                )
            ax.set_title(METHOD_LABELS[method])
            ax.set_xlabel("stress share in the subreddit")
            apply_common_axis_format(ax)
        axes[0].set_ylabel("coverage (median over resplits)")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
        # Overlapping labels (close subreddits) are moved down one by one until they no longer overlap.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            placed = []
            for label in [t for t in ax.texts if t.arrow_patch is not None]:
                label.set_in_layout(False)
                # Text box only (an annotation's own extent includes its connector line to the point).
                while any(Text.get_window_extent(label, renderer).overlaps(box) for box in placed):
                    label.xyann = (label.xyann[0], label.xyann[1] - 4)
                placed.append(Text.get_window_extent(label, renderer))
        save_figure(fig, output_path)
        plt.close(fig)


def plot_reliability(bins: pd.DataFrame, output_path: str | Path, positive: str = "stress") -> None:
    """
    Appendix: reliability diagrams (15 equal-width bins of p(positive class), clean test, seeds averaged per bin) per
    model, at most 4 panels per row. positive: name of the positive class ("stress" on Dreaddit, "risk" on SDCNL).
    """

    models = [m for m in PAPER_MODELS if m in set(bins["model"])]
    columns = min(4, len(models))
    rows = -(-len(models) // columns)
    with figure_style():
        fig, axes = plt.subplots(
            rows, columns, figsize=(TEXT_WIDTH, 1.9 * rows + 0.3), sharex=True, sharey=True, squeeze=False
        )
        for ax in axes.flat[len(models) :]:
            ax.remove()
        for ax, model in zip(axes.flat, models):
            part = bins[bins["model"] == model]
            ax.plot([0, 1], [0, 1], color=LIGHT_GREY, linestyle="--", linewidth=0.9)
            ax.plot(
                part["mean_p"],
                part["stress_rate"],
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                color=PAPER_MODELS[model][1],
            )
            ax.set_title(PAPER_MODELS[model][0])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(labelbottom=True)
            ax.set_xlabel(rf"mean $p(\text{{{positive}}})$")
            apply_common_axis_format(ax)
        for row in range(rows):
            axes[row, 0].set_ylabel(f"observed {positive} rate")
        save_figure(fig, output_path)
        plt.close(fig)


def plot_c1_workload_curves(points: pd.DataFrame, output_path: str | Path, nominal: float = 0.90) -> None:
    """
    H7 appendix: risk-class coverage c1 against workload per 1,000 at pi = 0.10 over the alpha grid (median over
    resplits), one row per model, columns clean / head_25; lines = M0, M1 (clean calibration) and M3 (matched), with
    the alpha = 0.10 point enlarged. points: model, condition, method, alpha, metric (c1 / workload_per_1000), median.
    """

    methods = [
        ("M0", METHOD_LABELS["M0"], EC_DAY[0], "-"),
        ("M1", METHOD_LABELS["M1"], EC_DAY[1], "-"),
        ("M3", METHOD_LABELS["M3"], EC_DAY[3], "--"),
    ]
    conditions = ["clean", "head_25"]
    medians = points.pivot_table(index=["model", "condition", "method", "alpha"], columns="metric", values="median")
    with figure_style():
        fig, axes = plt.subplots(len(PAPER_MODELS), 2, figsize=(TEXT_WIDTH, 9.0), sharey="row")
        for row, (model, (model_label, _, _)) in enumerate(PAPER_MODELS.items()):
            for column, condition in enumerate(conditions):
                ax = axes[row, column]
                for method, method_label, color, style in methods:
                    line = medians.loc[(model, condition, method)].sort_index()
                    ax.plot(
                        line["workload_per_1000"],
                        line["c1"],
                        color=color,
                        linestyle=style,
                        linewidth=1.0,
                        marker="o",
                        markersize=1.8,
                        label=method_label,
                    )
                    at_main = line.loc[0.1]
                    ax.plot(at_main["workload_per_1000"], at_main["c1"], "o", color=color, markersize=4.5)
                ax.axhline(nominal, color=DARK_GREY, linestyle=":", linewidth=1.0)
                ax.set_title(f"{model_label}, {CONDITION_LABELS[condition]}")
                apply_common_axis_format(ax)
            axes[row, 0].set_ylabel(r"$c_1$")
        for ax in axes[-1]:
            ax.set_xlabel(r"workload per 1,000 screened ($\pi = 0.10$)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], color=DARK_GREY, marker="o", markersize=4.5, linestyle="none"))
        labels.append(r"$\alpha = 0.10$")
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        save_figure(fig, output_path)
        plt.close(fig)


def plot_cut_not_short(points: pd.DataFrame, output_path: str | Path, nominal: float = 0.90) -> None:
    """
    Risk-class coverage c1 of M1 calibrated on long clean excerpts (median over resplits, 2.5-97.5% bars) per model on
    long clean, naturally short, head-cut and tail-cut long test excerpts; light grey strip = c1 reference band of the
    short test excerpts. points: model, test, median, q025, q975, c1_band_low, c1_band_high.
    """

    conditions = [
        ("long clean", "long, clean", "o"),
        ("short clean", "naturally short", "D"),
        ("long head_50", "long, cut to first 50%", "^"),
        ("long tail_50", "long, cut to last 50%", "v"),
    ]
    offsets = np.linspace(-0.27, 0.27, len(conditions))
    band = points[points["test"] == "short clean"][["c1_band_low", "c1_band_high"]].drop_duplicates()
    assert len(band) == 1
    with figure_style():
        fig, ax = plt.subplots(figsize=(0.62 * TEXT_WIDTH, 3.0))
        ax.axhspan(band["c1_band_low"].iloc[0], band["c1_band_high"].iloc[0], color=LIGHT_GREY, linewidth=0, zorder=0.6)
        ax.axhline(nominal, color=DARK_GREY, linestyle=":", linewidth=1.0)
        for position, (model, (_, color, _)) in enumerate(PAPER_MODELS.items()):
            for offset, (test, _, marker) in zip(offsets, conditions):
                row = points[(points["model"] == model) & (points["test"] == test)].iloc[0]
                ax.errorbar(
                    position + offset,
                    row["median"],
                    yerr=[[row["median"] - row["q025"]], [row["q975"] - row["median"]]],
                    marker=marker,
                    color=color,
                    markersize=4,
                    markeredgecolor=DARK_GREY,
                    markeredgewidth=0.5,
                    elinewidth=0.7,
                    capsize=0,
                    linestyle="none",
                )
        ax.set_xticks(range(len(PAPER_MODELS)))
        ax.set_xticklabels(
            [label for label, _, _ in PAPER_MODELS.values()], rotation=35, ha="right", rotation_mode="anchor"
        )
        ax.set_xlim(-0.5, len(PAPER_MODELS) - 0.5)
        ax.set_ylabel(r"risk-class coverage $c_1$ (M1)")
        apply_common_axis_format(ax)
        ax.minorticks_off()
        ax.grid(False, axis="x")
        handles = [
            Line2D([], [], color=DARK_GREY, marker=marker, markersize=4, linestyle="none", label=label)
            for _, label, marker in conditions
        ]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=LIGHT_GREY, label="band, short test posts"))
        fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8, frameon=False, columnspacing=1.0)
        save_figure(fig, output_path)
        plt.close(fig)

"""Generate every figure included by the paper.

python -m experiments.paper_figures
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")

C_RCG = "tab:blue"
C_ALT = "tab:orange"
C_MUT = "#d9d9d9"


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def plot_dimension_sampling():
    """Plot chamber sampling behavior as dimension increases.

    The margin-cap bound is a lower bound on chamber mass and does not predict
    the measured decay, so the plot omits a reference line based on that bound.
    Measured zeros appear as open markers at the Monte Carlo resolution.
    """
    with open(os.path.join(RES, "dimension_sampling.json"), encoding="utf-8") as handle:
        d = json.load(handle)
    rows = d["rows"]
    dd = np.array([r["d"] for r in rows], float)
    sa = np.array([r["solid_angle"] for r in rows], float)
    res = 5e-6  # 1/200000 Monte Carlo resolution
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    meas = sa > 0
    ax[0].semilogy(dd[meas], sa[meas], "o-", color=C_RCG)
    ax[0].semilogy(
        dd[~meas], np.full((~meas).sum(), res), "v", mfc="white", color=C_RCG
    )
    ax[0].axhline(res, ls=":", lw=1, color="0.5")
    ax[0].text(
        dd[0] + 0.3, res * 1.35, "Monte Carlo resolution", fontsize=8, color="0.35"
    )
    ax[0].set_xlabel("dimension $d$")
    ax[0].set_ylabel("minimum teacher-chamber solid angle")
    ax[0].set_title("(a) measured solid angle", fontsize=10)
    co = np.array([r["adaptive_objective"] for r in rows])
    cs = np.array([r["adaptive_objective_std"] for r in rows])
    so = np.array([r["sampling_objective"] for r in rows])
    ss = np.array([r["sampling_objective_std"] for r in rows])
    ax[1].plot(dd, co, "o-", color=C_RCG, label="residual-guided")
    ax[1].fill_between(dd, co - cs, co + cs, color=C_RCG, alpha=0.18, lw=0)
    ax[1].plot(dd, so, "s-", color=C_ALT, label="random sampling")
    ax[1].fill_between(dd, so - ss, so + ss, color=C_ALT, alpha=0.18, lw=0)
    ax[1].set_xlabel("dimension $d$")
    ax[1].set_ylabel("objective at a 40-chamber budget")
    ax[1].set_title("(b) equal chamber budget", fontsize=10)
    ax[1].legend(fontsize=9, frameon=False)
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "dimension_sampling")


def plot_runtime_breakdown():
    """Per-phase median times for the three CPU variants (Table: cpu)."""
    artifact = os.path.join(
        RES, "controlled_cpu_variants_n10000_d16_tolerance1e-4.json"
    )
    with open(artifact, encoding="utf-8") as f:
        record = json.load(f)

    labels_by_backend = {
        "baseline": "two-trial,\ncorrelation",
        "damped": "damped,\ncorrelation",
        "damped_delta": "damped,\npredicted-decrease",
    }
    phase_rows = {
        name: [
            run["solver_phase_seconds"]
            for run in record["measured_runs"]
            if run["backend"] == name and run["status"] == "ok"
        ]
        for name in labels_by_backend
    }
    if any(not rows for rows in phase_rows.values()):
        raise ValueError(f"missing successful measured runs in {artifact}")

    variants = []
    for name, label in labels_by_backend.items():
        rows = phase_rows[name]
        variants.append(
            (
                label,
                float(record["summaries"][name]["wall_seconds"]["median"]),
                float(np.median([row["price"] for row in rows])),
                float(np.median([row["gated"] for row in rows])),
            )
        )
    labels = [v[0] for v in variants]
    price = np.array([v[2] for v in variants])
    gated = np.array([v[3] for v in variants])
    other = np.array([v[1] for v in variants]) - price - gated
    y = np.arange(len(variants))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    kw = dict(height=0.62, edgecolor="white", linewidth=1.5)
    ax.barh(y, price, color=C_RCG, label="pricing search", **kw)
    ax.barh(y, gated, left=price, color=C_ALT, label="gated consolidation", **kw)
    ax.barh(y, other, left=price + gated, color=C_MUT, label="other", **kw)
    for yi, p, g, o in zip(y, price, gated, other):
        ax.text(
            p / 2, yi, f"{p:.2f}", ha="center", va="center", fontsize=8, color="white"
        )
        ax.text(
            p + g / 2,
            yi,
            f"{g:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
        ax.text(
            p + g + o + 0.06,
            yi,
            f"{p + g + o:.2f} s total",
            ha="left",
            va="center",
            fontsize=8,
            color="0.3",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("median seconds per solve")
    ax.set_xlim(0, 6.6)
    ax.legend(
        fontsize=8, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.02)
    )
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    _save(fig, "runtime_breakdown")


def plot_bisection_construction():
    """The angular bisection nontermination construction of the appendix."""
    tgt = np.pi / 3
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.75", lw=1)
    # accumulated atoms: initial e1, e2, then bisections toward pi/3
    angles = [0.0, np.pi / 2]
    lo, hi = 0.0, np.pi / 2
    adds = []
    for _ in range(4):
        m = (lo + hi) / 2
        adds.append(m)
        if tgt > m:
            lo = m
        else:
            hi = m
    for a in angles:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="0.35", lw=1.4)
    radii = [1.06, 1.13, 1.06, 1.21]
    for i, (a, rr) in enumerate(zip(adds, radii)):
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=C_RCG, lw=1.2)
        ax.annotate(
            f"{i + 1}",
            (rr * np.cos(a), rr * np.sin(a)),
            ha="center",
            va="center",
            fontsize=9,
            color=C_RCG,
        )
    ax.plot([0, 1.3 * np.cos(tgt)], [0, 1.3 * np.sin(tgt)], color=C_ALT, lw=2.2)
    ax.annotate(
        r"$y/\|y\|_2$ at $\pi/3$",
        (1.36 * np.cos(tgt), 1.36 * np.sin(tgt)),
        ha="left",
        va="bottom",
        fontsize=10,
        color=C_ALT,
    )
    ax.annotate("$u_0$", (1.08, 0.0), ha="left", va="center", fontsize=10, color="0.2")
    ax.annotate(
        "$u_{\\pi/2}$", (0.0, 1.09), ha="center", va="bottom", fontsize=10, color="0.2"
    )
    ax.set_xlim(-0.05, 1.42)
    ax.set_ylim(-0.05, 1.24)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "bisection_construction")


def plot_chamber_arrangement():
    """A central arrangement in R^2: n=4 rows, 8 chambers, one shaded."""
    np.random.default_rng(3)
    angs = np.sort(np.array([0.3, 1.45, 2.1, 2.75]))  # row normal angles
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.8", lw=1)
    for i, a in enumerate(angs):
        # hyperplane x_i^T u = 0 is the line orthogonal to the normal
        b = a + np.pi / 2
        ax.plot(
            [1.06 * np.cos(b), -1.06 * np.cos(b)],
            [1.06 * np.sin(b), -1.06 * np.sin(b)],
            color="0.55",
            lw=1.1,
        )
        ax.annotate(
            f"$x_{i + 1}$",
            (1.2 * np.cos(a), 1.2 * np.sin(a)),
            ha="center",
            va="center",
            fontsize=10,
            color="0.25",
        )
        ax.annotate(
            "",
            xytext=(0, 0),
            xy=(1.02 * np.cos(a), 1.02 * np.sin(a)),
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0),
        )
    # shade the chamber where all four rows are positive: intersection of
    # half-planes x_i^T u >= 0; boundary normals are angs, so the chamber is
    # the sector between (max ang - pi/2) and (min ang + pi/2)
    lo = np.max(angs) - np.pi / 2
    hi = np.min(angs) + np.pi / 2
    ts = np.linspace(lo, hi, 60)
    ax.fill(
        np.concatenate([[0], np.cos(ts), [0]]),
        np.concatenate([[0], np.sin(ts), [0]]),
        color=C_RCG,
        alpha=0.22,
        lw=0,
    )
    mid = (lo + hi) / 2
    ax.annotate(
        "$K_s$",
        (0.55 * np.cos(mid), 0.55 * np.sin(mid)),
        ha="center",
        va="center",
        fontsize=12,
        color=C_RCG,
    )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "chamber_arrangement")


def plot_real_data_comparison():
    """Compare RCG and tuned neural-network baselines on three datasets."""
    records = []
    for dataset in ("california", "covtype", "msd"):
        path = os.path.join(RES, f"real_data_{dataset}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                records.append((dataset, json.load(handle)))
    if not records:
        return

    fig, axes = plt.subplots(
        2,
        len(records),
        figsize=(4.8 * len(records), 6.6),
        constrained_layout=True,
        sharex="col",
        squeeze=False,
    )
    for column, (dataset, record) in enumerate(records):
        order = ["RCG"]
        colors = {"RCG": C_RCG}
        objectives = {"RCG": record["rcg"]["obj"]}
        test_mse = {"RCG": record["rcg"]["test_mse"]}
        for optimizer, color in (
            ("adam", C_ALT),
            ("adamw", "tab:red"),
            ("sgd", "tab:gray"),
        ):
            if not record.get(optimizer):
                continue
            for label, source in (
                ("best", record[optimizer]["best"]),
                ("med", record[optimizer]["median"]),
            ):
                key = f"{optimizer}\n({label})"
                order.append(key)
                colors[key] = color
                objectives[key] = source["obj"]
                test_mse[key] = source["test_mse"]

        bar_colors = [colors[key] for key in order]
        x = np.arange(len(order))
        best = min(objectives.values())
        excess = [(objectives[key] - best) / best * 100 for key in order]
        top = axes[0][column]
        top.bar(x, excess, color=bar_colors)
        for xi, value in zip(x, excess):
            label = f"{value:.1f}%" if value < 10 else f"{value:.0f}%"
            top.text(xi, value, label, ha="center", va="bottom", fontsize=6)
        top.set_ylim(bottom=0)
        top.margins(y=0.18)
        top.set_title(
            f"{dataset}\n(n={record['setup']['n']:,}, d={record['setup']['d']})",
            fontsize=9,
        )
        if column == 0:
            top.set_ylabel("% above best\ntrain objective", fontsize=8)

        bottom = axes[1][column]
        mse_values = [test_mse[key] for key in order]
        bottom.bar(x, mse_values, color=bar_colors)
        bottom.axhline(1.0, ls="--", lw=1, color="k")
        for xi, value in zip(x, mse_values):
            bottom.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6)
        bottom.set_ylim(bottom=0)
        bottom.margins(y=0.18)
        bottom.set_xticks(x)
        bottom.set_xticklabels(order, fontsize=7)
        if column == 0:
            bottom.set_ylabel("test MSE\n(var. units; 1.0 = mean)", fontsize=8)
            bottom.text(
                0.4,
                1.0,
                "predict-the-mean",
                va="bottom",
                ha="left",
                fontsize=6,
                style="italic",
            )
    _save(fig, "real_data_comparison")


if __name__ == "__main__":
    plot_chamber_arrangement()
    plot_bisection_construction()
    plot_dimension_sampling()
    plot_real_data_comparison()
    plot_runtime_breakdown()

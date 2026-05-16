"""Step-size robustness sweep for off-policy prediction baselines.

For each (environment, algorithm) we evaluate a grid of learning rates and
record the last-50% time-average RMSVE (the same AUC metric used in the main
comparison) over a population of independent seeds.  The output is one curve
per algorithm: AUC versus alpha, with shaded standard deviation across seeds.

A "robust" algorithm has both:
  (i)  a low minimum AUC over the alpha grid, and
  (ii) a wide region of alpha for which AUC stays close to that minimum.

This is the same protocol as the "sensitive test" panels in Chen et al.
(UAI 2023, "Modified Retrace for Off-Policy Temporal Difference Learning").
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the experimental machinery from the main script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from paper_prediction_experiments import (  # noqa: E402
    TD, GTD2, TDC, TDRC, GTD2_MP, HTD, ETD0, STHTD, STHTD_MP,
    run_once, score_curve,
)


DEFAULT_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
DEFAULT_ALGORITHMS = [TD, GTD2, TDC, TDRC, GTD2_MP, HTD, ETD0, STHTD, STHTD_MP]
DEFAULT_ENVS = ["two_state", "baird", "random_walk", "boyan_chain"]
DEFAULT_STEPS = {"two_state": 5000, "baird": 2000, "random_walk": 5000, "boyan_chain": 20000}

# Two-timescale baselines need a beta.  We use a fixed beta that is competitive
# in the main comparison so the robustness story is about alpha alone.
TWO_TIMESCALE = {GTD2, TDC, GTD2_MP, HTD, STHTD, STHTD_MP}
FIXED_BETA = 0.05
FIXED_TDRC_REG = 1.0


def make_cfg(alg, alpha):
    cfg = {"alpha": alpha}
    if alg in TWO_TIMESCALE:
        cfg["beta"] = FIXED_BETA
    if alg == TDRC:
        cfg["tdrc_reg"] = FIXED_TDRC_REG
    return cfg


def auc_last_half(curve):
    arr = np.asarray(curve, dtype=float)
    tail = arr[len(arr) // 2:]
    if not np.all(np.isfinite(tail)):
        return float("inf")
    return float(np.mean(tail))


def sweep_env(env_name, steps, alphas, algorithms, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    rows = []
    for alg in algorithms:
        means = []
        stds = []
        for alpha in alphas:
            cfg = make_cfg(alg, alpha)
            scores = []
            for seed in seeds:
                curve = run_once(env_name, alg, cfg, steps, seed)
                scores.append(auc_last_half(curve))
            scores = np.asarray(scores, dtype=float)
            finite = scores[np.isfinite(scores)]
            if len(finite) == 0:
                m, s = float("inf"), float("inf")
            else:
                m = float(np.mean(scores))
                s = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            means.append(m)
            stds.append(s)
            rows.append({
                "algorithm": alg, "alpha": alpha,
                "auc_mean": m, "auc_std": s,
                "n_diverged": int(np.sum(~np.isfinite(scores))),
                "n_seeds": len(seeds),
            })
        results[alg] = {"alphas": list(alphas), "means": means, "stds": stds}

    with open(os.path.join(out_dir, "robustness.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = ["algorithm", "alpha", "auc_mean", "auc_std", "n_diverged", "n_seeds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(out_dir, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def plot_env(env_name, results, out_dir, ylim=None):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 9,
        "savefig.dpi": 600,
    })
    colors = {
        TD: "#7f7f7f", GTD2: "#1f77b4", TDC: "#ff7f0e", TDRC: "#2ca02c",
        GTD2_MP: "#9467bd", HTD: "#8c564b", ETD0: "#17becf",
        STHTD: "#d62728", STHTD_MP: "#000000",
    }
    markers = {
        TD: "o", GTD2: "s", TDC: "^", TDRC: "D", GTD2_MP: "v",
        HTD: "P", ETD0: "X", STHTD: "*", STHTD_MP: "h",
    }
    linestyles = {
        TD: "-", GTD2: "--", TDC: "-.", TDRC: (0, (3, 1, 1, 1)),
        GTD2_MP: (0, (5, 2)), HTD: (0, (3, 1, 1, 1)), ETD0: (0, (1, 1)),
        STHTD: (0, (5, 1, 1, 1)), STHTD_MP: "-",
    }

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for alg, data in results.items():
        alphas = np.asarray(data["alphas"], dtype=float)
        means = np.asarray(data["means"], dtype=float)
        stds = np.asarray(data["stds"], dtype=float)
        ax.plot(
            alphas, means,
            label=alg,
            color=colors.get(alg, None),
            linestyle=linestyles.get(alg, "-"),
            marker=markers.get(alg, "o"),
            markersize=5,
            linewidth=1.6,
        )
        finite = np.isfinite(means) & np.isfinite(stds)
        if np.any(finite):
            ax.fill_between(
                alphas[finite],
                np.maximum(means[finite] - stds[finite], 1e-30),
                means[finite] + stds[finite],
                color=colors.get(alg, "gray"),
                alpha=0.12,
                linewidth=0,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"learning rate $\alpha$")
    ax.set_ylabel("last-50% RMSVE")
    ax.set_title(env_name.replace("_", " "))
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", ncol=2, frameon=False)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"robustness_{env_name}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Step-size robustness sweep.")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", default=os.path.join("paper_results", "robustness"))
    args = parser.parse_args()

    seeds = list(range(30_000, 30_000 + args.seeds))
    for env_name in args.envs:
        steps = DEFAULT_STEPS[env_name]
        out_dir = os.path.join(args.out, env_name)
        print(f"[robustness] env={env_name} steps={steps} seeds={len(seeds)}")
        results = sweep_env(env_name, steps, args.alphas, args.algorithms, seeds, out_dir)
        plot_env(env_name, results, out_dir)


if __name__ == "__main__":
    main()

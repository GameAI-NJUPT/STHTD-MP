# STHTD-MP

Reference implementation and experiment scripts for the STHTD-MP off-policy
prediction algorithm.

STHTD-MP is a single-timescale hybrid temporal-difference method for off-policy
prediction with linear function approximation. It replaces the feature
covariance metric used by GTD2-MP with the symmetric part of the behavior-policy
Bellman matrix, and applies a Mirror-Prox prediction-correction step to the
resulting saddle-point operator.

## Repository layout

```
STHTD-MP/
  src/                          # Python source code (reproduces all experiments)
    paper_prediction_experiments.py
    step_size_robustness.py
    numerical_rate_analysis.py
    refresh_prediction_plots.py
```

Running the scripts in `src/` creates a `paper_results/` directory in the repo
root with all CSV tables, JSON hyperparameter records, `.npz` learning-curve
archives, and PDF figures. `paper_results/` is intentionally not tracked.

## Algorithms implemented

`src/paper_prediction_experiments.py` provides reference implementations of:

- `TD`     -- semi-gradient off-policy TD(0)
- `GTD2`   -- gradient TD with two-timescale auxiliary variable
- `TDC`    -- TD with gradient correction
- `TDRC`   -- TDC with regularized correction and shared learning rate
- `GTD2-MP`-- Mirror-Prox extension of GTD2 (covariance metric)
- `HTD`    -- hybrid TD with mixture parameter eta
- `ETD`    -- emphatic TD(0) with full follow-on trace, no clipping
- `STHTD`  -- single-timescale hybrid TD (behavior-induced metric)
- `STHTD-MP` -- proposed method: Mirror-Prox over STHTD operator

`numerical_rate_analysis.py` is purely matrix-based: it constructs the exact
finite-state matrices A_pi, A_mu, C, H and reports the Mirror-Prox spectral
factors q_C(alpha), q_H(alpha) without any Monte Carlo simulation.

## Environments

All four benchmarks are off-policy and follow standard conventions:

- `two_state`    -- two-state counterexample (Sutton & Barto)
- `baird`        -- Baird's seven-state counterexample
- `random_walk`  -- five-state random walk, behavior 0.5/0.5, target 0.4/0.6, gamma 0.99
- `boyan_chain`  -- thirteen-state Boyan chain with linear features

## Reproducing the experiments

Requirements: Python 3.10+, `numpy`, `matplotlib`.

```
# Main prediction experiments (100 seeds each)
python src/paper_prediction_experiments.py --env two_state    --steps 5000  --eval-seeds 100
python src/paper_prediction_experiments.py --env baird        --steps 2000  --eval-seeds 100
python src/paper_prediction_experiments.py --env random_walk  --steps 5000  --eval-seeds 100
python src/paper_prediction_experiments.py --env boyan_chain  --steps 20000 --eval-seeds 100

# Step-size robustness sweep (30 seeds per cell)
python src/step_size_robustness.py --seeds 30

# Exact mean-operator spectral analysis (no sampling)
python src/numerical_rate_analysis.py

# Regenerate publication-quality learning curves
python src/refresh_prediction_plots.py
```

All results are written to a local `paper_results/` directory.

## Reporting protocol

- Tuning: 8 disjoint seeds; tuning objective is the average error over the last
  20% of the tuning trajectory.
- Evaluation: 100 disjoint seeds for the main comparison, 30 disjoint seeds for
  the step-size robustness sweep.
- Scalar summaries: steady-state AUC (time-average RMSVE over the last 50% of
  the trajectory) and final RMSVE; both reported as mean +/- one sample standard
  deviation.
- Divergent runs are kept as inf / NaN and propagate to the reported mean and
  standard deviation; no clipping is applied.

## License

The source code is released under the MIT License (see `LICENSE`).

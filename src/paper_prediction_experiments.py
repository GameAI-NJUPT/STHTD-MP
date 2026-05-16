import argparse
import csv
import json
import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TD = "TD"
GTD2 = "GTD2"
TDC = "TDC"
TDRC = "TDRC"
GTD2_ST = "GTD2-ST"
GTD2_MP = "GTD2-MP"
HTD = "HTD"
ETD0 = "ETD"
LSTD = "LSTD"
STHTD = "STHTD"
STHTD_MP = "STHTD-MP"
STHTD_MU = "STHTD-Mu"
STHTD_DAMPED = "STHTD-Damped"
STHTD_MP_DAMPED = "STHTD-MP-Damped"


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    rho: float
    done: bool = False


class PredictionEnv:
    name = "base"
    gamma = 0.99

    def reset(self, rng):
        raise NotImplementedError

    def step(self, state, rng):
        raise NotImplementedError

    def phi(self, state):
        raise NotImplementedError

    def error(self, theta):
        raise NotImplementedError

    @property
    def feature_size(self):
        return len(self.phi(self.reset(np.random.default_rng(0))))


class TwoStateEnv(PredictionEnv):
    name = "two_state"
    gamma = 0.9

    def __init__(self):
        self.features = np.array([[1.0], [2.0]])

    def reset(self, rng):
        return 0

    def step(self, state, rng):
        # action 0 keeps/moves to state 0; action 1 moves to state 1.
        action = int(rng.random() >= 0.5)
        next_state = action
        rho = 2.0 if action == 1 else 0.0
        return Transition(state, action, 0.0, next_state, rho)

    def phi(self, state):
        return self.features[state]

    def error(self, theta):
        values = self.features @ theta
        return float(np.sqrt(0.5 * np.sum(values * values)))


class BairdEnv(PredictionEnv):
    name = "baird"
    gamma = 0.99

    def __init__(self):
        self.lower_state = 6
        self.features = np.zeros((7, 8))
        for i in range(self.lower_state):
            self.features[i, i] = 2.0
            self.features[i, 7] = 1.0
        self.features[self.lower_state, 6] = 1.0
        self.features[self.lower_state, 7] = 2.0
        self.d_mu = np.ones(7) / 7.0

    def reset(self, rng):
        return 0

    def step(self, state, rng):
        solid = int(rng.random() < (1.0 / 7.0))
        if solid:
            next_state = self.lower_state
            rho = 7.0
        else:
            next_state = int(rng.integers(0, self.lower_state))
            rho = 0.0
        return Transition(state, solid, 0.0, next_state, rho)

    def phi(self, state):
        return self.features[state]

    def error(self, theta):
        values = self.features @ theta
        return float(np.sqrt(np.sum(self.d_mu * values * values)))


class RandomWalkEnv(PredictionEnv):
    name = "random_walk"
    gamma = 0.99

    def __init__(self):
        self.features = np.zeros((7, 5))
        for state in range(1, 6):
            self.features[state, state - 1] = 1.0
        self.true_values = self._true_values()
        # Behavior-policy stationary distribution over the 5 interior states.
        # Matches the regenerative chain used in the numerical analysis: at
        # terminal transitions the agent resets to state 3.
        self.d_mu_interior = self._behavior_stationary()

    def reset(self, rng):
        return 3

    def step(self, state, rng):
        action = int(rng.random() >= 0.5)
        next_state = state + 1 if action == 1 else state - 1
        reward = 1.0 if next_state == 6 else 0.0
        rho = 1.2 if action == 1 else 0.8
        done = next_state in (0, 6)
        return Transition(state, action, reward, next_state, rho, done)

    def phi(self, state):
        return self.features[state]

    def error(self, theta):
        # RMSVE under the behavior-policy stationary distribution over the
        # five interior states. Terminal states have zero feature, so they
        # are excluded by construction.
        values = self.features @ theta
        diff = values[1:6] - self.true_values[1:6]
        return float(np.sqrt(np.sum(self.d_mu_interior * diff * diff)))

    def _behavior_stationary(self):
        # Regenerative behavior chain: terminal transitions reset to state 3.
        # This matches numerical_rate_analysis.py exactly.
        P_mu = np.zeros((7, 7))
        for s in range(1, 6):
            left = s - 1
            right = s + 1
            P_mu[s, 3 if left == 0 else left] += 0.5
            P_mu[s, 3 if right == 6 else right] += 0.5
        P_mu[0, 3] = 1.0
        P_mu[6, 3] = 1.0
        eigvals, eigvecs = np.linalg.eig(P_mu.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        d = np.real(eigvecs[:, idx])
        if np.sum(d) < 0:
            d = -d
        d = d / np.sum(d)
        d_int = d[1:6]
        d_int = d_int / np.sum(d_int)
        return d_int

    def _true_values(self):
        p_left = 0.4
        p_right = 0.6
        a_mat = np.eye(5)
        b_vec = np.zeros(5)
        for idx, state in enumerate(range(1, 6)):
            left = state - 1
            right = state + 1
            if left != 0:
                a_mat[idx, left - 1] -= self.gamma * p_left
            if right != 6:
                a_mat[idx, right - 1] -= self.gamma * p_right
            else:
                b_vec[idx] += p_right
        values = np.zeros(7)
        values[1:6] = np.linalg.solve(a_mat, b_vec)
        return values


class BoyanChainEnv(PredictionEnv):
    name = "boyan_chain"
    gamma = 0.9

    def __init__(self):
        self.features = np.array([
            [1, 0, 0, 0],
            [0.75, 0.25, 0, 0],
            [0.5, 0.5, 0, 0],
            [0.25, 0.75, 0, 0],
            [0, 1, 0, 0],
            [0, 0.75, 0.25, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.25, 0.75, 0],
            [0, 0, 1, 0],
            [0, 0, 0.75, 0.25],
            [0, 0, 0.5, 0.5],
            [0, 0, 0.25, 0.75],
            [0, 0, 0, 1],
        ], dtype=float)
        self.true_values = self._true_values()
        self.d_mu = self._behavior_stationary()

    def reset(self, rng):
        return 0

    def step(self, state, rng):
        if state == 11:
            return Transition(state, 1, -2.0, 12, 1.0)
        if state == 12:
            return Transition(state, 1, 0.0, 0, 1.0)
        action = 1 if rng.random() < 0.5 else 2
        next_state = min(state + action, 12)
        pi_prob = 0.4 if action == 1 else 0.6
        return Transition(state, action, -3.0, next_state, pi_prob / 0.5)

    def phi(self, state):
        return self.features[state]

    def error(self, theta):
        # RMSVE under the behavior-policy stationary distribution. This is
        # the same inner product as the one used in the numerical analysis.
        values = self.features @ theta
        diff = values - self.true_values
        return float(np.sqrt(np.sum(self.d_mu * diff * diff)))

    def _behavior_stationary(self):
        P_mu = np.zeros((13, 13))
        for s in range(11):
            P_mu[s, s + 1] = 0.5
            P_mu[s, s + 2] = 0.5
        P_mu[11, 12] = 1.0
        P_mu[12, 0] = 1.0
        eigvals, eigvecs = np.linalg.eig(P_mu.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        d = np.real(eigvecs[:, idx])
        if np.sum(d) < 0:
            d = -d
        d = d / np.sum(d)
        d[np.abs(d) < 1e-14] = 0.0
        d = d / np.sum(d)
        return d

    def _true_values(self):
        p = np.zeros((13, 13))
        for state in range(11):
            p[state, state + 1] = 0.4
            p[state, state + 2] = 0.6
        p[11, 12] = 1.0
        p[12, 0] = 1.0
        rewards = np.full(13, -3.0)
        rewards[11] = -2.0
        rewards[12] = 0.0
        return np.linalg.solve(np.eye(13) - self.gamma * p, rewards)


def make_env(name):
    if name == "two_state":
        return TwoStateEnv()
    if name == "baird":
        return BairdEnv()
    if name == "random_walk":
        return RandomWalkEnv()
    if name == "boyan_chain":
        return BoyanChainEnv()
    raise ValueError(f"Unknown environment: {name}")


def initial_theta(env):
    if env.name == "two_state":
        return np.array([10.0])
    if env.name == "baird":
        theta = np.ones(env.feature_size)
        theta[6] = 10.0
        return theta
    if env.name == "boyan_chain":
        return np.ones(env.feature_size)
    return np.zeros(env.feature_size)


class Learner:
    def __init__(self, name, env, alpha, beta=None, eta=0.5, tdrc_reg=1.0, kappa=1.0, lstd_reg=1e-4):
        self.name = name
        self.env = env
        self.alpha = float(alpha)
        self.beta = float(beta if beta is not None else alpha)
        self.eta = float(eta)
        self.tdrc_reg = float(tdrc_reg)
        self.kappa = float(kappa)
        self.lstd_reg = float(lstd_reg)
        self.theta = initial_theta(env)
        self.w = np.zeros(env.feature_size)
        self.a_mat = self.lstd_reg * np.eye(env.feature_size)
        self.b_vec = np.zeros(env.feature_size)
        self.followon = 0.0
        self.prev_rho = 0.0

    def reset_episode(self):
        self.followon = 0.0
        self.prev_rho = 0.0

    def update(self, tr):
        gamma = self.env.gamma
        phi = self.env.phi(tr.state)
        phi_next = self.env.phi(tr.next_state)
        if tr.done:
            phi_next = np.zeros_like(phi_next)
        rho = tr.rho
        delta = tr.reward + gamma * np.dot(self.theta, phi_next) - np.dot(self.theta, phi)
        correction = np.dot(phi, self.w)

        if self.name == TD:
            self.theta += self.alpha * rho * delta * phi
        elif self.name == GTD2:
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction
            self.w += self.beta * (rho * delta - correction) * phi
        elif self.name == TDC:
            self.theta += self.alpha * rho * (delta * phi - gamma * phi_next * correction)
            self.w += self.beta * (rho * delta - correction) * phi
        elif self.name == TDRC:
            self.theta += self.alpha * rho * (delta * phi - gamma * phi_next * correction)
            self.w += self.alpha * ((rho * delta - correction) * phi - self.tdrc_reg * self.w)
        elif self.name == GTD2_ST:
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction
            self.w += self.alpha * (rho * delta - correction) * phi
        elif self.name == GTD2_MP:
            theta_m = self.theta + self.alpha * rho * (phi - gamma * phi_next) * correction
            w_m = self.w + self.alpha * (rho * delta - correction) * phi
            delta_m = tr.reward + gamma * np.dot(theta_m, phi_next) - np.dot(theta_m, phi)
            correction_m = np.dot(phi, w_m)
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction_m
            self.w += self.alpha * (rho * delta_m - correction_m) * phi
        elif self.name == HTD:
            # Hackman-style hybrid TD interpolation: eta=0 recovers TDC's
            # correction term; eta=1 recovers semi-gradient off-policy TD.
            self.theta += self.alpha * rho * (
                delta * phi - gamma * (1.0 - self.eta) * phi_next * correction
            )
            self.w += self.beta * (rho * delta - correction) * phi
        elif self.name == ETD0:
            self.followon = 1.0 + gamma * self.prev_rho * self.followon
            self.theta += self.alpha * rho * self.followon * delta * phi
            self.prev_rho = rho
        elif self.name == LSTD:
            self.a_mat += rho * np.outer(phi, phi - gamma * phi_next)
            self.b_vec += rho * tr.reward * phi
            self.theta = np.linalg.pinv(self.a_mat) @ self.b_vec
        elif self.name == STHTD:
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction
            self.w += self.alpha * (
                (rho * delta - correction + 0.5 * gamma * np.dot(phi_next, self.w)) * phi
                + 0.5 * gamma * correction * phi_next
            )
        elif self.name == STHTD_MP:
            theta_m = self.theta + self.alpha * rho * (phi - gamma * phi_next) * correction
            w_m = self.w + self.alpha * (
                (rho * delta - correction + 0.5 * gamma * np.dot(phi_next, self.w)) * phi
                + 0.5 * gamma * correction * phi_next
            )
            delta_m = tr.reward + gamma * np.dot(theta_m, phi_next) - np.dot(theta_m, phi)
            correction_m = np.dot(phi, w_m)
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction_m
            self.w += self.alpha * (
                (rho * delta_m - correction_m + 0.5 * gamma * np.dot(phi_next, w_m)) * phi
                + 0.5 * gamma * correction_m * phi_next
            )
        elif self.name == STHTD_MU:
            # A more stable variant for Baird-like cases: keep the single-timescale
            # hybrid y-update, but use the behavior Bellman direction in theta.
            # This reduces importance-ratio amplification in the theta block.
            self.theta += self.alpha * (phi - gamma * phi_next) * correction
            self.w += self.alpha * (
                (rho * delta - correction + 0.5 * gamma * np.dot(phi_next, self.w)) * phi
                + 0.5 * gamma * correction * phi_next
            )
        elif self.name == STHTD_DAMPED:
            # kappa=1 recovers STHTD; kappa=0 recovers the GTD2-ST auxiliary block.
            kappa = self.kappa
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction
            self.w += self.alpha * (
                (rho * delta - correction + kappa * 0.5 * gamma * np.dot(phi_next, self.w)) * phi
                + kappa * 0.5 * gamma * correction * phi_next
            )
        elif self.name == STHTD_MP_DAMPED:
            kappa = self.kappa
            theta_m = self.theta + self.alpha * rho * (phi - gamma * phi_next) * correction
            w_m = self.w + self.alpha * (
                (rho * delta - correction + kappa * 0.5 * gamma * np.dot(phi_next, self.w)) * phi
                + kappa * 0.5 * gamma * correction * phi_next
            )
            delta_m = tr.reward + gamma * np.dot(theta_m, phi_next) - np.dot(theta_m, phi)
            correction_m = np.dot(phi, w_m)
            self.theta += self.alpha * rho * (phi - gamma * phi_next) * correction_m
            self.w += self.alpha * (
                (rho * delta_m - correction_m + kappa * 0.5 * gamma * np.dot(phi_next, w_m)) * phi
                + kappa * 0.5 * gamma * correction_m * phi_next
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.name}")


def run_once(env_name, alg_name, cfg, steps, seed):
    env = make_env(env_name)
    learner = Learner(alg_name, env, **cfg)
    rng = np.random.default_rng(seed)
    state = env.reset(rng)
    errors = np.empty(steps + 1)
    errors[0] = env.error(learner.theta)
    for t in range(1, steps + 1):
        tr = env.step(state, rng)
        learner.update(tr)
        if tr.done:
            learner.reset_episode()
            state = env.reset(rng)
        else:
            state = tr.next_state
        errors[t] = env.error(learner.theta)
        # No artificial clipping: divergent runs are kept as inf/nan so that
        # downstream statistics (mean, std, AUC) honestly reflect divergence.
    return errors


def default_grid(alg_name):
    # All grids respect the protocol-level cap alpha <= 0.1 (and beta <= 0.1
    # for two-rate methods) so that no algorithm gets a stepsize the others
    # are forbidden to use.
    alphas = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
    two_rate_alphas = [0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
    two_rate_ratios = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    if alg_name in (GTD2, TDC):
        return [
            {"alpha": a, "beta": a * z, "eta": 0.5}
            for a in two_rate_alphas
            for z in two_rate_ratios
            if a * z <= 0.1
        ]
    if alg_name == TDRC:
        return [
            {"alpha": a, "tdrc_reg": 1.0, "eta": 0.5}
            for a in two_rate_alphas
        ]
    if alg_name == HTD:
        return [
            {"alpha": a, "beta": a * z, "eta": eta}
            for a in two_rate_alphas
            for z in [0.25, 0.5, 1.0, 2.0, 4.0]
            if a * z <= 0.1
            for eta in [0.25, 0.5, 0.75]
        ]
    if alg_name == ETD0:
        etd_alphas = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
        return [{"alpha": a} for a in etd_alphas]
    if alg_name == LSTD:
        return [{"alpha": 1.0, "lstd_reg": r} for r in [1e-6, 1e-4, 1e-2]]
    if alg_name in (STHTD_DAMPED, STHTD_MP_DAMPED):
        return [{"alpha": a, "kappa": k} for a in alphas for k in [0.0, 0.25, 0.5, 0.75, 1.0]]
    return [{"alpha": a} for a in alphas]


def score_curve(curve):
    # last-20% mean; non-finite (divergent) curves get +inf so the tuner
    # naturally avoids them.
    start = int(len(curve) * 0.8)
    tail = np.asarray(curve[start:], dtype=float)
    if not np.all(np.isfinite(tail)):
        return float("inf")
    return float(np.mean(tail))


def tune(env_name, alg_name, steps, seeds):
    best_cfg = None
    best_score = math.inf
    for cfg in default_grid(alg_name):
        scores = []
        for seed in seeds:
            scores.append(score_curve(run_once(env_name, alg_name, cfg, steps, seed)))
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_cfg = cfg
    return best_cfg, best_score


def normal_p_value_from_t(t_value):
    # Normal approximation avoids adding a scipy dependency.
    z = abs(float(t_value))
    return float(math.erfc(z / math.sqrt(2.0)))


def paired_stats(reference_scores, candidate_scores):
    diff = np.asarray(candidate_scores) - np.asarray(reference_scores)
    mean_diff = float(np.mean(diff))
    if len(diff) < 2:
        return mean_diff, math.nan, math.nan
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return mean_diff, math.inf if mean_diff != 0.0 else 0.0, 0.0 if mean_diff != 0.0 else 1.0
    t_value = mean_diff / (sd / math.sqrt(len(diff)))
    return mean_diff, float(t_value), normal_p_value_from_t(t_value)


def summarize(curves, reference_curves=None):
    curves = np.asarray(curves, dtype=float)
    n_runs, T = curves.shape
    final_scores = curves[:, -1]
    # AUC = time-average RMSVE over the last 50% of the trajectory.
    # This skips the burn-in transient (where high-variance algorithms such as
    # ETD can exhibit early spikes on the 2-state counterexample) and reports
    # the steady-state behaviour, consistent with the learning-curve reading
    # in Chen et al. (UAI 2023, "Modified Retrace ...").
    auc_start = T // 2
    auc_scores = np.mean(curves[:, auc_start:], axis=1)
    # Count divergent runs so the table is honest about them.
    diverged_mask = ~np.all(np.isfinite(curves), axis=1)
    n_diverged = int(np.sum(diverged_mask))
    summary = {
        "n_runs": n_runs,
        "n_diverged": n_diverged,
        "final_mean": float(np.mean(final_scores)),
        "final_std": float(np.std(final_scores, ddof=1)) if len(final_scores) > 1 else 0.0,
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores, ddof=1)) if len(auc_scores) > 1 else 0.0,
    }
    if reference_curves is not None:
        ref_auc = np.mean(np.asarray(reference_curves), axis=1)
        mean_diff, t_value, p_value = paired_stats(ref_auc, auc_scores)
        summary.update({"auc_paired_diff_vs_ref": mean_diff, "auc_t_vs_ref": t_value, "auc_p_vs_ref": p_value})
    return summary


def plot_results(out_dir, env_name, results, reference_name, suffix="comparison"):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 9,
        "savefig.dpi": 600,
    })
    colors = {
        TD: "#7f7f7f", GTD2: "#1f77b4", TDC: "#ff7f0e", TDRC: "#2ca02c", GTD2_ST: "#2ca02c",
        GTD2_MP: "#9467bd", HTD: "#8c564b", ETD0: "#17becf", LSTD: "#aec7e8", STHTD: "#d62728",
        STHTD_MP: "#000000", STHTD_MU: "#bcbd22",
        STHTD_DAMPED: "#e377c2", STHTD_MP_DAMPED: "#4d4d4d",
    }
    linestyles = {
        TD: "-", GTD2: "--", TDC: "-.", TDRC: (0, (3, 1, 1, 1)), GTD2_ST: ":",
        GTD2_MP: (0, (5, 2)), HTD: (0, (3, 1, 1, 1)), ETD0: (0, (1, 1)),
        LSTD: (0, (7, 1, 1, 1)), STHTD: (0, (5, 1, 1, 1)),
        STHTD_MP: "-", STHTD_MU: (0, (2, 2)),
        STHTD_DAMPED: (0, (4, 2, 1, 2)), STHTD_MP_DAMPED: (0, (8, 2)),
    }
    markers = {
        TD: None, GTD2: "o", TDC: "s", TDRC: "^", GTD2_ST: "^",
        GTD2_MP: "D", HTD: "v", ETD0: "P", LSTD: "X",
        STHTD: "*", STHTD_MP: "s", STHTD_MU: "h",
        STHTD_DAMPED: "<", STHTD_MP_DAMPED: ">",
    }
    mean_values = []
    for alg_name, curves in results.items():
        if alg_name == LSTD:
            continue
        curves = np.asarray(curves, dtype=float)
        with np.errstate(invalid="ignore"):
            mean_values.append(np.nanmean(np.where(np.isfinite(curves), curves, np.nan), axis=0))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for alg_name, curves in results.items():
        if alg_name == LSTD:
            continue
        curves = np.asarray(curves, dtype=float)
        # Mask non-finite entries (divergent runs) so the plotted mean/std are
        # computed from the finite subset of seeds at each step.
        finite_curves = np.where(np.isfinite(curves), curves, np.nan)
        x = np.arange(curves.shape[1])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(finite_curves, axis=0)
            spread = (
                np.nanstd(finite_curves, axis=0, ddof=1)
                if curves.shape[0] > 1 else np.zeros_like(mean)
            )
        lw = 1.8 if alg_name in (reference_name, STHTD_MP) else 1.1
        marker = markers.get(alg_name)
        marker_every = max(1, len(x) // 12)
        ax.plot(
            x,
            mean,
            label=alg_name,
            color=colors.get(alg_name),
            linestyle=linestyles.get(alg_name, "-"),
            linewidth=lw,
            marker=marker,
            markevery=marker_every if marker is not None else None,
            markersize=4.2 if alg_name != STHTD_MP else 4.8,
            markerfacecolor="white" if marker is not None else None,
            markeredgewidth=0.9,
        )
        ax.fill_between(x, mean - spread, mean + spread, color=colors.get(alg_name), alpha=0.18, linewidth=0)
    ax.set_xlabel("Steps")
    ax.set_ylabel("RMSVE")
    if env_name == "two_state":
        ax.set_ylim(bottom=-2.0, top=30.0)
    elif env_name == "boyan_chain":
        # Start from a negative lower limit so the shaded std band around
        # values close to zero remains visible to the reader.
        ax.set_ylim(bottom=-2.0, top=28.0)
    elif env_name == "random_walk":
        ax.set_ylim(bottom=-0.05, top=1.0)
    elif env_name == "baird":
        ax.set_ylim(bottom=-2.0, top=42.0)
    elif mean_values:
        finite_means = np.concatenate([m[np.isfinite(m)] for m in mean_values])
        if finite_means.size:
            upper = float(np.quantile(finite_means, 0.98) * 1.15)
            if upper > 0:
                ax.set_ylim(bottom=0.0, top=upper)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out_dir, f"{env_name}_{suffix}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{env_name}_{suffix}.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fair prediction baselines with CI and paired tests.")
    parser.add_argument("--env", choices=["two_state", "baird", "random_walk", "boyan_chain"], default="baird")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tune-seeds", type=int, default=8)
    parser.add_argument("--eval-seeds", type=int, default=100)
    parser.add_argument("--out", default=os.path.join("paper_results", "prediction"))
    parser.add_argument("--algorithms", nargs="+", default=[TD, GTD2, TDC, TDRC, GTD2_MP, HTD, ETD0, STHTD, STHTD_MP])
    parser.add_argument("--reference", default=STHTD_MP)
    args = parser.parse_args()

    default_steps = {"two_state": 5000, "baird": 2000, "random_walk": 5000, "boyan_chain": 20000}
    steps = args.steps if args.steps is not None else default_steps[args.env]
    out_dir = os.path.join(args.out, args.env)
    os.makedirs(out_dir, exist_ok=True)

    tune_seeds = list(range(10_000, 10_000 + args.tune_seeds))
    eval_seeds = list(range(20_000, 20_000 + args.eval_seeds))
    tuned = {}
    tune_rows = []
    results = {}

    for alg_name in args.algorithms:
        cfg, score = tune(args.env, alg_name, max(steps // 2, 200), tune_seeds)
        tuned[alg_name] = cfg
        tune_rows.append({"algorithm": alg_name, "score": score, **cfg})
        curves = [run_once(args.env, alg_name, cfg, steps, seed) for seed in eval_seeds]
        results[alg_name] = np.asarray(curves)

    np.savez_compressed(os.path.join(out_dir, "curves.npz"), **results)
    with open(os.path.join(out_dir, "selected_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "tuning_scores.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in tune_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tune_rows)

    reference_curves = results.get(args.reference)
    summary_rows = []
    for alg_name, curves in results.items():
        row = {"algorithm": alg_name, **summarize(curves, reference_curves if alg_name != args.reference else None)}
        summary_rows.append(row)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_results(out_dir, args.env, results, args.reference)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()

import csv
import json
import math
import os

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_results", "numerical_rate_analysis")


def stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    d = np.real(eigvecs[:, idx])
    if np.sum(d) < 0:
        d = -d
    d = d / np.sum(d)
    d[np.abs(d) < 1e-14] = 0.0
    return d


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def symmetrize(M):
    return 0.5 * (M + M.T)


def key_matrix_stats(A_pi, metric, tol=1e-10):
    key = symmetrize(A_pi.T @ np.linalg.pinv(metric) @ A_pi)
    eig = np.linalg.eigvalsh(key)
    eig_min = float(np.min(eig))
    eig_max = float(np.max(eig))
    if eig_min > tol:
        condition = eig_max / eig_min
        optimal_gd_factor = (condition - 1.0) / (condition + 1.0)
    else:
        condition = math.inf
        optimal_gd_factor = 1.0
    return key, eig_min, eig_max, condition, optimal_gd_factor


def mp_stats(K, alpha_grid):
    dim = K.shape[0]
    eye = np.eye(dim)
    best_mp = (math.inf, None)
    max_stable_mp = 0.0
    for alpha in alpha_grid:
        rho_mp = spectral_radius(eye - alpha * K + alpha * alpha * (K @ K))
        if rho_mp < best_mp[0]:
            best_mp = (rho_mp, alpha)
        if rho_mp < 1.0:
            max_stable_mp = alpha
    return best_mp, max_stable_mp


def analyze(name, env_key, gamma, features, transitions, d):
    d = np.asarray(d, dtype=float)
    features = np.asarray(features, dtype=float)
    n_features = features.shape[1]
    A_pi = np.zeros((n_features, n_features))
    A_mu = np.zeros((n_features, n_features))
    C = np.zeros((n_features, n_features))
    b = np.zeros(n_features)

    for tr in transitions:
        s, mu_prob, rho, reward, next_state, done = tr
        phi = features[s]
        phi_next = np.zeros(n_features) if done else features[next_state]
        prob = d[s] * mu_prob
        A_pi += prob * rho * np.outer(phi, phi - gamma * phi_next)
        A_mu += prob * np.outer(phi, phi - gamma * phi_next)
        C += prob * np.outer(phi, phi)
        b += prob * rho * reward * phi

    H = 0.5 * (A_mu + A_mu.T)
    K_c = np.block([[np.zeros_like(A_pi), -A_pi.T], [A_pi, C]])
    K_h = np.block([[np.zeros_like(A_pi), -A_pi.T], [A_pi, H]])
    G_c = -K_c
    G_h = -K_h
    h_margin_c = -float(np.max(np.real(np.linalg.eigvals(G_c))))
    h_margin_h = -float(np.max(np.real(np.linalg.eigvals(G_h))))
    c_min = float(np.min(np.linalg.eigvalsh(C)))
    c_max = float(np.max(np.linalg.eigvalsh(C)))
    h_min = float(np.min(np.linalg.eigvalsh(H)))
    h_max = float(np.max(np.linalg.eigvalsh(H)))
    L_c = float(np.linalg.norm(K_c, 2))
    L_h = float(np.linalg.norm(K_h, 2))
    A_min_sv = float(np.min(np.linalg.svd(A_pi, compute_uv=False)))
    B_c, B_c_min, B_c_max, B_c_cond, B_c_gd = key_matrix_stats(A_pi, C)
    B_h, B_h_min, B_h_max, B_h_cond, B_h_gd = key_matrix_stats(A_pi, H)
    key_matrix_condition_satisfied = (
        A_min_sv > 1e-10
        and c_min > 1e-10
        and h_min > 1e-10
        and math.isfinite(B_c_cond)
        and math.isfinite(B_h_cond)
        and B_h_min > B_c_min
        and B_h_cond <= B_c_cond
        and B_h_gd <= B_c_gd
    )

    dim = K_c.shape[0]
    alpha_grid = np.logspace(-5, 0, 20000)
    best_gtd2_mp, max_stable_gtd2_mp = mp_stats(K_c, alpha_grid)
    best_sthtd_mp, max_stable_sthtd_mp = mp_stats(K_h, alpha_grid)
    relative_best_gain = (best_gtd2_mp[0] - best_sthtd_mp[0]) / best_gtd2_mp[0] if best_gtd2_mp[0] > 0 else 0.0

    return {
        "environment": name,
        "dim": n_features,
        "lambda_min_C_GTD2": c_min,
        "lambda_max_C_GTD2": c_max,
        "lambda_min_H": h_min,
        "lambda_max_H": h_max,
        "sigma_min_A_pi": A_min_sv,
        "lambda_min_key_AinvC_A_GTD2": B_c_min,
        "lambda_max_key_AinvC_A_GTD2": B_c_max,
        "condition_key_AinvC_A_GTD2": B_c_cond,
        "optimal_gd_factor_key_GTD2": B_c_gd,
        "lambda_min_key_AinvH_A_HGTD2": B_h_min,
        "lambda_max_key_AinvH_A_HGTD2": B_h_max,
        "condition_key_AinvH_A_HGTD2": B_h_cond,
        "optimal_gd_factor_key_HGTD2": B_h_gd,
        "key_matrix_condition_satisfied": key_matrix_condition_satisfied,
        "operator_L_GTD2_MP_C": L_c,
        "operator_L_STHTD_MP_H": L_h,
        "operator_L_reduction_H_vs_C": (L_c - L_h) / L_c if L_c > 0 else 0.0,
        "hurwitz_margin_GTD2_MP_C": h_margin_c,
        "hurwitz_margin_STHTD_MP_H": h_margin_h,
        "alpha_GTD2_MP_best_mean": best_gtd2_mp[1],
        "rho_GTD2_MP_best_mean": best_gtd2_mp[0],
        "alpha_STHTD_MP_best_mean": best_sthtd_mp[1],
        "rho_STHTD_MP_best_mean": best_sthtd_mp[0],
        "best_mean_relative_gain": relative_best_gain,
        "max_stable_alpha_GTD2_MP": max_stable_gtd2_mp,
        "max_stable_alpha_STHTD_MP": max_stable_sthtd_mp,
        "stable_range_ratio_STHTD_MP_over_GTD2_MP": max_stable_sthtd_mp / max_stable_gtd2_mp if max_stable_gtd2_mp > 0 else math.inf,
    }


def two_state():
    features = np.array([[1.0], [2.0]])
    d = np.array([0.5, 0.5])
    transitions = []
    for s in [0, 1]:
        transitions.append((s, 0.5, 0.0, 0.0, 0, False))
        transitions.append((s, 0.5, 2.0, 0.0, 1, False))
    return analyze("Two-state", "two_state", 0.9, features, transitions, d)


def baird():
    features = np.zeros((7, 8))
    for i in range(6):
        features[i, i] = 2.0
        features[i, 7] = 1.0
    features[6, 6] = 1.0
    features[6, 7] = 2.0
    d = np.ones(7) / 7.0
    transitions = []
    for s in range(7):
        transitions.append((s, 1.0 / 7.0, 7.0, 0.0, 6, False))
        for ns in range(6):
            transitions.append((s, (6.0 / 7.0) * (1.0 / 6.0), 0.0, 0.0, ns, False))
    return analyze("Baird", "baird", 0.99, features, transitions, d)


def random_walk():
    features = np.zeros((7, 5))
    for s in range(1, 6):
        features[s, s - 1] = 1.0
    # Regenerative behavior chain: terminal transitions reset to state 3.
    P_mu = np.zeros((7, 7))
    for s in range(1, 6):
        left = s - 1
        right = s + 1
        P_mu[s, 3 if left == 0 else left] += 0.5
        P_mu[s, 3 if right == 6 else right] += 0.5
    P_mu[0, 3] = 1.0
    P_mu[6, 3] = 1.0
    d_full = stationary_distribution(P_mu)
    transitions = []
    for s in range(1, 6):
        left = s - 1
        right = s + 1
        transitions.append((s, 0.5, 0.8, 0.0, left, left in (0, 6)))
        transitions.append((s, 0.5, 1.2, 1.0 if right == 6 else 0.0, right, right in (0, 6)))
    return analyze("Random Walk", "random_walk", 0.99, features, transitions, d_full)


def boyan_chain():
    features = np.array([
        [1, 0, 0, 0], [0.75, 0.25, 0, 0], [0.5, 0.5, 0, 0], [0.25, 0.75, 0, 0],
        [0, 1, 0, 0], [0, 0.75, 0.25, 0], [0, 0.5, 0.5, 0], [0, 0.25, 0.75, 0],
        [0, 0, 1, 0], [0, 0, 0.75, 0.25], [0, 0, 0.5, 0.5], [0, 0, 0.25, 0.75],
        [0, 0, 0, 1],
    ], dtype=float)
    P_mu = np.zeros((13, 13))
    for s in range(11):
        P_mu[s, s + 1] = 0.5
        P_mu[s, s + 2] = 0.5
    P_mu[11, 12] = 1.0
    P_mu[12, 0] = 1.0
    d = stationary_distribution(P_mu)
    transitions = []
    for s in range(11):
        transitions.append((s, 0.5, 0.4 / 0.5, -3.0, s + 1, False))
        transitions.append((s, 0.5, 0.6 / 0.5, -3.0, s + 2, False))
    transitions.append((11, 1.0, 1.0, -2.0, 12, False))
    transitions.append((12, 1.0, 1.0, 0.0, 0, False))
    return analyze("Boyan Chain", "boyan_chain", 0.9, features, transitions, d)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [two_state(), baird(), random_walk(), boyan_chain()]
    with open(os.path.join(OUT, "rate_constants.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(OUT, "rate_constants.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

"""
run_phase1.py
Phase I estimation accuracy experiment  (Figure 1 of the paper).

Tests whether Algorithm 1 can recover the true U₀, σ₀, and Σ_x from IC data
for N ∈ {500, 1000, 2000, 5000},  R = 100 replications each.

The true model parameters (A₀, B₀, Ψ₀, σ₀) are loaded from config.
DyPPCA.fit() is used (NOT from_true_model), since the whole point is to
evaluate Phase I estimation accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import config
from data_generator import build_ic_model, simulate_ic
from methods.dyppca import DyPPCA


def run_phase1(N_values=None, R=100, seed=config.SEED, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    if N_values is None:
        N_values = [500, 1000, 2000, 5000]

    # Fixed true IC model (parameters from the paper, equations 63–64)
    ic_model = build_ic_model(
        p      = config.P,
        q      = config.Q,
        sigma0 = config.SIGMA0,
        A0     = config.A0,
        B0     = config.B0,
        Psi0   = config.PSI0,
    )
    U0     = ic_model["U0"]
    A0     = ic_model["A0"]
    sigma0 = ic_model["sigma0"]
    Sigma_x_true = A0 @ A0.T + sigma0 * np.eye(config.P)

    results = {N: {"EU": [], "Esigma": [], "ESigmax": []} for N in N_values}

    for N in N_values:
        print(f"  N = {N} ...", flush=True)
        for _ in range(R):
            X     = simulate_ic(ic_model, N + 1, rng=rng)
            model = DyPPCA(q=config.Q).fit(X)

            # EU: subspace projection error  ‖ÛÛᵀ − U₀U₀ᵀ‖_F
            EU = np.linalg.norm(
                model.U @ model.U.T - U0 @ U0.T, "fro"
            )

            # Eσ: noise variance error
            Esigma = abs(model.sigma0 - sigma0)

            # EΣx: marginal covariance error  ‖Σ̂_x − Σ_x‖_F
            Sigma_x_hat = (model.U @ np.diag(model.Lambda0) @ model.U.T
                           + model.sigma0 * np.eye(config.P))
            ESigmax = np.linalg.norm(Sigma_x_hat - Sigma_x_true, "fro")

            results[N]["EU"].append(EU)
            results[N]["Esigma"].append(Esigma)
            results[N]["ESigmax"].append(ESigmax)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["$E_U$ (subspace)", "$E_\\sigma$ (noise var.)", "$E_{\\Sigma_x}$ (covariance)"]
    keys   = ["EU", "Esigma", "ESigmax"]
    colors = ["tab:blue", "tab:red", "tab:green"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        data = [results[N][key] for N in N_values]
        bp = ax.boxplot(data, labels=N_values, patch_artist=True,
                        medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_xlabel("Phase I sample size $N$")
        ax.set_ylabel("Estimation error")

    #fig.suptitle(f"Phase I Estimation Accuracy  "
                 #f"(p={config.P}, q={config.Q}, {R} replications)",
                 #fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "phase1_estimation.pdf"), dpi=300)
    fig.savefig(os.path.join(save_dir, "phase1_estimation.png"), dpi=300)
    plt.close(fig)
    print(f"Saved Phase I figure → {save_dir}/phase1_estimation.{{pdf,png}}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'N':>6}  {'EU mean':>10}  {'EU std':>9}  "
          f"{'Esigma mean':>12}  {'Esigma std':>11}  "
          f"{'ESigmax mean':>13}  {'ESigmax std':>12}")
    for N in N_values:
        r = results[N]
        print(f"{N:>6}  "
              f"{np.mean(r['EU']):>10.4f}  {np.std(r['EU']):>9.4f}  "
              f"{np.mean(r['Esigma']):>12.6f}  {np.std(r['Esigma']):>11.6f}  "
              f"{np.mean(r['ESigmax']):>13.4f}  {np.std(r['ESigmax']):>12.4f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase I estimation accuracy experiment")
    print(f"  p={config.P}  q={config.Q}  sigma0={config.SIGMA0}")
    print("=" * 60)
    run_phase1()

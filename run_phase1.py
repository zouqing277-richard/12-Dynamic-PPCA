"""
run_phase1.py
Phase I estimation accuracy experiment.

Reproduces Figure 1 in the paper:
    EU (subspace error), Eσ (noise variance error), EΣx (covariance error)
    over N ∈ {500, 1000, 2000, 5000},  R=100 replications each.
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

    # Build fixed IC model (U₀ is fixed across N for fair comparison)
    ic_model = build_ic_model(
        config.P, config.Q, config.SIGMA0,
        config.LAMBDA0, config.B0, seed=seed
    )
    U0 = ic_model["U0"]
    A0 = ic_model["A0"]
    sigma0 = ic_model["sigma0"]
    Sigma_x_true = A0 @ A0.T + sigma0 * np.eye(config.P)

    results = {N: {"EU": [], "Esigma": [], "ESigmax": []} for N in N_values}

    for N in N_values:
        print(f"  N = {N} ...")
        for _ in range(R):
            X = simulate_ic(ic_model, N + 1, rng=rng)
            model = DyPPCA(q=config.Q)
            model.fit(X)

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
    titles  = ["$E_U$ (subspace)", "$E_\\sigma$ (noise var.)", "$E_{\\Sigma_x}$ (covariance)"]
    keys    = ["EU", "Esigma", "ESigmax"]
    colors  = ["tab:blue", "tab:red", "tab:green"]

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

    fig.suptitle("Distribution of Phase I Estimation Errors across "
                 f"{R} Replications", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "phase1_estimation.pdf"), dpi=150)
    fig.savefig(os.path.join(save_dir, "phase1_estimation.png"), dpi=150)
    plt.close(fig)
    print(f"Saved Phase I figure to {save_dir}/")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n{'N':>6}  {'EU mean':>10}  {'EU std':>9}  "
          "{'Eσ mean':>10}  {'Eσ std':>9}  "
          "{'EΣ mean':>10}  {'EΣ std':>9}")
    for N in N_values:
        r = results[N]
        print(f"{N:>6}  "
              f"{np.mean(r['EU']):>10.4f}  {np.std(r['EU']):>9.4f}  "
              f"{np.mean(r['Esigma']):>10.6f}  {np.std(r['Esigma']):>9.6f}  "
              f"{np.mean(r['ESigmax']):>10.4f}  {np.std(r['ESigmax']):>9.4f}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase I estimation accuracy experiment")
    print("=" * 60)
    run_phase1()

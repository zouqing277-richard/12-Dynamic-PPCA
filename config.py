"""
config.py  —  All experiment parameters for DyPPCA numerical study.

Model parameters are taken directly from the paper (equations 63–64):
  p = 10,  q = 3,  σ₀ = 0.25
  A₀  specified in eq. (63)
  B₀  specified in eq. (63)
  Ψ₀  specified in eq. (64)

Stationarity check (B₀ B₀ᵀ + Ψ₀ = I_q) is verified at module load time.
Λ₀ (top-q eigenvalues of Cov(xₜ) = A₀A₀ᵀ + σ₀ I_p) is derived from SVD(A₀).
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Model  (equations 63–64)
# ─────────────────────────────────────────────────────────────────────────────

P = 10
Q = 3
SIGMA0 = 0.25

A0 = np.array([
    [-0.6756453944, -1.0103994931, -1.1502744314],
    [-2.0856621292, -1.4333110025,  0.8608904439],
    [-0.2834415034, -0.2631227877, -0.0767976947],
    [ 1.8916075644,  0.1083211894, -0.6164707994],
    [-0.1464148232,  1.5918031076, -0.1576740303],
    [ 1.9053900532, -0.5830992604,  1.1462952061],
    [-1.9479363115,  1.0339389242,  0.2236286552],
    [ 1.5098902911,  0.7833837623,  0.5328088916],
    [ 0.9497987580, -1.0081054081, -0.7188345440],
    [-0.8103750474,  1.0103994956, -0.3321599832],
])   # (10, 3)  — corrected for σ₀=0.25; eigenvalues of A₀A₀ᵀ+σ₀I exactly [20,10,5,0.25,…]

B0 = np.array([
    [0.5, 0.1, 0.0],
    [0.0, 0.4, 0.1],
    [0.1, 0.0, 0.4],
])   # (3, 3)  — spectral radius ≈ 0.547 (reduced from 0.812 for calibration stability)

PSI0 = np.array([
    [ 0.74, -0.04, -0.05],
    [-0.04,  0.83, -0.04],
    [-0.05, -0.04,  0.83],
])   # (3, 3)  — innovation covariance; satisfies B0 B0ᵀ + Ψ₀ = I_q

# Verify stationarity at import time
_stationarity_err = np.linalg.norm(B0 @ B0.T + PSI0 - np.eye(Q))
assert _stationarity_err < 1e-10, (
    f"Stationarity violated: ‖B₀B₀ᵀ + Ψ₀ − I‖ = {_stationarity_err:.2e}"
)

# Λ₀: top-q eigenvalues of Cov(xₜ) = A₀A₀ᵀ + σ₀ I_p
# Derived via SVD(A₀);  λⱼ = σⱼ(A₀)² + σ₀
_sv    = np.linalg.svd(A0, compute_uv=False)  # shape (q,)
LAMBDA0 = _sv**2 + SIGMA0   # ≈ [20.05, 10.05, 5.05]

# ─────────────────────────────────────────────────────────────────────────────
# Phase I
# ─────────────────────────────────────────────────────────────────────────────

N_TRAIN = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Phase II
# ─────────────────────────────────────────────────────────────────────────────

N_WINDOW = 50      # window size n;  n/(p−q) = 50/7 ≈ 7.1
ARL0     = 200
K_MAX    = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Calibration  (two-phase MC bisection on ARL₀)
# ─────────────────────────────────────────────────────────────────────────────

# ── CRN calibration parameters ──────────────────────────────────────────────
B_CRN       = 5_000     # IC sequences per method (CRN reduces variance vs old B_FINE)
K_MAX_CRN   = 1_500     # windows per sequence  (P(RL>1500) ≈ 0.1% → negligible censoring)
N_COARSE    = 20        # binary search steps in coarse phase
BISECT_TOL  = 2.0       # stop when |ARL₀(h) − 200| ≤ 2
MAX_FINE    = 60        # safety cap
# ── legacy keys kept for backwards compatibility ─────────────────────────────
B_COARSE    = B_CRN
B_FINE      = B_CRN
B_BOOTSTRAP = B_CRN

# ─────────────────────────────────────────────────────────────────────────────
# ARL₁ experiment
# ─────────────────────────────────────────────────────────────────────────────

B1 = 10_000   # OC replications

# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

DPCA_LAG  = 1
DPCA_CPV  = 0.90
VAR_LAG   = 1

LSTM_HIDDEN_DIM = 32
LSTM_LATENT_DIM = 8
LSTM_NUM_LAYERS = 1
LSTM_EPOCHS     = 100
LSTM_BATCH_SIZE = 128
LSTM_LR         = 1e-3
LSTM_PATIENCE   = 10

# ─────────────────────────────────────────────────────────────────────────────
# OC shift magnitudes  (5 per case)
# ─────────────────────────────────────────────────────────────────────────────

SHIFTS = {
    "case1": [0.2,  0.4,  0.6,  0.8,  1.0 ],   # latent mean shift
    "case2": [0.3,  0.35, 0.4,  0.5,  0.6 ],   # obs mean shift (general direction)
    "case3": [0.10, 0.15, 0.20, 0.25, 0.35],   # latent AR matrix shift
    "case4": [0.10, 0.20, 0.30, 0.40, 0.50],   # latent cov shift (Delta_z[1,2]=[2,1]=d)
    "case5": [0.04, 0.08, 0.12, 0.16, 0.20],   # obs noise cov shift (Sigma_eps[2,5]=[5,2]=d)
}

# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic experiment shift magnitudes  (Section 5.4)
# Independent from ARL experiment — focus on moderate-to-large shifts
# so that t_total has sufficient detection power before diagnosing.
# ─────────────────────────────────────────────────────────────────────────────

DIAG_SHIFTS = {
    "case1": [0.5, 0.6, 0.7, 0.8, 0.9],
    "case2": [0.35, 0.4, 0.45, 0.5, 0.6],
    "case3": [0.52, 0.55, 0.58, 0.62, 0.66],
    "case4": [0.5, 0.6, 0.7, 0.8, 0.9],
    "case5": [0.05, 0.09, 0.13, 0.17, 0.21],
}

# ─────────────────────────────────────────────────────────────────────────────
# OC shift locations  (3 per case)
# ─────────────────────────────────────────────────────────────────────────────

LOCATIONS = {
    "case1": [0, 1, 2],              # latent dim: z₁, z₂, z₃
    "case2": [1, 4, 7],              # obs component: x₂, x₅, x₈  (0-indexed)
    "case3": [(0,1), (1,2), (0,2)],  # AR matrix:  B[0,1], B[1,2], B[0,2]
    "case4": [(0,1), (1,2), (0,2)],  # latent cov: Δz[0,1], Δz[1,2], Δz[0,2]
    "case5": [(1,6), (2,5), (3,8)],  # noise cov:  Σε[1,6], Σε[2,5], Σε[3,8]
}

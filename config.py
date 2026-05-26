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
    [-0.6765, -1.0130, -1.1563],
    [-2.0883, -1.4370,  0.8654],
    [-0.2838, -0.2638, -0.0772],
    [ 1.8940,  0.1086, -0.6197],
    [-0.1466,  1.5959, -0.1585],
    [ 1.9078, -0.5846,  1.1523],
    [-1.9504,  1.0366,  0.2248],
    [ 1.5118,  0.7854,  0.5356],
    [ 0.9510, -1.0107, -0.7226],
    [-0.8114,  1.0130, -0.3339],
])   # (10, 3)

B0 = np.array([
    [0.8, 0.1, 0.0],
    [0.0, 0.6, 0.1],
    [0.1, 0.0, 0.4],
])   # (3, 3)

PSI0 = np.array([
    [ 0.35, -0.06, -0.08],
    [-0.06,  0.63, -0.04],
    [-0.08, -0.04,  0.83],
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

B_COARSE    = 300
N_COARSE    = 15
B_FINE      = 10_000    # SE(ARL₀) ≈ 200/√10000 = 2.0
BISECT_TOL  = 2.0       # stop when |ARL₀(h) − 200| ≤ 2
MAX_FINE    = 60
B_BOOTSTRAP = 5000

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
    "case2": [0.2,  0.4,  0.6,  0.8,  1.0 ],   # obs noise mean shift
    "case3": [0.05, 0.10, 0.15, 0.20, 0.25],   # latent AR matrix shift
    "case4": [0.10, 0.20, 0.30, 0.40, 0.50],   # latent covariance shift
    "case5": [0.25, 0.50, 0.75, 1.00, 1.25],   # obs noise cov shift
}

# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

"""
config.py  —  All experiment parameters for DyPPCA numerical study.
Edit this file only; everything else reads from here.
"""
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
P      = 10
Q      = 3
SIGMA0 = 0.2
LAMBDA0 = np.array([20., 10., 5.])
B0 = np.array([[0.6, 0.1, 0.0],
               [0.0, 0.5, 0.1],
               [0.1, 0.0, 0.4]])

# ─────────────────────────────────────────────────────────────────────────────
# Phase I
# ─────────────────────────────────────────────────────────────────────────────
N_TRAIN = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Phase II
# ─────────────────────────────────────────────────────────────────────────────
N_WINDOW = 60      # n/(p-q) = 60/7 ≈ 8.6 → stable covariance estimation
ARL0     = 200
K_MAX    = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Calibration  (two-phase MC bisection on ARL₀)
# ─────────────────────────────────────────────────────────────────────────────
B_COARSE    = 300
N_COARSE    = 15
B_FINE      = 10_000    # SE(ARL₀) ≈ 200/√10000 = 2.0
# N_FINE removed: fine phase runs until convergence (not fixed steps)
BISECT_TOL  = 2.0       # stop when |ARL₀(h) − 200| ≤ 2
MAX_FINE    = 60        # safety cap against infinite loop
B_BOOTSTRAP = 5000

# ─────────────────────────────────────────────────────────────────────────────
# ARL₁ experiment
# ─────────────────────────────────────────────────────────────────────────────
B1 = 10_000   # OC replications (Phase I fixed, only OC sequences vary)

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
    "case3": [0.05, 0.10, 0.15, 0.20, 0.25],   # latent AR matrix shift (ΔB=d·E₁₂)
    "case4": [0.10, 0.20, 0.30, 0.40, 0.50],   # latent covariance shift
    "case5": [0.25, 0.50, 0.75, 1.00, 1.25],   # local obs noise cov shift
}

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

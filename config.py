"""
config.py  ─  All experiment parameters for DyPPCA numerical study.
"""
import numpy as np

# Model
P      = 10
Q      = 3
SIGMA0 = 0.2
LAMBDA0 = np.array([20, 10, 5])
B0 = np.array([[0.6, 0.1, 0.0],
               [0.0, 0.5, 0.1],
               [0.1, 0.0, 0.4]])

# Phase I
N_TRAIN = 2000

# Phase II
N_WINDOW = 50
ARL0     = 200
K_MAX    = 2000

# Calibration (two-phase bisection on ARL0)
B_COARSE    = 300
N_COARSE    = 15
B_FINE      = 2000
N_FINE      = 10
BISECT_TOL  = 2.0
B_BOOTSTRAP = 5000

# ARL1 experiment
B1 = 10_000   # OC replications (Phase I fixed, only OC varies)

# Baselines
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

# OC shift magnitudes (5 per case)
SHIFTS = {
    "case1": [0.2,  0.4,  0.6,  0.8,  1.0 ],
    "case2": [0.2,  0.4,  0.6,  0.8,  1.0 ],
    "case3": [0.05, 0.10, 0.15, 0.20, 0.25],
    "case4": [0.10, 0.20, 0.30, 0.40, 0.50],
    "case5": [0.25, 0.50, 0.75, 1.00, 1.25],
}

SEED = 42

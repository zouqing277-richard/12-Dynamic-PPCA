"""
evaluation.py
ARL₁ estimation for all OC cases.

Design
──────
Calibration phase  (calibration.py):
    Every (method, statistic) pair gets its own ARL₀-calibrated threshold.

OC evaluation phase  (this file):
    For each case, only the *theoretically relevant* (method, statistic) pairs
    are compared (see OC_COMPARISON_STATS).  DyPPCA's individual components
    expose which kind of fault is occurring; t_total is always included as the
    all-purpose monitor.

Alarm rule
──────────
For every comparison pair:  alarm  ⟺  stat[si] > ucl[stat_name]
Single-threshold, no OR-combining across statistics.
"""

import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Statistic index registry  (must match monitor_window return order)
# ─────────────────────────────────────────────────────────────────────────────

STAT_INDEX: Dict[str, Dict[str, int]] = {
    "dyppca": {
        "t1":      0,   # latent mean component
        "t2":      1,   # residual mean component
        "t3":      2,   # latent dynamics / covariance component
        "t4":      3,   # obs noise covariance component
        "t_total": 4,   # combined LRT
    },
    "dpca": {
        "T2": 0,        # Hotelling T² on augmented scores
        "Q":  1,        # SPE (squared prediction error)
    },
    "static_ppca": {
        "W":  0,        # combined mean LRT  (T² + Q/σ)
        "R1": 1,        # latent covariance chart
        "R2": 2,        # residual noise chart
        "R":  3,        # R1 + R2  (unknown-source covariance)
    },
    "var_residual": {
        "T2": 0,        # mean-shift Hotelling T² on VAR residuals
        "W":  1,        # covariance-shift trace statistic
    },
    "lstm_ae": {
        "T2": 0,        # reconstruction-residual Hotelling T²
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# OC comparison pairs  — one entry per (method, statistic) to evaluate
# ─────────────────────────────────────────────────────────────────────────────
#
# Selection rationale
# ───────────────────
# case1 — latent mean shift  (E(z_t) = d·e₁):
#   DyPPCA t1  detects latent-subspace mean deviations directly.
#   DPCA T2, PPCA W, VAR T2 all monitor mean shifts in various representations.
#
# case2 — obs noise mean shift  (E(ε_t) = d·u_{q+1}):
#   DyPPCA t2  is the residual-subspace mean component.
#   DPCA Q, PPCA W  capture residual/SPE changes.
#
# case3 — latent AR matrix shift  (B₁ = B₀ + d·E₁₂):
#   DyPPCA t3  monitors latent dynamics/covariance structure.
#   DPCA Q, PPCA R1, VAR W  are the covariance-sensitive counterparts.
#
# case4 — latent covariance shift  (Cov(z_t) = I + d·e₁e₁ᵀ):
#   DyPPCA t3 (latent covariance term) is the primary sensor.
#
# case5 — local obs noise covariance shift  (σ₀ I + d·σ₀·u u ᵀ):
#   DyPPCA t4  is the obs-noise covariance component.
#   PPCA R2  measures residual noise variance deviation.
#
# t_total is always included to show overall DyPPCA power regardless of type.

OC_COMPARISON_STATS: Dict[str, List[Tuple[str, str]]] = {
    "case1": [
        ("dyppca",       "t1"),
        ("dyppca",       "t_total"),
        ("dpca",         "T2"),
        ("static_ppca",  "W"),
        ("var_residual", "T2"),
        ("lstm_ae",      "T2"),
    ],
    "case2": [
        ("dyppca",       "t2"),
        ("dyppca",       "t_total"),
        ("dpca",         "Q"),
        ("static_ppca",  "W"),
        ("var_residual", "T2"),
        ("lstm_ae",      "T2"),
    ],
    "case3": [
        ("dyppca",       "t3"),
        ("dyppca",       "t_total"),
        ("dpca",         "Q"),
        ("static_ppca",  "R1"),
        ("var_residual", "W"),
        ("lstm_ae",      "T2"),
    ],
    "case4": [
        ("dyppca",       "t3"),
        ("dyppca",       "t_total"),
        ("dpca",         "Q"),
        ("static_ppca",  "R1"),
        ("var_residual", "W"),
        ("lstm_ae",      "T2"),
    ],
    "case5": [
        ("dyppca",       "t4"),
        ("dyppca",       "t_total"),
        ("dpca",         "Q"),
        ("static_ppca",  "R2"),
        ("var_residual", "W"),
        ("lstm_ae",      "T2"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _stat_val(row, si: int) -> float:
    """Extract a scalar from monitor_window output (scalar or tuple/array)."""
    if np.isscalar(row):
        return float(row)
    return float(row[si])


# ─────────────────────────────────────────────────────────────────────────────
# ARL₁ estimation
# ─────────────────────────────────────────────────────────────────────────────

def run_arl_experiment(base_method: str,
                       stat_name: str,
                       monitor,
                       ucls: dict,
                       model_ic: dict,
                       case: str,
                       d: float,
                       n_reps: int,
                       n_window: int,
                       K_max: int = 2000,
                       rng=None):
    """
    Estimate ARL₁ for one (method, statistic) pair under OC case `case` with
    shift magnitude `d`.

    Parameters
    ----------
    base_method : key in STAT_INDEX  (e.g. "dyppca", "dpca")
    stat_name   : key in STAT_INDEX[base_method]  (e.g. "t1", "T2")
    monitor     : fitted monitor object
    ucls        : ucls[base_method]  dict of calibrated thresholds
    model_ic    : IC model dict from build_ic_model()
    case        : "case1" … "case5"
    d           : shift magnitude
    n_reps      : number of OC replications
    n_window    : monitoring window size
    K_max       : censoring run length
    rng         : numpy Generator

    Returns
    -------
    arl_mean : float
    arl_se   : float  (std / √n_reps)
    rls      : (n_reps,) raw run lengths
    """
    from data_generator import simulate_ic_stateful, simulate_oc_stateful

    if rng is None:
        rng = np.random.default_rng()

    si = STAT_INDEX[base_method][stat_name]
    h  = ucls[stat_name]

    WARMUP = 5 * n_window
    q      = model_ic["B0"].shape[0]
    rls    = np.empty(n_reps, dtype=np.float64)

    for b in range(n_reps):
        # Warm-up under IC to reach approximate stationarity
        z = np.zeros(q)
        X_warmup, z = simulate_ic_stateful(model_ic, WARMUP, z, rng)
        x_lag       = X_warmup[-1:]

        delay = K_max
        for k in range(K_max):
            X_new, z = simulate_oc_stateful(model_ic, n_window, case, d, z, rng)
            X_win    = np.vstack([x_lag, X_new])

            row = monitor.monitor_window(X_win)
            if _stat_val(row, si) > h:
                delay = k + 1
                break

            x_lag = X_new[-1:]

        rls[b] = delay

    arl_mean = float(rls.mean())
    arl_se   = float(rls.std() / np.sqrt(n_reps))
    return arl_mean, arl_se, rls


# ─────────────────────────────────────────────────────────────────────────────
# DyPPCA diagnostic component ratios  (unchanged — still uses t_total alarm)
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_ratios(dyppca_monitor,
                      ucls: dict,
                      model_ic: dict,
                      case: str,
                      d: float,
                      n_reps: int,
                      n_window: int,
                      K_max: int = 2000,
                      rng=None) -> dict:
    """
    At the alarm window, compute the mean fraction of t_total contributed by
    each component:  ρⱼ = tⱼ / (t₁+t₂+t₃+t₄).

    Returns {"rho1": …, "rho2": …, "rho3": …, "rho4": …}.
    """
    from data_generator import simulate_ic_stateful, simulate_oc_stateful

    if rng is None:
        rng = np.random.default_rng()

    h      = ucls["t_total"]
    WARMUP = 5 * n_window
    q      = model_ic["B0"].shape[0]
    rhos   = []

    for _ in range(n_reps):
        z = np.zeros(q)
        X_warmup, z = simulate_ic_stateful(model_ic, WARMUP, z, rng)
        x_lag       = X_warmup[-1:]

        for k in range(K_max):
            X_new, z = simulate_oc_stateful(model_ic, n_window, case, d, z, rng)
            X_win    = np.vstack([x_lag, X_new])
            row      = dyppca_monitor.monitor_window(X_win)

            if row[4] > h:           # t_total alarm
                total = row[4]
                if total > 0:
                    rhos.append(np.array(row[:4]) / total)
                break

            x_lag = X_new[-1:]

    if not rhos:
        return {"rho1": np.nan, "rho2": np.nan,
                "rho3": np.nan, "rho4": np.nan}

    rhos = np.array(rhos).mean(axis=0)
    return {
        "rho1": float(rhos[0]),
        "rho2": float(rhos[1]),
        "rho3": float(rhos[2]),
        "rho4": float(rhos[3]),
    }

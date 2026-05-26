"""
evaluation.py
ARL₁ estimation for all OC cases.

Statistic index registry
────────────────────────
  DyPPCA         : t1(0), t2(1), t3(2), t4(3), t_total(4)
  DPCA           : T2(0), Q(1)
  Static PPCA    : T(0)   — single combined statistic
  VAR-residual   : T(0)   — single combined statistic
  LSTM-AE        : T2(0)

Phase II comparison sets  (OC_COMPARISON_STATS)
───────────────────────────────────────────────
For every case we compare:
  • DyPPCA: the theoretically sensitive component + t_total
  • DPCA: T2 and Q  (both, since DPCA has no further decomposition)
  • Static PPCA: T  (combined)
  • VAR-residual: T  (combined)
  • LSTM-AE: T2

Alarm rule:  alarm  ⟺  stat[si] > h     (single threshold, no OR-combining)
"""

import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Statistic index registry
# ─────────────────────────────────────────────────────────────────────────────

STAT_INDEX: Dict[str, Dict[str, int]] = {
    "dyppca": {
        "t1":      0,   # latent mean component
        "t2":      1,   # residual mean component
        "t3":      2,   # latent dynamics / covariance component
        "t4":      3,   # obs noise covariance component
        "t_total": 4,   # combined LRT  (t1+t2+t3+t4)
    },
    "dpca": {
        "T2": 0,        # Hotelling T² on augmented scores
        "Q":  1,        # SPE (squared prediction error)
    },
    "static_ppca": {
        "T": 0,         # combined: T²+Q/σ₀+R1+R2  (scalar return)
    },
    "var_residual": {
        "T": 0,         # combined: T²_resid+W_cov  (scalar return)
    },
    "lstm_ae": {
        "T2": 0,        # reconstruction-residual Hotelling T²
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# OC comparison sets
# ─────────────────────────────────────────────────────────────────────────────
#
# Rationale per case (from Table 1 sensitivity analysis)
# ───────────────────────────────────────────────────────
# Case I   — latent mean shift  E(z_t)=d·e₁:
#     t1 is the sensitive DyPPCA component.
#
# Case II  — obs noise mean shift  E(ε_t)=d·u_{e,1}:
#     t2 is the sensitive DyPPCA component.
#
# Case III — latent AR matrix shift  B₁=B₀+d·E₁₂:
#     t3 captures the lagged covariance change.
#
# Case IV  — latent covariance shift  Cov(z_t)=I+d·e₁e₁ᵀ:
#     t3 captures the change in latent dynamics (B fixed, Σ_z changes).
#
# Case V   — obs noise covariance shift  Cov(ε_t)=σ₀I+d·σ₀·u u ᵀ:
#     t4 is the sensitive DyPPCA component.
#
# In all cases DPCA contributes both T2 and Q (no further decomposition).
# VAR-residual, Static PPCA, and LSTM-AE each contribute their single statistic.

OC_COMPARISON_STATS: Dict[str, List[Tuple[str, str]]] = {
    "case1": [
        ("dyppca",       "t1"),
        ("dyppca",       "t_total"),
        ("dpca",         "T2"),
        ("dpca",         "Q"),
        ("static_ppca",  "T"),
        ("var_residual", "T"),
        ("lstm_ae",      "T2"),
    ],
    "case2": [
        ("dyppca",       "t2"),
        ("dyppca",       "t_total"),
        ("dpca",         "T2"),
        ("dpca",         "Q"),
        ("static_ppca",  "T"),
        ("var_residual", "T"),
        ("lstm_ae",      "T2"),
    ],
    "case3": [
        ("dyppca",       "t3"),
        ("dyppca",       "t_total"),
        ("dpca",         "T2"),
        ("dpca",         "Q"),
        ("static_ppca",  "T"),
        ("var_residual", "T"),
        ("lstm_ae",      "T2"),
    ],
    "case4": [
        ("dyppca",       "t3"),
        ("dyppca",       "t_total"),
        ("dpca",         "T2"),
        ("dpca",         "Q"),
        ("static_ppca",  "T"),
        ("var_residual", "T"),
        ("lstm_ae",      "T2"),
    ],
    "case5": [
        ("dyppca",       "t4"),
        ("dyppca",       "t_total"),
        ("dpca",         "T2"),
        ("dpca",         "Q"),
        ("static_ppca",  "T"),
        ("var_residual", "T"),
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
    Estimate ARL₁ for one (method, statistic) pair under OC case `case`.

    Parameters
    ----------
    base_method : key in STAT_INDEX
    stat_name   : key in STAT_INDEX[base_method]
    monitor     : fitted monitor object
    ucls        : ucls[base_method]  (dict {stat_name: h_star})
    model_ic    : IC model dict from build_ic_model()
    case        : "case1" … "case5"
    d           : shift magnitude
    n_reps      : OC replications
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
# DyPPCA diagnostic component ratios
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
    At the t_total alarm window, compute mean fraction contributed by each
    component:  ρⱼ = tⱼ / (t₁+t₂+t₃+t₄).

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

"""
evaluation.py — CRN-based ARL₁ estimation for Phase II.

Common Random Numbers design for Phase II
──────────────────────────────────────────
For each (case, d), B OC sequences are pre-generated ONCE using
simulate_oc_batch_stateful(), producing a statistics matrix for EVERY method.
All methods' ARL₁ values are then computed from the same OC trajectories,
ensuring fair comparison and reducing variance in method differences.

OC stats matrix structure:
    oc_stats[method][b, k] = statistic for sequence b at window k
All methods see the same B fault realisations.
"""

import numpy as np
from typing import Dict, List, Tuple

# ── Statistic index registry ──────────────────────────────────────────────────
STAT_INDEX: Dict[str, Dict[str, int]] = {
    "dyppca":       {"t1":0,"t2":1,"t3":2,"t4":3,"t_total":4},
    "dpca":         {"T2":0,"Q":1},
    "static_ppca":  {"T":0},
    "var_residual": {"T":0},
    "lstm_ae":      {"T2":0},
}

# ── Phase II comparison sets ──────────────────────────────────────────────────
OC_COMPARISON_STATS: Dict[str, List[Tuple[str, str]]] = {
    "case1": [("dyppca","t1"),("dyppca","t_total"),
              ("dpca","T2"),("dpca","Q"),
              ("static_ppca","T"),("var_residual","T"),("lstm_ae","T2")],
    "case2": [("dyppca","t2"),("dyppca","t_total"),
              ("dpca","T2"),("dpca","Q"),
              ("static_ppca","T"),("var_residual","T"),("lstm_ae","T2")],
    "case3": [("dyppca","t3"),("dyppca","t_total"),
              ("dpca","T2"),("dpca","Q"),
              ("static_ppca","T"),("var_residual","T"),("lstm_ae","T2")],
    "case4": [("dyppca","t3"),("dyppca","t_total"),
              ("dpca","T2"),("dpca","Q"),
              ("static_ppca","T"),("var_residual","T"),("lstm_ae","T2")],
    "case5": [("dyppca","t4"),("dyppca","t_total"),
              ("dpca","T2"),("dpca","Q"),
              ("static_ppca","T"),("var_residual","T"),("lstm_ae","T2")],
}


# ─────────────────────────────────────────────────────────────────────────────
# CRN OC statistics matrix generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_oc_stats(model_ic, monitors_needed, case, d,
                        n_window, K_max_crn, B_crn, rng):
    """
    Pre-generate OC statistics for B_crn sequences × K_max_crn windows,
    for ALL required monitors simultaneously (CRN across methods).

    Parameters
    ----------
    monitors_needed : dict {method_name: monitor_obj}
                      only methods needed for this case
    Returns
    -------
    oc_stats : dict {method_name: (B_crn, K_max_crn, d)}
    """
    from data_generator import (simulate_ic_batch_stateful,
                                 simulate_oc_batch_stateful)

    WARMUP = 5 * n_window
    q      = model_ic["B0"].shape[0]

    # Warmup under IC, then switch to OC
    Z = np.zeros((B_crn, q))
    _, Z      = simulate_ic_batch_stateful(model_ic, WARMUP, Z, rng)
    X_last, Z = simulate_ic_batch_stateful(model_ic, 1, Z, rng)
    x_lag     = X_last   # (B, 1, p)

    # Pre-allocate output buffers
    oc_stats = {}
    for name, mon in monitors_needed.items():
        X_test, _ = simulate_oc_batch_stateful(model_ic, n_window, case, d,
                                                Z[:2].copy(),
                                                np.random.default_rng(99))
        Xw_test = np.concatenate([x_lag[:2], X_test], axis=1)
        d_stats = mon.monitor_window_batch(Xw_test).shape[1]
        oc_stats[name] = np.empty((B_crn, K_max_crn, d_stats))

    # Generate K_max_crn windows for all B sequences simultaneously
    for k in range(K_max_crn):
        X_new, Z = simulate_oc_batch_stateful(model_ic, n_window, case, d, Z, rng)
        X_win    = np.concatenate([x_lag, X_new], axis=1)   # (B, n+1, p)
        for name, mon in monitors_needed.items():
            oc_stats[name][:, k, :] = mon.monitor_window_batch(X_win)
        x_lag = X_new[:, -1:, :]

    return oc_stats


# ─────────────────────────────────────────────────────────────────────────────
# ARL₁ from pre-generated OC stats (CRN)
# ─────────────────────────────────────────────────────────────────────────────

def _arl_from_stats(stats_col, h):
    """
    Compute ARL₁ from (B, K_max) stats column and threshold h.
    Same logic as calibration: find first alarm window per sequence.
    """
    B, K_max   = stats_col.shape
    exceed     = stats_col > h
    has_alarm  = exceed.any(axis=1)
    first_k    = np.argmax(exceed, axis=1)
    run_lengths = np.where(has_alarm, first_k + 1, K_max)
    arl_mean    = float(run_lengths.mean())
    arl_se      = float(run_lengths.std() / np.sqrt(B))
    return arl_mean, arl_se


# ─────────────────────────────────────────────────────────────────────────────
# Main Phase II ARL₁ experiment (CRN)
# ─────────────────────────────────────────────────────────────────────────────

def run_arl_experiment_crn(case, d, monitors, ucls, model_ic,
                            n_window, K_max_crn, B_crn, rng):
    """
    Compute ARL₁ for all relevant (method, statistic) pairs using CRN.

    All methods see the SAME B_crn OC realisations, ensuring:
    - Fair comparison (same fault paths)
    - Reduced variance in ARL₁ differences between methods

    Returns
    -------
    results : dict {f"{method}.{stat}": {"arl": float, "se": float}}
    """
    pairs = [(m, s) for m, s in OC_COMPARISON_STATS[case]
             if m in monitors]

    # Only generate for methods actually needed
    methods_needed = {m: monitors[m]
                      for m in set(m for m, _ in pairs)
                      if m in monitors}

    oc_stats = _generate_oc_stats(
        model_ic, methods_needed, case, d,
        n_window, K_max_crn, B_crn, rng)

    results = {}
    for method, stat_name in pairs:
        if method not in oc_stats:
            continue
        si  = STAT_INDEX[method][stat_name]
        h   = ucls[method][stat_name]
        col = oc_stats[method][:, :, si]          # (B, K_max)
        arl, se = _arl_from_stats(col, h)
        key = f"{method}.{stat_name}"
        results[key] = {"arl": round(arl, 2), "se": round(se, 3)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DyPPCA diagnostic component ratios (CRN)
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_ratios_crn(dyppca_monitor, ucls, model_ic,
                            case, d, n_window, K_max_crn, B_crn, rng):
    """
    At alarm windows, compute mean fraction of t_total from each component.
    Uses CRN (same OC sequences as ARL computation if called with same rng).
    """
    from data_generator import (simulate_ic_batch_stateful,
                                 simulate_oc_batch_stateful)

    WARMUP = 5 * n_window
    q      = model_ic["B0"].shape[0]
    h      = ucls["dyppca"]["t_total"]

    Z = np.zeros((B_crn, q))
    _, Z      = simulate_ic_batch_stateful(model_ic, WARMUP, Z, rng)
    X_last, Z = simulate_ic_batch_stateful(model_ic, 1, Z, rng)
    x_lag     = X_last

    rho_sum  = np.zeros(4)
    n_alarms = 0

    for k in range(K_max_crn):
        X_new, Z = simulate_oc_batch_stateful(model_ic, n_window, case, d, Z, rng)
        X_win    = np.concatenate([x_lag, X_new], axis=1)
        batch    = dyppca_monitor.monitor_window_batch(X_win)   # (B, 5)
        x_lag    = X_new[:, -1:, :]

        alarms   = batch[:, 4] > h     # t_total > h
        if alarms.any():
            t_alarm = batch[alarms, :4]      # (n_alarm, 4)
            t_tot   = batch[alarms, 4:5]     # (n_alarm, 1)
            rhos    = (t_alarm / t_tot).mean(0)
            rho_sum += rhos * alarms.sum()
            n_alarms += int(alarms.sum())

    if n_alarms == 0:
        return {"rho1":np.nan,"rho2":np.nan,"rho3":np.nan,"rho4":np.nan}

    rho_mean = rho_sum / n_alarms
    return {"rho1":float(rho_mean[0]),"rho2":float(rho_mean[1]),
            "rho3":float(rho_mean[2]),"rho4":float(rho_mean[3])}

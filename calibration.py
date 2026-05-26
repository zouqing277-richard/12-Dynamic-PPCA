"""
calibration.py  —  CRN-based Monte Carlo calibration for UCLs.

Common Random Numbers (CRN) design
────────────────────────────────────
For each method, B_crn IC trajectories are pre-generated ONCE, producing a
statistics matrix  stats_mat[b, k, d]  (B × K_max_crn × n_stats).

For any threshold h, ARL₀(h) is computed from the SAME pre-generated matrix:
    RL_b(h) = first window k where stats_mat[b,k,si] > h  (or K_max_crn)
    ARL₀(h) = mean(RL_b(h))

Since stats_mat is fixed, ARL₀(h) is a non-decreasing step function of h,
so bisection is always valid and gives consistent results across runs.

Statistic registry
──────────────────
  DyPPCA         : t1(0), t2(1), t3(2), t4(3), t_total(4)
  DPCA           : T2(0), Q(1)
  Static PPCA    : T(0)
  VAR-residual   : T(0)
  LSTM-AE        : T2(0)
"""

import os
import numpy as np
from joblib import Parallel, delayed

CALIBRATION_STATS = {
    "dyppca":       {"t1":0, "t2":1, "t3":2, "t4":3, "t_total":4},
    "dpca":         {"T2":0, "Q":1},
    "static_ppca":  {"T":0},
    "var_residual": {"T":0},
    "lstm_ae":      {"T2":0},
}


# ─────────────────────────────────────────────────────────────────────────────
# CRN statistics matrix generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_ic_stats_matrix(model_ic, monitor, n_window, K_max_crn, B_crn, rng,
                               verbose=False):
    """
    Pre-generate statistics for B_crn IC sequences × K_max_crn windows.

    All B_crn sequences are simulated simultaneously (batch).
    Returns stats_mat : (B_crn, K_max_crn, d)  where d = n_stats of this monitor.

    This matrix is shared across ALL UCL bisections for this monitor,
    ensuring CRN: ARL₀(h) is monotone non-decreasing in h for any fixed matrix.
    """
    from data_generator import simulate_ic_batch_stateful

    WARMUP = 5 * n_window
    q      = model_ic["B0"].shape[0]

    # Warmup: drive B_crn sequences to stationarity
    Z = np.zeros((B_crn, q))
    _, Z      = simulate_ic_batch_stateful(model_ic, WARMUP, Z, rng)
    X_last, Z = simulate_ic_batch_stateful(model_ic, 1, Z, rng)
    x_lag     = X_last                             # (B, 1, p)

    # Determine output dimension from one test call
    X_test, _ = simulate_ic_batch_stateful(model_ic, n_window,
                                            Z[:2].copy(),
                                            np.random.default_rng(0))
    Xw_test = np.concatenate([x_lag[:2], X_test], axis=1)
    d_stats = monitor.monitor_window_batch(Xw_test).shape[1]

    stats_mat = np.empty((B_crn, K_max_crn, d_stats))

    for k in range(K_max_crn):
        X_new, Z = simulate_ic_batch_stateful(model_ic, n_window, Z, rng)
        X_win    = np.concatenate([x_lag, X_new], axis=1)   # (B, n+1, p)
        stats_mat[:, k, :] = monitor.monitor_window_batch(X_win)
        x_lag = X_new[:, -1:, :]

        if verbose and (k+1) % 200 == 0:
            print(f"      {k+1}/{K_max_crn} windows ...", flush=True)

    return stats_mat   # (B_crn, K_max_crn, d_stats)


# ─────────────────────────────────────────────────────────────────────────────
# ARL computation from pre-generated stats (CRN guaranteed monotone)
# ─────────────────────────────────────────────────────────────────────────────

def _arl_from_stats(stats_col, h):
    """
    Compute ARL₀ from a pre-generated statistics column.

    Parameters
    ----------
    stats_col : (B, K_max) array — one statistic for B sequences × K_max windows
    h         : threshold

    Returns
    -------
    float  — estimated ARL₀(h), monotone non-decreasing in h for fixed stats_col
    """
    B, K_max = stats_col.shape
    exceed   = stats_col > h              # (B, K_max) bool

    has_alarm  = exceed.any(axis=1)       # (B,)
    first_k    = np.argmax(exceed, axis=1)  # 0 if never (handled below)
    run_lengths = np.where(has_alarm, first_k + 1, K_max)

    censor_rate = float((~has_alarm).mean())
    if censor_rate > 0.05:
        import warnings
        warnings.warn(
            f"CRN ARL: censoring rate = {censor_rate:.3f} > 5%. "
            f"Consider increasing K_max_crn.", RuntimeWarning, stacklevel=2)

    return float(run_lengths.mean())


# ─────────────────────────────────────────────────────────────────────────────
# CRN bisection for one UCL
# ─────────────────────────────────────────────────────────────────────────────

def _bisect_crn(stats_col, target_arl, tol, h_lo, h_hi,
                n_coarse, max_fine, name=""):
    """
    Two-phase bisection using pre-generated stats_col (CRN).

    Since ARL₀(h) is monotone in h for fixed stats_col, bisection is exact
    (no Monte Carlo noise between evaluations at different h values).

    Coarse phase: n_coarse binary steps (fast, same stats_col).
    Fine phase  : continue until |ARL₀(h) - target| ≤ tol.
    """
    def arl_at(h):
        return _arl_from_stats(stats_col, h)

    # Coarse phase
    for _ in range(n_coarse):
        h_mid = (h_lo + h_hi) / 2
        arl   = arl_at(h_mid)
        if arl > target_arl: h_hi = h_mid
        else:                 h_lo = h_mid

    # Fine phase
    h_mid     = (h_lo + h_hi) / 2
    final_arl = None
    for step in range(max_fine):
        arl       = arl_at(h_mid)
        final_arl = arl
        err       = abs(arl - target_arl)
        print(f"  [{name}] step {step+1:>2}: "
              f"h={h_mid:.4f}  ARL={arl:.2f}  |err|={err:.2f}"
              f"  {'✓' if err<=tol else ''}",
              flush=True)
        if err <= tol:
            break
        if arl > target_arl: h_hi = h_mid
        else:                 h_lo = h_mid
        h_mid = (h_lo + h_hi) / 2
    else:
        print(f"  WARNING [{name}]: max_fine={max_fine} reached, "
              f"last ARL={final_arl:.2f}", flush=True)

    return h_mid, final_arl


# ─────────────────────────────────────────────────────────────────────────────
# Main entry — CRN calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_all(model_ic, methods, n_window, arl0, K_max_crn, rng,
                  B_coarse=None,     # unused, kept for API compat
                  n_coarse=20,
                  B_fine=None,       # unused
                  bisect_tol=2.0,
                  B_bootstrap=None,  # unused
                  max_fine=60,
                  fast=False,
                  verbose=True,
                  n_jobs=-1,
                  B_crn=5000):
    """
    CRN-based UCL calibration.

    For each method:
      1. Pre-generate stats matrix (B_crn × K_max_crn × d) from IC trajectories.
         All UCLs of the same method share this matrix (CRN across UCLs too).
      2. Bisect each UCL using the monotone ARL₀(h) from the fixed matrix.

    Parameters
    ----------
    B_crn      : number of IC sequences per method (default 5000).
                 CRN reduces Monte Carlo variance so fewer sequences suffice.
    K_max_crn  : max windows per sequence (should be >> ARL₀; default ≈1500).
    n_coarse   : binary search steps in coarse phase.
    bisect_tol : fine phase stops when |ARL₀(h)−arl0| ≤ tol.
    n_jobs     : joblib parallel jobs (methods run in parallel).

    Returns
    -------
    ucls : dict  {method: {stat_name: h_star}}
    """
    if fast:
        return _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_crn)

    if verbose:
        print(f"CRN calibration: B_crn={B_crn}  K_max_crn={K_max_crn}  "
              f"n_coarse={n_coarse}  tol={bisect_tol}", flush=True)

    def _calibrate_one_method(name, monitor):
        """Calibrate all UCLs for one method."""
        if name not in CALIBRATION_STATS:
            return name, {}

        stats_map = CALIBRATION_STATS[name]
        if verbose:
            print(f"\n[{name}] generating IC stats matrix "
                  f"({B_crn}×{K_max_crn}) ...", flush=True)

        # Fresh RNG per method (seeded from parent)
        method_rng = np.random.default_rng(rng.integers(0, 2**31))
        stats_mat  = _generate_ic_stats_matrix(
            model_ic, monitor, n_window, K_max_crn, B_crn, method_rng)
        # stats_mat: (B_crn, K_max_crn, d_stats)

        # Estimate bootstrap bounds from stats_mat marginals
        ucls_method = {}
        for stat_name, si in stats_map.items():
            # For each sequence, take the max statistic value
            # (conservative bound for quantile estimation)
            flat = stats_mat[:, :, si].ravel()
            h_lo = float(np.quantile(flat, 0.990))
            h_hi = float(np.quantile(flat, 0.9999))
            if verbose:
                print(f"  [{name}] {stat_name:8s}: "
                      f"bootstrap [{h_lo:.4f}, {h_hi:.4f}]", flush=True)

            h_star, arl_f = _bisect_crn(
                stats_mat[:, :, si],
                target_arl = arl0,
                tol        = bisect_tol,
                h_lo       = h_lo,
                h_hi       = h_hi,
                n_coarse   = n_coarse,
                max_fine   = max_fine,
                name       = f"{name}.{stat_name}",
            )
            ucls_method[stat_name] = h_star
            if verbose:
                arl_str = f"{arl_f:.1f}" if arl_f else "n/a"
                print(f"  [{name}] {stat_name:8s}: "
                      f"h*={h_star:.4f}  ARL={arl_str}", flush=True)

        return name, ucls_method

    # Parallel across methods
    if verbose:
        ncores = os.cpu_count() if n_jobs == -1 else n_jobs
        print(f"\nRunning {len(methods)} methods "
              f"(n_jobs={n_jobs}, up to {min(len(methods),ncores)} parallel)\n",
              flush=True)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_calibrate_one_method)(name, mon)
        for name, mon in methods.items()
    )

    return {name: ucl_dict for name, ucl_dict in results}


# ─────────────────────────────────────────────────────────────────────────────
# Fast mode (debug)
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_fast=500):
    """Fast calibration: empirical quantile only (UCLs not accurate)."""
    from data_generator import simulate_ic_batch_stateful

    WARMUP, q = 5*n_window, model_ic["B0"].shape[0]
    alpha = 1.0 / arl0
    ucls  = {}

    for name, monitor in methods.items():
        if name not in CALIBRATION_STATS:
            continue
        Z = np.zeros((B_fast, q))
        _, Z      = simulate_ic_batch_stateful(model_ic, WARMUP, Z, rng)
        X_last, Z = simulate_ic_batch_stateful(model_ic, 1, Z, rng)
        x_lag     = X_last
        X_new, _  = simulate_ic_batch_stateful(model_ic, n_window, Z, rng)
        X_win     = np.concatenate([x_lag, X_new], axis=1)
        batch     = monitor.monitor_window_batch(X_win)   # (B, d)

        ucls[name] = {}
        for stat_name, si in CALIBRATION_STATS[name].items():
            ucls[name][stat_name] = float(np.quantile(batch[:, si], 1-alpha))
        print(f"  [fast] {name}: UCL (quantile only, NOT accurate)", flush=True)

    return ucls


# ─────────────────────────────────────────────────────────────────────────────
# Post-calibration ARL₀ verification (CRN)
# ─────────────────────────────────────────────────────────────────────────────

def verify_arl0_crn(model_ic, monitors, ucls, n_window, K_max_crn,
                    B_verify, rng, verbose=True):
    """
    Verify calibrated UCLs using a fresh CRN stats matrix.
    Reports ARL₀ estimate and censoring rate for each method's primary stat.
    """
    VERIFY_STAT = {
        "dyppca":      "t_total",
        "dpca":        "T2",
        "static_ppca": "T",
        "var_residual":"T",
        "lstm_ae":     "T2",
    }
    if verbose:
        print(f"\nARL₀ verification  (B_verify={B_verify})")

    results = {}
    for name, monitor in monitors.items():
        stat_name = VERIFY_STAT.get(name, next(iter(CALIBRATION_STATS[name])))
        si        = CALIBRATION_STATS[name][stat_name]
        h         = ucls[name][stat_name]

        method_rng = np.random.default_rng(rng.integers(0, 2**31))
        stats_mat  = _generate_ic_stats_matrix(
            model_ic, monitor, n_window, K_max_crn, B_verify, method_rng)
        arl0_hat   = _arl_from_stats(stats_mat[:, :, si], h)
        err        = abs(arl0_hat - 200) / 200 * 100
        ok         = "✓" if err < 10 else "~"

        if verbose:
            print(f"  {name:<16} [{stat_name}]  "
                  f"ARL₀={arl0_hat:6.1f}  err={err:.1f}%  {ok}")
        results[name] = arl0_hat

    return results

"""
calibration.py  —  Two-phase Monte Carlo bisection for UCL calibration.

Every (method, statistic) pair gets its own independently calibrated threshold
h* such that |ARL0(h*) - target| <= BISECT_TOL.

Statistic registry
──────────────────
  DyPPCA         : t1, t2, t3, t4, t_total   (5 separate UCLs)
  DPCA           : T2, Q                       (2 separate UCLs)
  Static PPCA    : T   (combined T²+Q/σ+R1+R2) (1 UCL)
  VAR-residual   : T   (combined T²+W)          (1 UCL)
  LSTM-AE        : T2                           (1 UCL)

Algorithm
─────────
1. Bootstrap  (B_BOOTSTRAP independent IC windows):
   estimate initial bounds [h_lo, h_hi] from the [0.98, 0.9999] quantile.

2. Coarse phase  (N_COARSE bisection steps, B_COARSE IC sequences each):
   evaluate ARL0(h_mid) with censoring K_MAX.  Update [h_lo, h_hi].

3. Fine phase  (runs until |ARL0(h_mid) - target| <= bisect_tol):
   continue with B_FINE sequences per step; safety cap MAX_FINE steps.

Each statistic is bisected independently:
    alarm  ⟺  stat[si] > h
"""

import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Master statistic registry
# ─────────────────────────────────────────────────────────────────────────────

# Maps  method_name → {stat_name: column_index_in_monitor_window_output}
#
# DyPPCA   : monitor_window returns (t1, t2, t3, t4, t_total)  → indices 0–4
# DPCA     : monitor_window returns (T2, Q)                     → indices 0–1
# StaticPPCA: monitor_window returns a scalar float             → index  0
# VARResidual: monitor_window returns a scalar float            → index  0
# LSTMAEMonitor: monitor_window returns a scalar float          → index  0

CALIBRATION_STATS = {
    "dyppca": {
        "t1":      0,
        "t2":      1,
        "t3":      2,
        "t4":      3,
        "t_total": 4,
    },
    "dpca": {
        "T2": 0,
        "Q":  1,
    },
    "static_ppca": {
        "T": 0,   # combined: T² + Q/σ₀ + R1 + R2
    },
    "var_residual": {
        "T": 0,   # combined: T²_resid + W_cov
    },
    "lstm_ae": {
        "T2": 0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stat_val(row, si: int) -> float:
    """
    Extract statistic value from monitor_window output.

    Handles both scalar returns (static_ppca, var_residual, lstm_ae) and
    tuple/array returns (dyppca, dpca).
    """
    if np.isscalar(row):
        return float(row)
    return float(row[si])


def _bootstrap_stats(model_ic, monitor, n_window, B, rng):
    """B independent IC windows → (B, d) statistics array."""
    from data_generator import simulate_ic
    X_p = simulate_ic(model_ic, n_window + 1, rng=rng)
    out = monitor.monitor_window(X_p)
    d   = 1 if np.isscalar(out) else len(out)
    stats = np.empty((B, d))
    for i in range(B):
        X = simulate_ic(model_ic, n_window + 1, rng=rng)
        r = monitor.monitor_window(X)
        stats[i] = [r] if np.isscalar(r) else list(r)
    return stats


def _estimate_arl(model_ic, monitor, alarm_fn, n_window, K_max, B, rng):
    """
    Estimate ARL0 via B IC replications with a continuous latent trajectory.

    Uses simulate_ic_stateful() so consecutive windows share the same z_t
    path (no fake restarts from z=0).
    """
    from data_generator import simulate_ic_stateful

    WARMUP = 5 * n_window

    delays = []
    for _ in range(B):
        q   = model_ic["B0"].shape[0]
        z   = np.zeros(q)
        X_warmup, z = simulate_ic_stateful(model_ic, WARMUP, z, rng)
        x_lag = X_warmup[-1:]

        delay = K_max
        for k in range(K_max):
            X_new, z = simulate_ic_stateful(model_ic, n_window, z, rng)
            X_win    = np.vstack([x_lag, X_new])
            row      = monitor.monitor_window(X_win)
            if alarm_fn(row):
                delay = k + 1
                break
            x_lag = X_new[-1:]

        delays.append(delay)
    return float(np.mean(delays))


def _bisect_scalar(model_ic, monitor, stat_idx, n_window, K_max,
                   h_lo, h_hi, target_arl, tol,
                   B_coarse, n_coarse, B_fine, rng, name="",
                   max_fine=60):
    """
    Two-phase bisection for a single scalar threshold.

    Coarse phase : n_coarse fixed steps with B_coarse sequences.
    Fine phase   : runs until |ARL0(h) - target| <= tol  (or max_fine cap).
    """
    def alarm_at(h):
        return lambda row: _stat_val(row, stat_idx) > h

    def arl_at(h, B):
        return _estimate_arl(model_ic, monitor, alarm_at(h),
                             n_window, K_max, B, rng)

    # ── Coarse phase ─────────────────────────────────────────────────────
    for _ in range(n_coarse):
        h_mid = (h_lo + h_hi) / 2
        arl   = arl_at(h_mid, B_coarse)
        if arl > target_arl:
            h_hi = h_mid
        else:
            h_lo = h_mid

    # ── Fine phase ────────────────────────────────────────────────────────
    h_mid     = (h_lo + h_hi) / 2
    final_arl = None

    for step in range(max_fine):
        arl       = arl_at(h_mid, B_fine)
        final_arl = arl
        err       = abs(arl - target_arl)

        print(f"    [{name}] Fine step {step+1:>2}: h={h_mid:.4f}  "
              f"ARL0={arl:.2f}  |err|={err:.2f}  "
              f"{'✓' if err <= tol else ''}",
              flush=True)

        if err <= tol:
            break

        if arl > target_arl:
            h_hi = h_mid
        else:
            h_lo = h_mid
        h_mid = (h_lo + h_hi) / 2
    else:
        print(f"    WARNING [{name}]: fine phase did not converge in {max_fine} steps. "
              f"Last ARL0={final_arl:.2f}  (target={target_arl}±{tol}). "
              f"Returning best h so far.", flush=True)

    return h_mid, final_arl


# ─────────────────────────────────────────────────────────────────────────────
# Fast mode (debug / smoke-test only)
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_bootstrap):
    """
    Fast calibration for debug/smoke-test only.
    Uses bootstrap quantile — UCLs will NOT be accurate.
    """
    alpha = 1.0 / arl0
    ucls  = {}
    for name, monitor in methods.items():
        boot = _bootstrap_stats(model_ic, monitor, n_window, B_bootstrap, rng)
        ucls[name] = {}
        for stat_name, si in CALIBRATION_STATS[name].items():
            ucls[name][stat_name] = float(np.quantile(boot[:, si], 1 - alpha))
        print(f"  [fast] {name}: UCL (quantile only, NOT accurate)", flush=True)
    return ucls


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_all(model_ic, methods, n_window, arl0, K_max, rng,
                  B_coarse, n_coarse, B_fine,
                  bisect_tol, B_bootstrap,
                  max_fine=60, fast=False, verbose=True):
    """
    Calibrate UCLs for all methods via two-phase ARL bisection.

    Every (method, statistic) pair is bisected independently so that each
    statistic satisfies |ARL0(h*) - arl0| <= bisect_tol on its own.

    Returns
    -------
    ucls : dict  {method_name: {stat_name: h_star, ...}, ...}
    """
    if fast:
        print("[fast mode] Using bootstrap quantile (code check only, UCLs inaccurate)",
              flush=True)
        return _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_bootstrap)

    ucls     = {}
    iterator = tqdm(methods.items(), desc="Calibrating") if verbose else methods.items()

    for name, monitor in iterator:
        if name not in CALIBRATION_STATS:
            print(f"  WARNING: no CALIBRATION_STATS entry for '{name}', skipping.",
                  flush=True)
            continue

        if verbose:
            print(f"\n{'─'*60}", flush=True)
            print(f"[{name}] bootstrapping {B_bootstrap} windows ...", flush=True)

        boot   = _bootstrap_stats(model_ic, monitor, n_window, B_bootstrap, rng)
        stats  = CALIBRATION_STATS[name]
        ucls[name] = {}

        for stat_name, si in stats.items():
            h_lo = float(np.quantile(boot[:, si], 0.980))
            h_hi = float(np.quantile(boot[:, si], 0.9999))

            if verbose:
                print(f"  [{name}] bisecting {stat_name:8s}  "
                      f"[{h_lo:.4f}, {h_hi:.4f}]", flush=True)

            h_s, arl_f = _bisect_scalar(
                model_ic, monitor, si, n_window, K_max,
                h_lo, h_hi, arl0, bisect_tol,
                B_coarse, n_coarse, B_fine, rng,
                name=f"{name}.{stat_name}",
                max_fine=max_fine,
            )
            ucls[name][stat_name] = h_s

            if verbose:
                arl_str = f"{arl_f:.1f}" if arl_f is not None else "n/a"
                print(f"  [{name}] {stat_name:8s}: h*={h_s:.4f}  "
                      f"ARL0={arl_str}", flush=True)

    return ucls

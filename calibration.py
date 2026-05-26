"""
calibration.py  —  Two-phase Monte Carlo bisection for UCL calibration.
Vectorised + parallel edition.

Key changes vs. the sequential version
────────────────────────────────────────
1. _estimate_arl_vectorized():
   Simulates B IC sequences in parallel using simulate_ic_batch_stateful()
   and evaluates statistics with monitor.monitor_window_batch().
   Replaces the B-sequence Python for-loop with NumPy (B,q)/(B,p) ops.

2. calibrate_all() uses joblib.Parallel to run independent UCL bisections
   on separate CPU cores simultaneously.

Statistic registry
──────────────────
  DyPPCA         : t1, t2, t3, t4, t_total   (5 UCLs, indices 0-4)
  DPCA           : T2, Q                       (2 UCLs, indices 0-1)
  Static PPCA    : T   (combined)              (1 UCL,  index  0)
  VAR-residual   : T   (combined)              (1 UCL,  index  0)
  LSTM-AE        : T2                          (1 UCL,  index  0)
"""

import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Statistic registry
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_STATS = {
    "dyppca":       {"t1":0, "t2":1, "t3":2, "t4":3, "t_total":4},
    "dpca":         {"T2":0, "Q":1},
    "static_ppca":  {"T":0},
    "var_residual": {"T":0},
    "lstm_ae":      {"T2":0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stat_val(row, si: int) -> float:
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


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised ARL estimation  (replaces the sequential B-loop)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_arl_vectorized(model_ic, monitor, stat_idx, h,
                              n_window, K_max, B, rng):
    """
    Estimate ARL0 by simulating B IC sequences in parallel.

    Uses simulate_ic_batch_stateful() so all B latent states advance
    together via (B,q) matrix ops, and monitor_window_batch() to
    evaluate statistics for all active windows simultaneously.

    Returns
    -------
    float  estimated ARL0
    """
    from data_generator import simulate_ic_batch_stateful

    WARMUP = 5 * n_window
    q      = model_ic["B0"].shape[0]
    p      = model_ic["A0"].shape[0]

    # Initialise B latent states, run warmup
    Z = np.zeros((B, q))
    _, Z        = simulate_ic_batch_stateful(model_ic, WARMUP, Z, rng)
    X_last, Z   = simulate_ic_batch_stateful(model_ic, 1,      Z, rng)
    x_lag       = X_last                        # (B, 1, p)

    delays = np.full(B, K_max, dtype=np.float64)
    active = np.ones(B, dtype=bool)             # which sequences are still running

    for k in range(K_max):
        n_active = int(active.sum())
        if n_active == 0:
            break

        # Simulate one window for all active sequences
        Z_act  = Z[active]
        xl_act = x_lag[active]                  # (n_active, 1, p)
        X_new, Z_new = simulate_ic_batch_stateful(model_ic, n_window, Z_act, rng)
        X_win  = np.concatenate([xl_act, X_new], axis=1)  # (n_active, n+1, p)

        # Batch statistics
        stats_batch = monitor.monitor_window_batch(X_win)  # (n_active, d)
        alarms = stats_batch[:, stat_idx] > h               # (n_active,)

        # Record run lengths
        act_idx = np.where(active)[0]
        delays[act_idx[alarms]] = k + 1
        active[act_idx[alarms]] = False

        # Carry forward non-alarmed states
        still = ~alarms
        Z[active]     = Z_new[still]
        x_lag[active] = X_new[still, -1:, :]

    return float(delays.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Two-phase bisection for one UCL
# ─────────────────────────────────────────────────────────────────────────────

def _bisect_one_ucl(model_ic, monitor, stat_idx,
                    h_lo, h_hi, target_arl, tol,
                    n_window, K_max,
                    B_coarse, n_coarse, B_fine,
                    seed, name="", max_fine=60):
    """
    Two-phase bisection for a single (method, statistic) UCL.
    Called in parallel by calibrate_all().
    """
    rng = np.random.default_rng(seed)

    def arl_at(h, B):
        return _estimate_arl_vectorized(model_ic, monitor, stat_idx, h,
                                        n_window, K_max, B, rng)

    # Coarse phase
    for _ in range(n_coarse):
        h_mid = (h_lo + h_hi) / 2
        arl   = arl_at(h_mid, B_coarse)
        if arl > target_arl:
            h_hi = h_mid
        else:
            h_lo = h_mid

    # Fine phase
    h_mid     = (h_lo + h_hi) / 2
    final_arl = None
    for step in range(max_fine):
        arl       = arl_at(h_mid, B_fine)
        final_arl = arl
        err       = abs(arl - target_arl)
        print(f"  [{name}] Fine step {step+1:>2}: "
              f"h={h_mid:.4f}  ARL0={arl:.2f}  |err|={err:.2f}"
              f"  {'✓' if err<=tol else ''}",
              flush=True)
        if err <= tol:
            break
        h_hi = h_mid if arl > target_arl else h_hi
        h_lo = h_mid if arl <= target_arl else h_lo
        h_mid = (h_lo + h_hi) / 2
    else:
        print(f"  WARNING [{name}]: max_fine reached, "
              f"last ARL0={final_arl:.2f}", flush=True)

    return h_mid, final_arl


# ─────────────────────────────────────────────────────────────────────────────
# Fast mode
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_bootstrap):
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
# Main entry — vectorised + parallel
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_all(model_ic, methods, n_window, arl0, K_max, rng,
                  B_coarse, n_coarse, B_fine,
                  bisect_tol, B_bootstrap,
                  max_fine=60, fast=False, verbose=True,
                  n_jobs=-1):
    """
    Calibrate UCLs for all methods via two-phase ARL bisection.

    Vectorised simulation (simulate_ic_batch_stateful + monitor_window_batch)
    replaces the inner B-sequence Python loop.

    joblib.Parallel runs independent UCL bisections concurrently.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs for joblib.  -1 = all available cores.
        Set to 1 to disable parallelism (useful for debugging).

    Returns
    -------
    ucls : dict  {method_name: {stat_name: h_star, ...}, ...}
    """
    if fast:
        print("[fast mode] bootstrap quantile only (UCLs inaccurate)", flush=True)
        return _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_bootstrap)

    # ── Bootstrap initial intervals ──────────────────────────────────────────
    if verbose:
        print("Bootstrapping initial intervals ...", flush=True)

    boot_cache = {}
    for name, monitor in methods.items():
        if name not in CALIBRATION_STATS:
            continue
        boot_cache[name] = _bootstrap_stats(
            model_ic, monitor, n_window, B_bootstrap, rng)

    # ── Build task list ───────────────────────────────────────────────────────
    tasks = []
    for name, monitor in methods.items():
        if name not in CALIBRATION_STATS:
            continue
        boot = boot_cache[name]
        for stat_name, si in CALIBRATION_STATS[name].items():
            h_lo = float(np.quantile(boot[:, si], 0.980))
            h_hi = float(np.quantile(boot[:, si], 0.9999))
            seed  = int(rng.integers(0, 2**31))
            tasks.append(dict(
                name=name, monitor=monitor, stat_name=stat_name,
                si=si, h_lo=h_lo, h_hi=h_hi, seed=seed,
            ))
            if verbose:
                print(f"  [{name}] {stat_name:8s}: "
                      f"bootstrap interval [{h_lo:.4f}, {h_hi:.4f}]", flush=True)

    if verbose:
        n_cores = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        print(f"\nRunning {len(tasks)} UCL bisections "
              f"(joblib n_jobs={n_jobs}, "
              f"up to {min(len(tasks), n_cores)} parallel) ...\n", flush=True)

    # ── Parallel bisection ────────────────────────────────────────────────────
    def _run(task):
        return _bisect_one_ucl(
            model_ic   = model_ic,
            monitor    = task["monitor"],
            stat_idx   = task["si"],
            h_lo       = task["h_lo"],
            h_hi       = task["h_hi"],
            target_arl = arl0,
            tol        = bisect_tol,
            n_window   = n_window,
            K_max      = K_max,
            B_coarse   = B_coarse,
            n_coarse   = n_coarse,
            B_fine     = B_fine,
            seed       = task["seed"],
            name       = f"{task['name']}.{task['stat_name']}",
            max_fine   = max_fine,
        )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run)(t) for t in tasks
    )

    # ── Collect results ───────────────────────────────────────────────────────
    ucls = {}
    for task, (h_star, arl_final) in zip(tasks, results):
        name      = task["name"]
        stat_name = task["stat_name"]
        if name not in ucls:
            ucls[name] = {}
        ucls[name][stat_name] = h_star
        if verbose:
            arl_str = f"{arl_final:.1f}" if arl_final is not None else "n/a"
            print(f"  [{name}] {stat_name:8s}: h*={h_star:.4f}  ARL0={arl_str}",
                  flush=True)

    return ucls

"""
calibration.py  —  Two-phase Monte Carlo bisection for UCL calibration.

For each method find threshold h* such that |ARL0(h*) - 200| <= BISECT_TOL.

Algorithm
─────────
1. Bootstrap  (B_bootstrap independent IC windows):
   estimate initial bounds [h_lo, h_hi] from the [0.98, 0.9999] quantile.

2. Coarse phase  (N_COARSE bisection steps, B_COARSE IC sequences each):
   evaluate ARL0(h_mid) = mean run-length with censoring K_MAX.
   Update [h_lo, h_hi].

3. Fine phase  (N_FINE steps, B_FINE IC sequences each):
   continue until |ARL0(h_mid) - 200| <= bisect_tol.

Single-stat methods  (DyPPCA t_total, PPCA W/R, VAR T2/W):
    bisect a scalar threshold h.

OR-rule method  (DPCA T2+Q):
    bisect a shared quantile level p: UCL_j = quantile(boot[:, j], p).
"""

import numpy as np
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

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
    Estimate ARL0 via B IC replications.

    Correctness fix (vs earlier version):
    The process x_t is driven by a latent VAR(1) state z_t. Consecutive
    monitoring windows MUST share the same continuous z_t trajectory.
    The previous implementation called simulate_ic() fresh for each window,
    restarting z from 0 every time — producing fake continuity.

    This version uses simulate_ic_stateful(), which accepts and returns
    the current latent state z so each window genuinely follows the previous.

    Timeline for one replication:
      z=0 ──[warm-up: WARMUP steps]──> z_warm
           ──[window 0: n steps]──> x_lag | x_1..x_n,  z_0
           ──[window 1: n steps]──> x_lag | x_1..x_n,  z_1
           ...  (early stop when alarm fires)
    """
    from data_generator import simulate_ic_stateful

    WARMUP = 5 * n_window   # enough steps for z to reach stationarity

    delays = []
    for _ in range(B):
        # ── Warm-up: start from z=0, reach approximate stationarity ─────────
        q   = model_ic["B0"].shape[0]
        z   = np.zeros(q)
        X_warmup, z = simulate_ic_stateful(model_ic, WARMUP, z, rng)

        # x_lag: the last warm-up observation serves as the lag for window 0
        x_lag = X_warmup[-1:]                   # shape (1, p)

        delay = K_max
        for k in range(K_max):
            # Generate n_window NEW observations from the current state z
            X_new, z = simulate_ic_stateful(model_ic, n_window, z, rng)

            # Window = [lag | new observations]  shape (n_window+1, p)
            X_win = np.vstack([x_lag, X_new])

            row = monitor.monitor_window(X_win)
            if alarm_fn(row):
                delay = k + 1
                break

            # The last new observation becomes the lag for the next window
            x_lag = X_new[-1:]                  # shape (1, p)

        delays.append(delay)
    return float(np.mean(delays))


def _bisect_scalar(model_ic, monitor, stat_idx, n_window, K_max,
                   h_lo, h_hi, target_arl, tol,
                   B_coarse, n_coarse, B_fine, rng, name="",
                   max_fine=60):
    """
    Bisect scalar threshold for one statistic column.

    Coarse phase : n_coarse fixed steps with B_coarse sequences each.
                   Purpose: quickly locate [h_lo, h_hi] near the true UCL.

    Fine phase   : runs UNTIL convergence  (no fixed step limit).
                   Each step uses B_fine sequences (large → low noise).
                   Stops when  |ARL₀(h_mid) − target_arl| ≤ tol.
                   Safety cap of max_fine steps prevents infinite loops.
    """

    def arl_at(h, B):
        return _estimate_arl(model_ic, monitor,
                             lambda row: float(row[stat_idx]) > h,
                             n_window, K_max, B, rng)

    # ── Coarse phase ─────────────────────────────────────────────────────
    for step in range(n_coarse):
        h_mid = (h_lo + h_hi) / 2
        arl   = arl_at(h_mid, B_coarse)
        if arl > target_arl:
            h_hi = h_mid
        else:
            h_lo = h_mid

    # ── Fine phase: run until |ARL₀ − 200| ≤ tol ─────────────────────────
    h_mid     = (h_lo + h_hi) / 2
    final_arl = None

    for step in range(max_fine):
        arl       = arl_at(h_mid, B_fine)
        final_arl = arl
        err       = abs(arl - target_arl)

        print(f"    Fine step {step+1:>2}: h={h_mid:.4f}  "
              f"ARL₀={arl:.2f}  |err|={err:.2f}  "
              f"{'✓ converged' if err <= tol else ''}",
              flush=True)

        if err <= tol:
            break                       # ← converged

        # Not yet converged: continue bisection
        if arl > target_arl:
            h_hi = h_mid               # h too high, lower it
        else:
            h_lo = h_mid               # h too low, raise it
        h_mid = (h_lo + h_hi) / 2

    else:
        # Safety cap reached without convergence
        print(f"    WARNING: fine phase did not converge in {max_fine} steps. "
              f"Last ARL₀={final_arl:.2f}  (target={target_arl}±{tol}). "
              f"Returning best h so far.", flush=True)

    return h_mid, final_arl


def _bisect_or(model_ic, monitor, boot_stats, n_window, K_max,
               target_arl, tol,
               B_coarse, n_coarse, B_fine, rng, max_fine=60):
    """Bisect shared quantile level p for OR-rule monitor (DPCA)."""
    lo = max(0.0, 1.0 - 20.0 / target_arl)
    hi = 1.0 - 0.5 / (target_arl * boot_stats.shape[1])

    def alarm_at_p(p):
        q = np.quantile(boot_stats, p, axis=0)
        return lambda row: bool(np.any(np.asarray(row) > q))

    # Coarse
    for _ in range(n_coarse):
        mid = (lo + hi) / 2
        arl = _estimate_arl(model_ic, monitor, alarm_at_p(mid),
                            n_window, K_max, B_coarse, rng)
        if arl > target_arl:
            hi = mid
        else:
            lo = mid

    # Fine phase: run until convergence
    mid       = (lo + hi) / 2
    final_arl = None

    for step in range(max_fine):
        arl       = _estimate_arl(model_ic, monitor, alarm_at_p(mid),
                                  n_window, K_max, B_fine, rng)
        final_arl = arl
        err       = abs(arl - target_arl)

        print(f"    Fine step {step+1:>2}: p={mid:.6f}  "
              f"ARL₀={arl:.2f}  |err|={err:.2f}  "
              f"{'✓ converged' if err <= tol else ''}",
              flush=True)

        if err <= tol:
            break

        if arl > target_arl:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) / 2
    else:
        print(f"    WARNING: OR fine phase did not converge in {max_fine} steps.",
              flush=True)

    return np.quantile(boot_stats, mid, axis=0), final_arl



def _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_bootstrap):
    """
    Fast calibration for debug/smoke-test only.
    Uses bootstrap quantile (no ARL simulation needed).
    UCLs will NOT be accurate — do not use for real experiments.
    """
    alpha = 1.0 / arl0
    ucls  = {}
    for name, monitor in methods.items():
        boot = _bootstrap_stats(model_ic, monitor, n_window, B_bootstrap, rng)
        if name == "dyppca":
            ucls[name] = {
                "t_total": float(np.quantile(boot[:, 4], 1 - alpha)),
                "t1": float(np.quantile(boot[:, 0], 1 - alpha)),
                "t2": float(np.quantile(boot[:, 1], 1 - alpha)),
                "t3": float(np.quantile(boot[:, 2], 1 - alpha)),
                "t4": float(np.quantile(boot[:, 3], 1 - alpha)),
            }
        elif name == "dpca":
            q = np.quantile(boot, 1 - alpha / 2, axis=0)
            ucls[name] = {"T2": float(q[0]), "Q": float(q[1])}
        elif name == "static_ppca":
            ucls[name] = {k: float(np.quantile(boot[:, i], 1 - alpha))
                          for i, k in enumerate(["W","R1","R2","R"])}
        elif name == "var_residual":
            ucls[name] = {"T2": float(np.quantile(boot[:, 0], 1 - alpha)),
                           "W":  float(np.quantile(boot[:, 1], 1 - alpha))}
        elif name == "lstm_ae":
            ucls[name] = {"T2": float(np.quantile(boot[:, 0], 1 - alpha))}
        else:
            ucls[name] = float(np.quantile(boot[:, 0], 1 - alpha))
        print(f"  [fast] {name}: UCL (quantile only, NOT accurate)", flush=True)
    return ucls


# ── main entry ────────────────────────────────────────────────────────────────

def calibrate_all(model_ic, methods, n_window, arl0, K_max, rng,
                  B_coarse, n_coarse, B_fine,
                  bisect_tol, B_bootstrap,
                  max_fine=60, fast=False, verbose=True):
    """
    Calibrate UCLs for all methods via two-phase ARL bisection.
    See module docstring for algorithm details.
    """
    # Fast mode: skip ARL simulation, use bootstrap quantile only
    if fast:
        print("[fast mode] Using bootstrap quantile (code check only, UCLs inaccurate)",
              flush=True)
        return _calibrate_fast(model_ic, methods, n_window, arl0, rng, B_bootstrap)

    ucls = {}
    alpha = 1.0 / arl0
    iterator = tqdm(methods.items(), desc="Calibrating") if verbose else methods.items()

    for name, monitor in iterator:
        if verbose:
            print(f"\n[{name}] bootstrapping {B_bootstrap} windows ...", flush=True)

        boot = _bootstrap_stats(model_ic, monitor, n_window, B_bootstrap, rng)

        # ── DyPPCA ────────────────────────────────────────────────────────
        if name == "dyppca":
            si = 4   # t_total column
            h_lo = float(np.quantile(boot[:, si], 0.980))
            h_hi = float(np.quantile(boot[:, si], 0.9999))
            if verbose:
                print(f"[{name}] bisecting t_total  [{h_lo:.2f}, {h_hi:.2f}]", flush=True)
            h_star, arl_f = _bisect_scalar(model_ic, monitor, si, n_window, K_max,
                                            h_lo, h_hi, arl0, bisect_tol,
                                            B_coarse, n_coarse, B_fine, rng)
            if verbose:
                print(f"[{name}] h*={h_star:.3f}  ARL0={arl_f:.1f}", flush=True)
            ucls[name] = {
                "t_total": h_star,
                "t1": float(np.quantile(boot[:, 0], 1 - alpha)),
                "t2": float(np.quantile(boot[:, 1], 1 - alpha)),
                "t3": float(np.quantile(boot[:, 2], 1 - alpha)),
                "t4": float(np.quantile(boot[:, 3], 1 - alpha)),
            }

        # ── DPCA (OR rule) ─────────────────────────────────────────────────
        elif name == "dpca":
            if verbose:
                print(f"[{name}] bisecting joint T2+Q ...", flush=True)
            ucl_pair, arl_f = _bisect_or(model_ic, monitor, boot,
                                          n_window, K_max, arl0, bisect_tol,
                                          B_coarse, n_coarse, B_fine, n_fine, rng)
            if verbose:
                print(f"[{name}] T2={ucl_pair[0]:.3f} Q={ucl_pair[1]:.3f}  ARL0={arl_f:.1f}", flush=True)
            ucls[name] = {"T2": float(ucl_pair[0]), "Q": float(ucl_pair[1])}

        # ── Static PPCA ────────────────────────────────────────────────────
        elif name == "static_ppca":
            result = {}
            for sn, si in [("W", 0), ("R1", 1), ("R2", 2), ("R", 3)]:
                h_lo = float(np.quantile(boot[:, si], 0.980))
                h_hi = float(np.quantile(boot[:, si], 0.9999))
                h_s, af = _bisect_scalar(model_ic, monitor, si, n_window, K_max,
                                         h_lo, h_hi, arl0, bisect_tol,
                                         B_coarse, n_coarse, B_fine, rng)
                if verbose:
                    print(f"[{name}] {sn}={h_s:.3f}  ARL0={af:.1f}", flush=True)
                result[sn] = h_s
            ucls[name] = result

        # ── VAR residual ───────────────────────────────────────────────────
        elif name == "var_residual":
            result = {}
            for sn, si in [("T2", 0), ("W", 1)]:
                h_lo = float(np.quantile(boot[:, si], 0.980))
                h_hi = float(np.quantile(boot[:, si], 0.9999))
                h_s, af = _bisect_scalar(model_ic, monitor, si, n_window, K_max,
                                         h_lo, h_hi, arl0, bisect_tol,
                                         B_coarse, n_coarse, B_fine, rng)
                if verbose:
                    print(f"[{name}] {sn}={h_s:.3f}  ARL0={af:.1f}", flush=True)
                result[sn] = h_s
            ucls[name] = result

        # ── LSTM-AE ────────────────────────────────────────────────────────
        elif name == "lstm_ae":
            si   = 0
            h_lo = float(np.quantile(boot[:, si], 0.980))
            h_hi = float(np.quantile(boot[:, si], 0.9999))
            h_s, af = _bisect_scalar(model_ic, monitor, si, n_window, K_max,
                                     h_lo, h_hi, arl0, bisect_tol,
                                     B_coarse, n_coarse, B_fine, rng)
            if verbose:
                print(f"[{name}] h*={h_s:.3f}  ARL0={af:.1f}", flush=True)
            ucls[name] = {"T2": h_s}

        else:
            # Generic fallback
            h_lo = float(np.quantile(boot[:, 0], 0.980))
            h_hi = float(np.quantile(boot[:, 0], 0.9999))
            h_s, _ = _bisect_scalar(model_ic, monitor, 0, n_window, K_max,
                                    h_lo, h_hi, arl0, bisect_tol,
                                    B_coarse, n_coarse, B_fine, rng)
            ucls[name] = h_s

    return ucls

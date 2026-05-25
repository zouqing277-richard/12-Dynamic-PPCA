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
    alarm_fn: (stats_row) -> bool
    """
    from data_generator import simulate_ic
    T = K_max * n_window + 1
    delays = []
    for _ in range(B):
        X     = simulate_ic(model_ic, T, rng=rng)
        stats = monitor.monitor_sequence(X, n=n_window)
        delay = K_max
        for k, row in enumerate(stats[:K_max]):
            if alarm_fn(row):
                delay = k + 1
                break
        delays.append(delay)
    return float(np.mean(delays))


def _bisect_scalar(model_ic, monitor, stat_idx, n_window, K_max,
                   h_lo, h_hi, target_arl, tol,
                   B_coarse, n_coarse, B_fine, n_fine, rng, name=""):
    """Bisect scalar threshold for one statistic column."""

    def arl_at(h, B):
        return _estimate_arl(model_ic, monitor,
                             lambda row: float(row[stat_idx]) > h,
                             n_window, K_max, B, rng)

    # Coarse phase
    for step in range(n_coarse):
        h_mid = (h_lo + h_hi) / 2
        arl   = arl_at(h_mid, B_coarse)
        if arl > target_arl:
            h_hi = h_mid
        else:
            h_lo = h_mid

    # Fine phase
    h_mid = (h_lo + h_hi) / 2
    final_arl = target_arl
    for step in range(n_fine):
        arl       = arl_at(h_mid, B_fine)
        final_arl = arl
        if abs(arl - target_arl) <= tol:
            break
        if arl > target_arl:
            h_hi = h_mid
        else:
            h_lo = h_mid
        h_mid = (h_lo + h_hi) / 2

    return h_mid, final_arl


def _bisect_or(model_ic, monitor, boot_stats, n_window, K_max,
               target_arl, tol,
               B_coarse, n_coarse, B_fine, n_fine, rng):
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

    # Fine
    mid = (lo + hi) / 2
    final_arl = target_arl
    for _ in range(n_fine):
        arl       = _estimate_arl(model_ic, monitor, alarm_at_p(mid),
                                  n_window, K_max, B_fine, rng)
        final_arl = arl
        if abs(arl - target_arl) <= tol:
            break
        if arl > target_arl:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) / 2

    return np.quantile(boot_stats, mid, axis=0), final_arl


# ── main entry ────────────────────────────────────────────────────────────────

def calibrate_all(model_ic, methods, n_window, arl0, K_max, rng,
                  B_coarse, n_coarse, B_fine, n_fine,
                  bisect_tol, B_bootstrap, verbose=True):
    """
    Calibrate UCLs for all methods via two-phase ARL bisection.
    See module docstring for algorithm details.
    """
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
                                            B_coarse, n_coarse, B_fine, n_fine, rng)
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
                                         B_coarse, n_coarse, B_fine, n_fine, rng)
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
                                         B_coarse, n_coarse, B_fine, n_fine, rng)
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
                                     B_coarse, n_coarse, B_fine, n_fine, rng)
            if verbose:
                print(f"[{name}] h*={h_s:.3f}  ARL0={af:.1f}", flush=True)
            ucls[name] = {"T2": h_s}

        else:
            # Generic fallback
            h_lo = float(np.quantile(boot[:, 0], 0.980))
            h_hi = float(np.quantile(boot[:, 0], 0.9999))
            h_s, _ = _bisect_scalar(model_ic, monitor, 0, n_window, K_max,
                                    h_lo, h_hi, arl0, bisect_tol,
                                    B_coarse, n_coarse, B_fine, n_fine, rng)
            ucls[name] = h_s

    return ucls

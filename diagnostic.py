"""
diagnostic.py
─────────────
Systematic Phase II diagnostic following the four-step protocol:

Step 1  Generate M IC windows → empirical 99.5% quantile h_emp for each statistic.
Step 2  On a fresh batch, verify P(stat > h_emp) ≈ 0.005.
Step 3  Run independent IC sequences → verify ARL ≈ 200 with h_emp.
Step 4  Check censoring rate & compare h_emp vs bisection h*.

Key questions answered:
  (A) Is the tail distribution stable across seeds?
  (B) Does P(alarm | IC) ≈ 1/ARL hold, or does autocorrelation break it?
  (C) Is the run-length simulation logic correct?
  (D) Does bisection give the same threshold as empirical quantile?

Usage:
    python diagnostic.py               # full run (M=500000)
    python diagnostic.py --M 50000     # faster check
    python diagnostic.py --stat t1     # focus on one statistic
"""

import argparse, os, sys, time
import numpy as np

sys.path.insert(0, ".")
import config
from data_generator import (build_ic_model, simulate_ic,
                             simulate_ic_stateful, simulate_ic_batch_stateful)
from methods.dyppca import DyPPCA
from methods.dpca    import DPCA
from methods.static_ppca  import StaticPPCA
from methods.var_residual import VARResidual


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TARGET_ARL  = 200
ALPHA       = 1.0 / TARGET_ARL        # 0.005
N_WINDOW    = config.N_WINDOW          # 50
K_MAX       = 2000                   # generous upper bound for ARL simulation
B_ARL       = 10000                   # sequences for ARL verification
SEED        = config.SEED

# Statistics to diagnose: (method_key, stat_index, label)
ALL_STATS = [
    ("dyppca",       0, "t1"),
    ("dyppca",       1, "t2"),
    ("dyppca",       2, "t3"),
    ("dyppca",       3, "t4"),
    ("dyppca",       4, "t_total"),
    ("dpca",         0, "dpca_T2"),
    ("dpca",         1, "dpca_Q"),
    ("static_ppca",  0, "ppca_T"),
    ("var_residual", 0, "var_T"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Empirical quantile from M IC windows
# ─────────────────────────────────────────────────────────────────────────────

def step1_empirical_quantile(monitors, ic, M, n, rng, batch_size=5000):
    """
    Generate M independent IC windows and compute all statistics.

    Windows are generated from stationary starts (warmup of 5n steps each)
    to reflect true IC stationary distribution, not cold-start z=0.

    Returns
    -------
    stats_all : dict {method: np.ndarray (M, d)}
    h_emp     : dict {(method, stat_idx): float}  99.5% quantile
    """
    print(f"\n{'─'*60}")
    print(f"Step 1  Empirical quantile  (M={M:,}, n={n})")
    print(f"{'─'*60}")

    n_batches = (M + batch_size - 1) // batch_size
    WARMUP    = 5 * n

    # Storage: accumulate batches
    store = {name: [] for name in monitors}

    for b_idx in range(n_batches):
        b = min(batch_size, M - b_idx * batch_size)

        # Warmup: B independent latent states from stationary distribution
        Z = np.zeros((b, ic["q"]))
        _, Z    = simulate_ic_batch_stateful(ic, WARMUP, Z, rng)
        X_lag, Z = simulate_ic_batch_stateful(ic, 1, Z, rng)

        # One window
        X_new, _ = simulate_ic_batch_stateful(ic, n, Z, rng)
        X_win    = np.concatenate([X_lag, X_new], axis=1)  # (b, n+1, p)

        for name, mon in monitors.items():
            out = mon.monitor_window_batch(X_win)  # (b, d)
            store[name].append(out)

        if (b_idx+1) % 10 == 0 or b_idx == n_batches-1:
            done = min((b_idx+1)*batch_size, M)
            print(f"  {done:>7,}/{M:,} windows generated ...", flush=True)

    stats_all = {name: np.vstack(arrs) for name, arrs in store.items()}

    # Compute 99.5% quantiles
    h_emp = {}
    print(f"\n  99.5% empirical quantiles:")
    for method, si, label in ALL_STATS:
        if method not in stats_all:
            continue
        vals = stats_all[method][:, si]
        h    = float(np.quantile(vals, 1 - ALPHA))
        h_emp[(method, si)] = h
        print(f"    {label:<14}: h_emp = {h:.4f}  "
              f"(mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"max={vals.max():.3f})")

    return stats_all, h_emp


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Verify P(stat > h_emp) ≈ 0.005 on fresh windows
# ─────────────────────────────────────────────────────────────────────────────

def step2_verify_exceedance(monitors, ic, h_emp, M_verify, n, rng):
    """
    On M_verify fresh independent IC windows, compute empirical P(stat > h).
    Should be ≈ ALPHA = 0.005.  Instability here → tail distribution is noisy.
    """
    print(f"\n{'─'*60}")
    print(f"Step 2  Verify exceedance probability  (M={M_verify:,})")
    print(f"{'─'*60}")
    print(f"  Target P(alarm | IC) = {ALPHA:.4f} = 1/{TARGET_ARL}")

    WARMUP = 5 * n
    Z = np.zeros((M_verify, ic["q"]))
    _, Z     = simulate_ic_batch_stateful(ic, WARMUP, Z, rng)
    X_lag, Z = simulate_ic_batch_stateful(ic, 1, Z, rng)
    X_new, _ = simulate_ic_batch_stateful(ic, n, Z, rng)
    X_win    = np.concatenate([X_lag, X_new], axis=1)

    results = {}
    for method, si, label in ALL_STATS:
        if method not in monitors:
            continue
        mon  = monitors[method]
        h    = h_emp.get((method, si))
        if h is None:
            continue
        vals = mon.monitor_window_batch(X_win)[:, si]
        p_hat = float((vals > h).mean())
        se    = float(np.sqrt(p_hat*(1-p_hat)/M_verify))
        ok    = "✓" if abs(p_hat - ALPHA) < 3*se + 0.001 else "✗"
        print(f"  {label:<14}: P(alarm)={p_hat:.5f}  SE={se:.5f}  "
              f"target={ALPHA:.4f}  {ok}")
        results[label] = dict(p_hat=p_hat, se=se)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verify ARL with continuous IC sequences
# ─────────────────────────────────────────────────────────────────────────────

def step3_verify_arl(monitors, ic, h_emp, B_arl, n, K_max, rng):
    """
    Run B_arl continuous IC sequences (z_t carried across windows).
    Check:
      - ARL estimate
      - Censoring rate  (RL == K_max)
      - SE(ARL) = std/sqrt(B)
    """
    print(f"\n{'─'*60}")
    print(f"Step 3  ARL verification with continuous sequences  (B={B_arl:,})")
    print(f"{'─'*60}")
    print(f"  K_max={K_max}  (sequences that never alarm are censored at K_max)")

    WARMUP = 5 * n
    results = {}

    for method, si, label in ALL_STATS:
        if method not in monitors:
            continue
        mon = monitors[method]
        h   = h_emp.get((method, si))
        if h is None:
            continue

        rng_local = np.random.default_rng(rng.integers(0, 2**31))
        delays = np.empty(B_arl)

        for b in range(B_arl):
            z = np.zeros(ic["q"])
            Xw, z = simulate_ic_stateful(ic, WARMUP, z, rng_local)
            x_lag = Xw[-1:]

            delay = K_max
            for k in range(K_max):
                X_new, z = simulate_ic_stateful(ic, n, z, rng_local)
                X_win    = np.vstack([x_lag, X_new])
                row      = mon.monitor_window(X_win)
                val      = float(row) if np.isscalar(row) else float(row[si])
                if val > h:
                    delay = k + 1
                    break
                x_lag = X_new[-1:]
            delays[b] = delay

        censor_rate = float((delays == K_max).mean())
        arl_hat     = float(delays.mean())
        arl_se      = float(delays.std() / np.sqrt(B_arl))
        ok = "✓" if abs(arl_hat - TARGET_ARL) < 3*arl_se + 5 else "✗"

        print(f"  {label:<14}: ARL={arl_hat:6.1f}  SE={arl_se:.2f}  "
              f"censor={censor_rate:.4f}  {ok}")
        results[label] = dict(arl=arl_hat, se=arl_se, censor=censor_rate)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compare h_emp vs bisection (if checkpoint exists)
# ─────────────────────────────────────────────────────────────────────────────

def step4_compare_thresholds(h_emp):
    print(f"\n{'─'*60}")
    print(f"Step 4  Compare empirical quantile vs bisection UCL")
    print(f"{'─'*60}")

    ckpt_path = "results/calibration/checkpoint.pkl"
    if not os.path.exists(ckpt_path):
        print("  No checkpoint found — run run_calibration.py first")
        return

    import pickle
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    bisect_ucls = ck["ucls"]

    # Map (method, si) → bisection h
    bisect_map = {
        ("dyppca",       0): bisect_ucls.get("dyppca",{}).get("t1"),
        ("dyppca",       1): bisect_ucls.get("dyppca",{}).get("t2"),
        ("dyppca",       2): bisect_ucls.get("dyppca",{}).get("t3"),
        ("dyppca",       3): bisect_ucls.get("dyppca",{}).get("t4"),
        ("dyppca",       4): bisect_ucls.get("dyppca",{}).get("t_total"),
        ("dpca",         0): bisect_ucls.get("dpca",{}).get("T2"),
        ("dpca",         1): bisect_ucls.get("dpca",{}).get("Q"),
        ("static_ppca",  0): bisect_ucls.get("static_ppca",{}).get("T"),
        ("var_residual", 0): bisect_ucls.get("var_residual",{}).get("T"),
    }

    print(f"  {'Stat':<14}  {'h_emp':>10}  {'h_bisect':>10}  {'ratio':>8}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*8}")
    for method, si, label in ALL_STATS:
        h_e = h_emp.get((method, si))
        h_b = bisect_map.get((method, si))
        if h_e is None:
            continue
        if h_b is not None:
            ratio = h_e / h_b
            print(f"  {label:<14}  {h_e:>10.4f}  {h_b:>10.4f}  {ratio:>8.4f}")
        else:
            print(f"  {label:<14}  {h_e:>10.4f}  {'n/a':>10}  {'—':>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: check t1 distribution under stationary vs cold start
# ─────────────────────────────────────────────────────────────────────────────

def check_stationary_vs_coldstart(dyppca_mon, ic, n, rng, M=50_000):
    """
    The ARL simulation uses stateful z_t (stationary).
    If we calibrate with cold-start z=0 windows, there may be a mismatch.
    Check how much the t1 distribution shifts.
    """
    print(f"\n{'─'*60}")
    print(f"Bonus  t1 distribution: stationary start vs cold start z=0")
    print(f"{'─'*60}")

    WARMUP = 5 * n

    # Stationary start
    Z = np.zeros((M, ic["q"]))
    _, Z      = simulate_ic_batch_stateful(ic, WARMUP, Z, rng)
    X_lag, Z  = simulate_ic_batch_stateful(ic, 1, Z, rng)
    X_new, _  = simulate_ic_batch_stateful(ic, n, Z, rng)
    X_win     = np.concatenate([X_lag, X_new], axis=1)
    t1_stat   = dyppca_mon.monitor_window_batch(X_win)[:, 0]

    # Cold start z=0
    Z0       = np.zeros((M, ic["q"]))
    X_lag0, Z0 = simulate_ic_batch_stateful(ic, 1, Z0, rng)
    X_new0, _  = simulate_ic_batch_stateful(ic, n, Z0, rng)
    X_win0     = np.concatenate([X_lag0, X_new0], axis=1)
    t1_cold    = dyppca_mon.monitor_window_batch(X_win0)[:, 0]

    for name, vals in [("stationary", t1_stat), ("cold start", t1_cold)]:
        q995 = np.quantile(vals, 0.995)
        print(f"  {name:<12}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"99.5%={q995:.4f}  max={vals.max():.2f}")

    diff = abs(np.quantile(t1_stat, 0.995) - np.quantile(t1_cold, 0.995))
    print(f"  Δ(99.5% quantile) = {diff:.4f}  "
          f"({'negligible' if diff < 0.5 else 'SIGNIFICANT — use stationary start'})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase II diagnostic")
    parser.add_argument("--M",      type=int, default=500_000,
                        help="IC windows for empirical quantile (default 500000)")
    parser.add_argument("--M2",     type=int, default=100_000,
                        help="Fresh windows for exceedance check (default 100000)")
    parser.add_argument("--B_arl",  type=int, default=B_ARL,
                        help=f"Sequences for ARL check (default {B_ARL})")
    parser.add_argument("--K_max",  type=int, default=K_MAX,
                        help=f"Max run length (default {K_MAX})")
    parser.add_argument("--no_arl", action="store_true",
                        help="Skip Step 3 ARL verification (slow)")
    parser.add_argument("--seed",   type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("DyPPCA Phase II Diagnostic")
    print(f"  M={args.M:,}  n={N_WINDOW}  target ARL={TARGET_ARL}")
    print(f"  Oracle parameters: DyPPCA + StaticPPCA from_true_model")
    print("=" * 60)

    # Build model
    ic = build_ic_model(config.P, config.Q, config.SIGMA0,
                        config.A0, config.B0, config.PSI0)

    # Fit monitors (DyPPCA + StaticPPCA use true model)
    X_train = simulate_ic(ic, config.N_TRAIN + 1, rng=rng)
    monitors = {
        "dyppca":       DyPPCA.from_true_model(ic),
        "dpca":         DPCA(config.DPCA_CPV, config.DPCA_LAG).fit(X_train),
        "static_ppca":  StaticPPCA.from_true_model(ic),
        "var_residual": VARResidual().fit(X_train),
    }

    t_total = time.perf_counter()

    # Step 1
    stats_all, h_emp = step1_empirical_quantile(
        monitors, ic, M=args.M, n=N_WINDOW, rng=rng)

    # Step 2
    step2_verify_exceedance(
        monitors, ic, h_emp, M_verify=args.M2, n=N_WINDOW, rng=rng)

    # Step 3 (optional, slow)
    if not args.no_arl:
        step3_verify_arl(
            monitors, ic, h_emp,
            B_arl=args.B_arl, n=N_WINDOW, K_max=args.K_max, rng=rng)
    else:
        print("\nStep 3 skipped (--no_arl)")

    # Step 4
    step4_compare_thresholds(h_emp)

    # Bonus
    check_stationary_vs_coldstart(monitors["dyppca"], ic, N_WINDOW, rng)

    print(f"\n{'='*60}")
    print(f"Total diagnostic time: {(time.perf_counter()-t_total)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
run_calibration.py
──────────────────
Step 1 of 2.  Run this ONCE to calibrate UCLs.

Fitting strategy
────────────────
  DyPPCA, StaticPPCA  : always use from_true_model() — the true IC parameters
                         (A₀, B₀, Ψ₀, σ₀) are fixed and known from the paper.
  DPCA                : Phase I estimates (no closed-form true-param version).
  VARResidual         : Phase I estimates by default; --oracle-var for true model.
  LSTM-AE             : Phase I training (always data-driven).

UCL calibration
───────────────
  Every (method, statistic) pair gets its own MC bisection:
    DyPPCA      → t1, t2, t3, t4, t_total   (5 UCLs)
    DPCA        → T2, Q                       (2 UCLs)
    StaticPPCA  → T  (combined)               (1 UCL)
    VARResidual → T  (combined)               (1 UCL)
    LSTM-AE     → T2                          (1 UCL)

CLI:
  python run_calibration.py               # standard run
  python run_calibration.py --no-lstm     # skip LSTM-AE
  python run_calibration.py --fast        # debug mode
  python run_calibration.py --oracle-var  # VAR also uses from_true_model
"""

import os, sys, time, argparse, pickle
import numpy as np

import config
from data_generator import build_ic_model, simulate_ic
from methods        import DyPPCA, DPCA, StaticPPCA, VARResidual
from calibration    import calibrate_all

CHECKPOINT_DIR  = "results/calibration"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")
SUMMARY_FILE    = os.path.join(CHECKPOINT_DIR, "ucls_summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Fit methods
# ─────────────────────────────────────────────────────────────────────────────

def fit_all_methods(X_train, ic_model, include_lstm=True, oracle_var=False):
    """
    Fit all monitoring methods.

    DyPPCA and StaticPPCA always use from_true_model().
    DPCA and LSTM-AE always use Phase I data.
    VARResidual uses Phase I OLS by default; set oracle_var=True for true model.
    """
    monitors = {
        "dyppca":       DyPPCA.from_true_model(ic_model),
        "static_ppca":  StaticPPCA.from_true_model(ic_model),
        "dpca":         DPCA(cpv_threshold=config.DPCA_CPV,
                             lag=config.DPCA_LAG).fit(X_train),
        "var_residual": (VARResidual.from_true_model(ic_model) if oracle_var
                         else VARResidual().fit(X_train)),
    }
    if include_lstm:
        from methods.lstm_ae import LSTMAEMonitor
        monitors["lstm_ae"] = LSTMAEMonitor(
            input_dim   = X_train.shape[1],
            hidden_dim  = config.LSTM_HIDDEN_DIM,
            latent_dim  = config.LSTM_LATENT_DIM,
            num_layers  = config.LSTM_NUM_LAYERS,
            epochs      = config.LSTM_EPOCHS,
            batch_size  = config.LSTM_BATCH_SIZE,
            lr          = config.LSTM_LR,
            patience    = config.LSTM_PATIENCE,
        ).fit(X_train)
    return monitors


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(ic_model, monitors, ucls, seed, oracle_var=False,
                    path=CHECKPOINT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lstm_state = None
    if "lstm_ae" in monitors:
        import torch
        lstm_state = {
            "state_dict": {k: v.cpu().numpy()
                           for k, v in monitors["lstm_ae"].model.state_dict().items()},
            "input_dim":  monitors["lstm_ae"].input_dim,
            "hidden_dim": monitors["lstm_ae"].hidden_dim,
            "latent_dim": monitors["lstm_ae"].latent_dim,
            "num_layers": monitors["lstm_ae"].num_layers,
            "mu_r":       monitors["lstm_ae"].mu_r,
            "Sig_r_inv":  monitors["lstm_ae"].Sig_r_inv,
        }
    checkpoint = {
        "ic_model":   ic_model,
        "oracle_var": oracle_var,
        "monitors":   {k: v for k, v in monitors.items() if k != "lstm_ae"},
        "lstm_state": lstm_state,
        "ucls":       ucls,
        "seed":       seed,
        "config": {
            "P": config.P, "Q": config.Q, "SIGMA0": config.SIGMA0,
            "N_TRAIN": config.N_TRAIN, "N_WINDOW": config.N_WINDOW,
            "ARL0": config.ARL0, "K_MAX": config.K_MAX,
            "BISECT_TOL": config.BISECT_TOL,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved → {path}")


def write_summary(ucls, arl_verified, path=SUMMARY_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "UCL Calibration Summary",
        f"ARL0 target = {config.ARL0}  |  bisect_tol = {config.BISECT_TOL}",
        "=" * 50,
    ]
    for method, ucl_dict in ucls.items():
        lines.append(f"\n[{method}]")
        for k, v in ucl_dict.items():
            lines.append(f"  {k:12s} = {v:.4f}")
        if method in arl_verified:
            lines.append(f"  Verified ARL0 ≈ {arl_verified[method]:.1f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick ARL0 verification after calibration
# ─────────────────────────────────────────────────────────────────────────────

_VERIFY_STAT = {
    "dyppca":       "t_total",
    "dpca":         "T2",
    "static_ppca":  "T",
    "var_residual": "T",
    "lstm_ae":      "T2",
}


def verify_arl0(ic_model, monitors, ucls, n_window, K_max, rng, B_verify=1000):
    from data_generator import simulate_ic_stateful
    from evaluation import STAT_INDEX

    print(f"\nVerifying ARL0 (B_verify={B_verify}) ...")
    results = {}
    WARMUP  = 5 * n_window

    for name, monitor in monitors.items():
        stat_name = _VERIFY_STAT.get(name, next(iter(STAT_INDEX[name])))
        si        = STAT_INDEX[name][stat_name]
        h         = ucls[name][stat_name]
        delays    = []

        for _ in range(B_verify):
            q = ic_model["B0"].shape[0]
            z = np.zeros(q)
            X_warmup, z = simulate_ic_stateful(ic_model, WARMUP, z, rng)
            x_lag       = X_warmup[-1:]

            delay = K_max
            for k in range(K_max):
                X_new, z  = simulate_ic_stateful(ic_model, n_window, z, rng)
                X_win     = np.vstack([x_lag, X_new])
                row       = monitor.monitor_window(X_win)
                val       = float(row) if np.isscalar(row) else float(row[si])
                if val > h:
                    delay = k + 1
                    break
                x_lag = X_new[-1:]
            delays.append(delay)

        arl0_hat = float(np.mean(delays))
        err      = abs(arl0_hat - config.ARL0) / config.ARL0 * 100
        ok       = "✓" if err < 15 else "~"
        print(f"  {name:<16}  [{stat_name}]  ARL0 = {arl0_hat:6.1f}"
              f"  (err={err:.1f}%)  {ok}")
        results[name] = arl0_hat
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(include_lstm=True, fast=False, seed=config.SEED,
                    oracle_var=False, verify=True):

    B_coarse  = 20    if fast else config.B_COARSE
    n_coarse  = 4     if fast else config.N_COARSE
    B_fine    = 200   if fast else config.B_FINE
    B_boot    = 200   if fast else config.B_BOOTSTRAP
    K_max_run = 50    if fast else config.K_MAX

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("DyPPCA UCL Calibration")
    print(f"  p={config.P}  q={config.Q}  sigma0={config.SIGMA0}")
    print(f"  N_TRAIN={config.N_TRAIN}  n={config.N_WINDOW}")
    print(f"  ARL0={config.ARL0}  K_max={K_max_run}  tol={config.BISECT_TOL}")
    print(f"  Coarse: {n_coarse}x{B_coarse}  "
          f"Fine: until |ARL0-200|<={config.BISECT_TOL} "
          f"(B={B_fine}/step, max {config.MAX_FINE} steps)")
    print(f"  DyPPCA+StaticPPCA: from_true_model  |  "
          f"DPCA+VAR+LSTM: Phase I")
    print("=" * 60)

    # ── Step 1: IC model ───────────────────────────────────────────────────
    ic = build_ic_model(
        p      = config.P,
        q      = config.Q,
        sigma0 = config.SIGMA0,
        A0     = config.A0,
        B0     = config.B0,
        Psi0   = config.PSI0,
    )

    # ── Step 2: Phase I data ───────────────────────────────────────────────
    print(f"\nGenerating Phase I data (N={config.N_TRAIN}) ...", flush=True)
    X_train = simulate_ic(ic, config.N_TRAIN + 1, rng=rng)

    # ── Step 3: Fit all methods ────────────────────────────────────────────
    print("Fitting monitors ...", flush=True)
    t0       = time.time()
    monitors = fit_all_methods(X_train, ic,
                               include_lstm=include_lstm,
                               oracle_var=oracle_var)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # ── Step 4: Calibrate UCLs ─────────────────────────────────────────────
    print("\nCalibrating UCLs ...", flush=True)
    t0   = time.time()
    ucls = calibrate_all(
        model_ic    = ic,
        methods     = monitors,
        n_window    = config.N_WINDOW,
        arl0        = config.ARL0,
        K_max       = K_max_run,
        rng         = rng,
        fast        = fast,
        B_coarse    = B_coarse,
        n_coarse    = n_coarse,
        B_fine      = B_fine,
        bisect_tol  = config.BISECT_TOL,
        max_fine    = config.MAX_FINE,
        B_bootstrap = B_boot,
        verbose     = True,
    )
    print(f"\nCalibration done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ── Step 5: Verify ARL0 ────────────────────────────────────────────────
    arl_verified = {}
    if verify:
        B_v = 200 if fast else 1000
        arl_verified = verify_arl0(ic, monitors, ucls,
                                   config.N_WINDOW, K_max_run, rng, B_v)

    # ── Step 6: Save ───────────────────────────────────────────────────────
    save_checkpoint(ic, monitors, ucls, seed, oracle_var=oracle_var)
    write_summary(ucls, arl_verified)

    print("\nCalibration complete. Next step: python run_phase2.py")
    return ic, monitors, ucls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lstm",    action="store_true")
    parser.add_argument("--fast",       action="store_true")
    parser.add_argument("--oracle-var", action="store_true",
                        help="Use from_true_model for VARResidual too")
    parser.add_argument("--no-verify",  action="store_true")
    parser.add_argument("--seed",       type=int, default=config.SEED)
    args = parser.parse_args()

    run_calibration(
        include_lstm = not args.no_lstm,
        fast         = args.fast,
        seed         = args.seed,
        oracle_var   = args.oracle_var,
        verify       = not args.no_verify,
    )

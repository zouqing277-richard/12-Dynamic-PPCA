"""
run_calibration.py
──────────────────
Step 1 of 2.  Run this ONCE to calibrate UCLs.

What it does:
  1. Build IC model  (U0 fixed by SEED)
  2. Generate Phase I data  (N = N_TRAIN)
  3. Fit all monitoring methods on Phase I data
  4. Two-phase MC bisection to find UCL for each method
     until |ARL0(h) - 200| <= BISECT_TOL
  5. Save everything to  results/calibration/checkpoint.pkl

After this script finishes, run:
  python run_phase2.py

CLI:
  python run_calibration.py               # all methods
  python run_calibration.py --no-lstm     # skip LSTM-AE
  python run_calibration.py --fast        # debug mode (tiny B values)
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

def fit_all_methods(X_train, include_lstm=True):
    monitors = {
        "dyppca":       DyPPCA(q=config.Q).fit(X_train),
        "dpca":         DPCA(cpv_threshold=config.DPCA_CPV,
                             lag=config.DPCA_LAG).fit(X_train),
        "static_ppca":  StaticPPCA(q=config.Q).fit(X_train),
        "var_residual": VARResidual().fit(X_train),
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

def save_checkpoint(ic_model, monitors, ucls, seed, path=CHECKPOINT_FILE):
    """
    Save calibration results to a single pickle file.

    For LSTM-AE the PyTorch model state_dict is extracted and stored as
    plain numpy/dict so the checkpoint is self-contained.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Serialise LSTM model separately if present
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
        "monitors":   {k: v for k, v in monitors.items() if k != "lstm_ae"},
        "lstm_state": lstm_state,
        "ucls":       ucls,
        "seed":       seed,
        "config": {
            "P": config.P, "Q": config.Q, "SIGMA0": config.SIGMA0,
            "N_TRAIN": config.N_TRAIN, "N_WINDOW": config.N_WINDOW,
            "ARL0": config.ARL0, "K_MAX": config.K_MAX,
            "BISECT_TOL": config.BISECT_TOL,
        }
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved → {path}")


def write_summary(ucls, arl_verified, path=SUMMARY_FILE):
    """Write human-readable UCL summary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "UCL Calibration Summary",
        f"ARL0 target = {config.ARL0}  |  bisect_tol = {config.BISECT_TOL}",
        "=" * 50,
    ]
    for method, ucl_dict in ucls.items():
        lines.append(f"\n[{method}]")
        if isinstance(ucl_dict, dict):
            for k, v in ucl_dict.items():
                lines.append(f"  {k:12s} = {v:.4f}")
        else:
            lines.append(f"  UCL = {ucl_dict:.4f}")
        if method in arl_verified:
            lines.append(f"  Verified ARL0 ≈ {arl_verified[method]:.1f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick ARL0 verification after calibration
# ─────────────────────────────────────────────────────────────────────────────

def verify_arl0(ic_model, monitors, ucls, n_window, K_max, rng, B_verify=1000):
    """
    Quick check: generate B_verify IC replications and confirm ARL0 ≈ 200.
    Returns dict {method_name: estimated_arl0}.
    """
    from data_generator import simulate_ic
    from evaluation     import ALARM_RULES

    print(f"\nVerifying ARL0 (B_verify={B_verify}) ...")
    results = {}
    for name, monitor in monitors.items():
        alarm_fn = ALARM_RULES["case1"][name]
        delays   = []
        WARMUP   = 5 * n_window
        for _ in range(B_verify):
            X_buf = simulate_ic(ic_model, WARMUP + n_window + 1, rng=rng)
            delay = K_max
            for k in range(K_max):
                X_win = X_buf[WARMUP:] if k == 0 else \
                        np.vstack([X_buf[-1:],
                                   simulate_ic(ic_model, n_window + 1, rng=rng)[1:]])
                if k > 0:
                    X_buf = X_win
                row = monitor.monitor_window(X_win)
                if alarm_fn(row, ucls[name]):
                    delay = k + 1
                    break
            delays.append(delay)
        arl0_hat = float(np.mean(delays))
        err      = abs(arl0_hat - config.ARL0) / config.ARL0 * 100
        ok       = "✓" if err < 15 else "~"
        print(f"  {name:<16}  ARL0 = {arl0_hat:6.1f}  (err={err:.1f}%)  {ok}")
        results[name] = arl0_hat
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(include_lstm=True, fast=False, seed=config.SEED,
                    verify=True):

    B_coarse  = 20   if fast else config.B_COARSE
    n_coarse  = 4    if fast else config.N_COARSE
    B_fine    = 50   if fast else config.B_FINE
    n_fine    = 3    if fast else config.N_FINE
    B_boot    = 200  if fast else config.B_BOOTSTRAP
    K_max_run = 50   if fast else config.K_MAX

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("DyPPCA UCL Calibration")
    print(f"  p={config.P}  q={config.Q}  N_TRAIN={config.N_TRAIN}")
    print(f"  ARL0={config.ARL0}  K_max={K_max_run}  tol={config.BISECT_TOL}")
    print(f"  Coarse: {n_coarse}×{B_coarse}   Fine: {n_fine}×{B_fine}")
    print(f"  LSTM: {'yes' if include_lstm else 'no'}")
    print("=" * 60)

    # ── Step 1: IC model ──────────────────────────────────────────────────
    ic = build_ic_model(config.P, config.Q, config.SIGMA0,
                        config.LAMBDA0, config.B0, seed=seed)

    # ── Step 2: Phase I (ONE dataset, fixed across all experiments) ───────
    print(f"\nGenerating Phase I data (N={config.N_TRAIN}) ...", flush=True)
    X_train = simulate_ic(ic, config.N_TRAIN + 1, rng=rng)

    # ── Step 3: Fit all methods ───────────────────────────────────────────
    print("Fitting all methods ...", flush=True)
    t0       = time.time()
    monitors = fit_all_methods(X_train, include_lstm=include_lstm)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # ── Step 4: Calibrate UCLs ────────────────────────────────────────────
    print("\nCalibrating UCLs (two-phase MC bisection) ...", flush=True)
    t0   = time.time()
    ucls = calibrate_all(
        model_ic    = ic,
        methods     = monitors,
        n_window    = config.N_WINDOW,
        arl0        = config.ARL0,
        K_max       = K_max_run,
        rng         = rng,
        B_coarse    = B_coarse,
        n_coarse    = n_coarse,
        B_fine      = B_fine,
        n_fine      = n_fine,
        bisect_tol  = config.BISECT_TOL,
        B_bootstrap = B_boot,
        verbose     = True,
    )
    print(f"\nCalibration done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ── Step 5: Verify ARL0 ───────────────────────────────────────────────
    arl_verified = {}
    if verify:
        B_v = 200 if fast else 1000
        arl_verified = verify_arl0(ic, monitors, ucls,
                                   config.N_WINDOW, K_max_run, rng, B_v)

    # ── Step 6: Save checkpoint ───────────────────────────────────────────
    save_checkpoint(ic, monitors, ucls, seed)
    write_summary(ucls, arl_verified)

    print("\n✓ Calibration complete.")
    print(f"  Next step:  python run_phase2.py")
    return ic, monitors, ucls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate UCLs and save checkpoint"
    )
    parser.add_argument("--no-lstm", action="store_true",
                        help="Skip LSTM-AE")
    parser.add_argument("--fast",    action="store_true",
                        help="Debug mode (tiny B, fast but inaccurate)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip ARL0 verification step")
    parser.add_argument("--seed",    type=int, default=config.SEED)
    args = parser.parse_args()

    run_calibration(
        include_lstm = not args.no_lstm,
        fast         = args.fast,
        seed         = args.seed,
        verify       = not args.no_verify,
    )

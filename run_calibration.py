"""
run_calibration.py  —  Step 1 of 2.

CRN-based UCL calibration:
  1. Build IC model from true parameters (equations 63–64 of the paper).
  2. Generate Phase I data for DPCA / VARResidual / LSTM-AE.
  3. Fit monitors  (DyPPCA + StaticPPCA use from_true_model).
  4. For each method, pre-generate IC stats matrix once (CRN):
       stats_mat[b, k]  =  statistic for sequence b at window k
     Then bisect each UCL on the same matrix → ARL₀(h) is monotone.
  5. Verify calibration with a fresh CRN stats matrix.
  6. Save checkpoint.pkl.

CLI:
  python run_calibration.py                 # full run
  python run_calibration.py --no-lstm       # skip LSTM-AE
  python run_calibration.py --fast          # debug (seconds, inaccurate)
  python run_calibration.py --oracle-var    # VARResidual also uses true params
  python run_calibration.py --no-verify     # skip post-calibration check
  python run_calibration.py --jobs 4        # limit parallel jobs
"""

import os, sys, time, argparse, pickle
import numpy as np

import config
from data_generator import build_ic_model, simulate_ic
from methods        import DyPPCA, DPCA, StaticPPCA, VARResidual
from calibration    import calibrate_all, verify_arl0_crn

CHECKPOINT_DIR  = "results/calibration"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")
SUMMARY_FILE    = os.path.join(CHECKPOINT_DIR, "ucls_summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Fit monitors
# ─────────────────────────────────────────────────────────────────────────────

def fit_all_methods(X_train, ic_model, include_lstm=True, oracle_var=False):
    """
    DyPPCA and StaticPPCA: always use from_true_model (true IC parameters).
    DPCA, VARResidual, LSTM-AE: fit on Phase I data.
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
# Save / load
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
            "B_CRN": config.B_CRN, "K_MAX_CRN": config.K_MAX_CRN,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved → {path}")


def write_summary(ucls, arl_verified, path=SUMMARY_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "UCL Calibration Summary  (CRN-based)",
        f"ARL0 target = {config.ARL0}  |  tol = {config.BISECT_TOL}",
        f"B_CRN = {config.B_CRN}  |  K_MAX_CRN = {config.K_MAX_CRN}",
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(include_lstm=True, fast=False, seed=config.SEED,
                    oracle_var=False, verify=True, n_jobs=-1):

    B_crn     = 200              if fast else config.B_CRN
    K_max_crn = 100              if fast else config.K_MAX_CRN
    n_coarse  = 4                if fast else config.N_COARSE
    tol       = 20.0             if fast else config.BISECT_TOL
    max_fine  = 4                if fast else config.MAX_FINE

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("DyPPCA UCL Calibration  (CRN-based)")
    print(f"  p={config.P}  q={config.Q}  sigma0={config.SIGMA0}")
    print(f"  n={config.N_WINDOW}  ARL0={config.ARL0}  tol={tol}")
    print(f"  B_crn={B_crn}  K_max_crn={K_max_crn}  n_coarse={n_coarse}")
    print(f"  DyPPCA+StaticPPCA: from_true_model")
    print(f"  DPCA+VAR+LSTM: Phase I fit")
    print(f"  n_jobs={n_jobs}")
    print("=" * 60)

    # Step 1: IC model
    ic = build_ic_model(config.P, config.Q, config.SIGMA0,
                        config.A0, config.B0, config.PSI0)

    # Step 2: Phase I data (for DPCA / VARResidual / LSTM)
    print(f"\nGenerating Phase I data  (N={config.N_TRAIN}) ...", flush=True)
    X_train = simulate_ic(ic, config.N_TRAIN + 1, rng=rng)

    # Step 3: Fit monitors
    print("Fitting monitors ...", flush=True)
    t0       = time.time()
    monitors = fit_all_methods(X_train, ic,
                               include_lstm=include_lstm,
                               oracle_var=oracle_var)
    print(f"  Done in {time.time()-t0:.1f}s\n", flush=True)

    # Step 4: CRN calibration
    print("Calibrating UCLs  (CRN) ...\n", flush=True)
    t0   = time.time()
    ucls = calibrate_all(
        model_ic   = ic,
        methods    = monitors,
        n_window   = config.N_WINDOW,
        arl0       = config.ARL0,
        K_max_crn  = K_max_crn,
        rng        = rng,
        B_crn      = B_crn,
        n_coarse   = n_coarse,
        bisect_tol = tol,
        max_fine   = max_fine,
        fast       = fast,
        verbose    = True,
        n_jobs     = n_jobs,
    )
    print(f"\nCalibration done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Step 5: Verify ARL0 with fresh CRN matrix
    arl_verified = {}
    if verify and not fast:
        B_v = max(500, B_crn // 10)
        print(f"\nVerifying ARL0  (fresh CRN matrix, B={B_v}) ...", flush=True)
        arl_verified = verify_arl0_crn(
            ic, monitors, ucls,
            n_window   = config.N_WINDOW,
            K_max_crn  = K_max_crn,
            B_verify   = B_v,
            rng        = rng,
            verbose    = True,
        )

    # Step 6: Save
    save_checkpoint(ic, monitors, ucls, seed, oracle_var=oracle_var)
    write_summary(ucls, arl_verified)

    print("\n✓  Calibration complete.")
    print("   Next step:  python run_phase2.py")
    return ic, monitors, ucls


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRN UCL calibration")
    parser.add_argument("--no-lstm",    action="store_true",
                        help="Skip LSTM-AE")
    parser.add_argument("--fast",       action="store_true",
                        help="Debug mode (seconds, UCLs inaccurate)")
    parser.add_argument("--oracle-var", action="store_true",
                        help="VARResidual also uses from_true_model")
    parser.add_argument("--no-verify",  action="store_true",
                        help="Skip post-calibration ARL0 check")
    parser.add_argument("--jobs",       type=int, default=-1,
                        help="joblib n_jobs (default=-1, all cores)")
    parser.add_argument("--seed",       type=int, default=config.SEED)
    args = parser.parse_args()

    run_calibration(
        include_lstm = not args.no_lstm,
        fast         = args.fast,
        seed         = args.seed,
        oracle_var   = args.oracle_var,
        verify       = not args.no_verify,
        n_jobs       = args.jobs,
    )

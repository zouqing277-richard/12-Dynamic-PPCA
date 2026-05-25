"""
run_phase2.py
─────────────
Step 2 of 2.  Load calibration checkpoint and run ARL1 experiments.

Requires:  results/calibration/checkpoint.pkl
           (produced by run_calibration.py)

CLI:
  python run_phase2.py                        # all 5 cases
  python run_phase2.py --cases case1 case3    # specific cases
  python run_phase2.py --b1 500              # fewer OC replications
  python run_phase2.py --fast                 # debug mode
"""

import os, sys, time, argparse, pickle
import numpy as np
import pandas as pd

import config
from data_generator import simulate_ic
from evaluation     import run_arl_experiment, diagnostic_ratios

CHECKPOINT_FILE = "results/calibration/checkpoint.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path=CHECKPOINT_FILE):
    """
    Load ic_model, monitors, and ucls from the saved checkpoint.
    Reconstructs the LSTM-AE monitor from its saved state_dict if present.
    """
    if not os.path.exists(path):
        print(f"ERROR: Checkpoint not found at '{path}'")
        print("       Run  python run_calibration.py  first.")
        sys.exit(1)

    with open(path, "rb") as f:
        ck = pickle.load(f)

    ic_model = ck["ic_model"]
    monitors = ck["monitors"]   # all methods except lstm_ae
    ucls     = ck["ucls"]

    # Rebuild LSTM-AE monitor from saved state if it was included
    if ck.get("lstm_state") is not None:
        st = ck["lstm_state"]
        from methods.lstm_ae import LSTMAEMonitor, LSTMAutoencoder
        import torch

        lstm_mon = LSTMAEMonitor(
            input_dim   = st["input_dim"],
            hidden_dim  = st["hidden_dim"],
            latent_dim  = st["latent_dim"],
            num_layers  = st["num_layers"],
        )
        # Rebuild model and load weights
        model = LSTMAutoencoder(st["input_dim"], st["hidden_dim"],
                                st["latent_dim"], st["num_layers"])
        state_dict = {k: torch.from_numpy(v)
                      for k, v in st["state_dict"].items()}
        model.load_state_dict(state_dict)
        model.eval()

        lstm_mon.model      = model
        lstm_mon.mu_r       = st["mu_r"]
        lstm_mon.Sig_r_inv  = st["Sig_r_inv"]
        monitors["lstm_ae"] = lstm_mon

    print(f"Checkpoint loaded from '{path}'")
    print(f"  Methods:  {list(monitors.keys())}")
    print(f"  Config:   p={ck['config']['P']}  q={ck['config']['Q']}"
          f"  N={ck['config']['N_TRAIN']}  n={ck['config']['N_WINDOW']}"
          f"  ARL0={ck['config']['ARL0']}")
    return ic_model, monitors, ucls


# ─────────────────────────────────────────────────────────────────────────────
# One OC case
# ─────────────────────────────────────────────────────────────────────────────

def run_one_case(case, ic_model, monitors, ucls, shifts,
                 B1, n_window, K_max, rng, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    method_names = list(monitors.keys())
    rows_arl, rows_se, rows_diag = [], [], []

    print(f"\n{chr(8212)*60}")
    print(f"  {case.upper()}  |  shifts={shifts}  B1={B1}")
    print(f"{chr(8212)*60}")

    for d in shifts:
        t0 = time.time()
        print(f"\n  d = {d}", flush=True)

        row_arl = {"magnitude": d}
        row_se  = {"magnitude": d}

        for name in method_names:
            arl_mean, arl_se, _ = run_arl_experiment(
                method_name = name,
                monitor     = monitors[name],
                ucls        = ucls[name],
                model_ic    = ic_model,
                case        = case,
                d           = d,
                n_reps      = B1,
                n_window    = n_window,
                K_max       = K_max,
                rng         = rng,
            )
            row_arl[name] = round(arl_mean, 2)
            row_se[name]  = round(arl_se,   3)
            print(f"    {name:<16}  ARL1={arl_mean:7.2f}  SE={arl_se:.3f}",
                  flush=True)

        rows_arl.append(row_arl)
        rows_se.append(row_se)

        # DyPPCA diagnostic ratios
        rhos = diagnostic_ratios(
            dyppca_monitor = monitors["dyppca"],
            ucls           = ucls["dyppca"],
            model_ic       = ic_model,
            case           = case,
            d              = d,
            n_reps         = min(B1, 500),
            n_window       = n_window,
            K_max          = K_max,
            rng            = rng,
        )
        rhos["magnitude"] = d
        rows_diag.append(rhos)
        print(f"    rho=({rhos.get("rho1",0):.3f}, {rhos.get("rho2",0):.3f}, "
              f"{rhos.get("rho3",0):.3f}, {rhos.get("rho4",0):.3f})"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    df_arl  = pd.DataFrame(rows_arl).set_index("magnitude")
    df_se   = pd.DataFrame(rows_se).set_index("magnitude")
    df_diag = pd.DataFrame(rows_diag).set_index("magnitude")

    df_arl.to_csv( os.path.join(save_dir, f"{case}_arl.csv"))
    df_se.to_csv(  os.path.join(save_dir, f"{case}_se.csv"))
    df_diag.to_csv(os.path.join(save_dir, f"{case}_diag.csv"))
    print(f"  Saved {case}_*.csv -> {save_dir}/")
    return df_arl, df_se, df_diag


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all(cases=None, B1=None, fast=False,
            seed=None, save_dir="results/tables",
            checkpoint=CHECKPOINT_FILE):

    if cases is None:
        cases = ["case1", "case2", "case3", "case4", "case5"]

    B1_run    = 20   if fast else (B1 or config.B1)
    K_max_run = 50   if fast else config.K_MAX

    # Load calibration results (no re-fitting, no re-calibrating)
    ic_model, monitors, ucls = load_checkpoint(checkpoint)

    # Use seed from checkpoint if not overridden
    rng = np.random.default_rng(seed or config.SEED + 1)

    print(f"\n{'='*60}")
    print("DyPPCA Phase II ARL Experiment")
    print(f"  B1={B1_run}  K_max={K_max_run}  Cases: {cases}")
    print(f"{'='*60}")

    all_results = {}
    for case in cases:
        df_arl, df_se, df_diag = run_one_case(
            case     = case,
            ic_model = ic_model,
            monitors = monitors,
            ucls     = ucls,
            shifts   = config.SHIFTS[case],
            B1       = B1_run,
            n_window = config.N_WINDOW,
            K_max    = K_max_run,
            rng      = rng,
            save_dir = save_dir,
        )
        all_results[case] = (df_arl, df_se, df_diag)

    print(f"\n{'='*60}")
    print(f"All done.  Results -> {save_dir}/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phase II ARL experiment (requires calibration checkpoint)"
    )
    parser.add_argument("--cases", nargs="+", default=None, metavar="CASE",
                        help="Cases to run (default: all 5)")
    parser.add_argument("--b1",   type=int, default=None, metavar="N",
                        help=f"OC replications (default: {config.B1})")
    parser.add_argument("--fast", action="store_true",
                        help="Debug mode: B1=20, K_max=50")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for OC sequences")
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE,
                        help=f"Path to checkpoint file (default: {CHECKPOINT_FILE})")
    args = parser.parse_args()

    run_all(
        cases      = args.cases,
        B1         = args.b1,
        fast       = args.fast,
        seed       = args.seed,
        checkpoint = args.checkpoint,
    )

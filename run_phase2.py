"""
run_phase2.py
─────────────
Step 2 of 2.  Load calibration checkpoint and run ARL₁ experiments.

Requires:  results/calibration/checkpoint.pkl
           (produced by run_calibration.py)

For each OC case the comparison set is defined in evaluation.OC_COMPARISON_STATS:
only the (method, statistic) pairs that are theoretically relevant to that case
are evaluated, keeping computation focused and results interpretable.

CSV output columns are named  "<method>.<stat>"  (e.g. "dyppca.t1", "dpca.T2").

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
from evaluation import run_arl_experiment, diagnostic_ratios, OC_COMPARISON_STATS

CHECKPOINT_FILE = "results/calibration/checkpoint.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path=CHECKPOINT_FILE):
    """Load ic_model, monitors, and ucls; reconstruct LSTM-AE if saved."""
    if not os.path.exists(path):
        print(f"ERROR: Checkpoint not found at '{path}'")
        print("       Run  python run_calibration.py  first.")
        sys.exit(1)

    with open(path, "rb") as f:
        ck = pickle.load(f)

    ic_model = ck["ic_model"]
    monitors = ck["monitors"]
    ucls     = ck["ucls"]

    if ck.get("lstm_state") is not None:
        st = ck["lstm_state"]
        from methods.lstm_ae import LSTMAEMonitor, LSTMAutoencoder
        import torch

        lstm_mon = LSTMAEMonitor(
            input_dim  = st["input_dim"],
            hidden_dim = st["hidden_dim"],
            latent_dim = st["latent_dim"],
            num_layers = st["num_layers"],
        )
        model = LSTMAutoencoder(st["input_dim"], st["hidden_dim"],
                                st["latent_dim"], st["num_layers"])
        state_dict = {k: torch.from_numpy(v)
                      for k, v in st["state_dict"].items()}
        model.load_state_dict(state_dict)
        model.eval()

        lstm_mon.model     = model
        lstm_mon.mu_r      = st["mu_r"]
        lstm_mon.Sig_r_inv = st["Sig_r_inv"]
        monitors["lstm_ae"] = lstm_mon

    oracle_flag = ck.get("oracle", False)
    print(f"Checkpoint loaded from '{path}'")
    print(f"  Methods : {list(monitors.keys())}")
    print(f"  Oracle  : {oracle_flag}")
    print(f"  Config  : p={ck['config']['P']}  q={ck['config']['Q']}"
          f"  N={ck['config']['N_TRAIN']}  n={ck['config']['N_WINDOW']}"
          f"  ARL0={ck['config']['ARL0']}")
    return ic_model, monitors, ucls


# ─────────────────────────────────────────────────────────────────────────────
# One OC case
# ─────────────────────────────────────────────────────────────────────────────

def run_one_case(case, ic_model, monitors, ucls, shifts,
                 B1, n_window, K_max, rng, save_dir):
    """
    Evaluate ARL₁ for all (method, statistic) pairs in OC_COMPARISON_STATS[case].

    CSV columns: "<method>.<stat>"  e.g. "dyppca.t1", "dpca.T2".
    """
    os.makedirs(save_dir, exist_ok=True)

    # Only run pairs whose method is present in monitors
    pairs = [
        (m, s) for m, s in OC_COMPARISON_STATS[case]
        if m in monitors
    ]
    col_names = [f"{m}.{s}" for m, s in pairs]

    print(f"\n{'─'*60}")
    print(f"  {case.upper()}  |  shifts={shifts}  B1={B1}")
    print(f"  Comparing: {col_names}")
    print(f"{'─'*60}")

    rows_arl, rows_se, rows_diag = [], [], []

    for d in shifts:
        t0 = time.time()
        print(f"\n  d = {d}", flush=True)

        row_arl = {"magnitude": d}
        row_se  = {"magnitude": d}

        # Common Random Numbers: all pairs see the same OC trajectories
        crn_seed = int(rng.integers(0, 2**31))

        for base_method, stat_name in pairs:
            col = f"{base_method}.{stat_name}"
            rng_pair = np.random.default_rng(crn_seed)

            arl_mean, arl_se, _ = run_arl_experiment(
                base_method = base_method,
                stat_name   = stat_name,
                monitor     = monitors[base_method],
                ucls        = ucls[base_method],
                model_ic    = ic_model,
                case        = case,
                d           = d,
                n_reps      = B1,
                n_window    = n_window,
                K_max       = K_max,
                rng         = rng_pair,
            )
            row_arl[col] = round(arl_mean, 2)
            row_se[col]  = round(arl_se,   3)
            print(f"    {col:<24}  ARL1={arl_mean:7.2f}  SE={arl_se:.3f}",
                  flush=True)

        rows_arl.append(row_arl)
        rows_se.append(row_se)

        # DyPPCA diagnostic component ratios (always computed if available)
        rhos = {"rho1": np.nan, "rho2": np.nan,
                "rho3": np.nan, "rho4": np.nan}
        if "dyppca" in monitors:
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

        print(f"    rho=({rhos.get('rho1', 0):.3f}, {rhos.get('rho2', 0):.3f}, "
              f"{rhos.get('rho3', 0):.3f}, {rhos.get('rho4', 0):.3f})"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    df_arl  = pd.DataFrame(rows_arl).set_index("magnitude")
    df_se   = pd.DataFrame(rows_se).set_index("magnitude")
    df_diag = pd.DataFrame(rows_diag).set_index("magnitude")

    # Also save a human-readable "_std" alias (for plot_results.py compatibility)
    df_arl.to_csv( os.path.join(save_dir, f"{case}_arl.csv"))
    df_se.to_csv(  os.path.join(save_dir, f"{case}_se.csv"))
    df_se.to_csv(  os.path.join(save_dir, f"{case}_std.csv"))   # alias
    df_diag.to_csv(os.path.join(save_dir, f"{case}_diag.csv"))
    print(f"  Saved {case}_*.csv → {save_dir}/")
    return df_arl, df_se, df_diag


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all(cases=None, B1=None, fast=False,
            seed=None, save_dir="results/tables",
            checkpoint=CHECKPOINT_FILE):

    if cases is None:
        cases = ["case1", "case2", "case3", "case4", "case5"]

    B1_run    = 20  if fast else (B1 or config.B1)
    K_max_run = 50  if fast else config.K_MAX

    ic_model, monitors, ucls = load_checkpoint(checkpoint)
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
    print(f"All done.  Results → {save_dir}/")
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

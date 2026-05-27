"""
run_phase2.py  —  Step 2 of 2.  Requires checkpoint.pkl from run_calibration.py.

Uses CRN for Phase II ARL₁ computation:
  For each (case, d), all methods see the SAME B_crn OC trajectories.
  ARL₁ differences between methods are free of cross-method MC noise.

CLI:
  python run_phase2.py                        # all 5 cases
  python run_phase2.py --cases case1 case3    # specific cases
  python run_phase2.py --B 1000              # fewer OC replications
  python run_phase2.py --fast                 # debug (B=50, K_max=100)
"""

import os, sys, time, argparse, pickle
import numpy as np
import pandas as pd

import config
from evaluation import run_arl_experiment_crn, diagnostic_ratios_crn, OC_COMPARISON_STATS

CHECKPOINT_FILE = "results/calibration/checkpoint.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path=CHECKPOINT_FILE):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run run_calibration.py first")
        sys.exit(1)
    with open(path, "rb") as f:
        ck = pickle.load(f)
    monitors = ck["monitors"]
    if ck.get("lstm_state"):
        st = ck["lstm_state"]
        from methods.lstm_ae import LSTMAEMonitor, LSTMAutoencoder
        import torch
        lstm = LSTMAEMonitor(st["input_dim"], st["hidden_dim"],
                             st["latent_dim"], st["num_layers"])
        model = LSTMAutoencoder(st["input_dim"], st["hidden_dim"],
                                st["latent_dim"], st["num_layers"])
        model.load_state_dict({k: torch.from_numpy(v)
                               for k, v in st["state_dict"].items()})
        model.eval()
        lstm.model = model
        lstm.mu_r  = st["mu_r"]; lstm.Sig_r_inv = st["Sig_r_inv"]
        monitors["lstm_ae"] = lstm

    print(f"Checkpoint loaded: methods={list(monitors.keys())}")
    cfg = ck["config"]
    print(f"  p={cfg['P']}  q={cfg['Q']}  n={cfg['N_WINDOW']}  ARL0={cfg['ARL0']}")
    return ck["ic_model"], monitors, ck["ucls"]


# ─────────────────────────────────────────────────────────────────────────────
# One OC case
# ─────────────────────────────────────────────────────────────────────────────

def run_one_case(case, ic_model, monitors, ucls, shifts,
                 B_crn, K_max_crn, rng, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    pairs     = [(m, s) for m, s in OC_COMPARISON_STATS[case] if m in monitors]
    col_names = [f"{m}.{s}" for m, s in pairs]

    print(f"\n{'─'*60}")
    print(f"  {case.upper()}  |  shifts={shifts}  B={B_crn}")
    print(f"  Comparing: {col_names}")
    print(f"{'─'*60}")

    rows_arl, rows_se, rows_diag = [], [], []

    for d in shifts:
        t0 = time.time()
        print(f"\n  d = {d}", flush=True)

        # CRN: all methods see the same OC sequences
        results = run_arl_experiment_crn(
            case       = case,
            d          = d,
            monitors   = monitors,
            ucls       = ucls,
            model_ic   = ic_model,
            n_window   = config.N_WINDOW,
            K_max_crn  = K_max_crn,
            B_crn      = B_crn,
            rng        = rng,
        )

        row_arl = {"magnitude": d}
        row_se  = {"magnitude": d}
        for key, val in results.items():
            row_arl[key] = val["arl"]
            row_se[key]  = val["se"]
            print(f"    {key:<30}: ARL={val['arl']:7.2f}  SE={val['se']:.3f}",
                  flush=True)

        rows_arl.append(row_arl)
        rows_se.append(row_se)

        # DyPPCA diagnostic ratios
        rhos = {"rho1":np.nan,"rho2":np.nan,"rho3":np.nan,"rho4":np.nan}
        if "dyppca" in monitors:
            rhos = diagnostic_ratios_crn(
                dyppca_monitor = monitors["dyppca"],
                ucls           = ucls,
                model_ic       = ic_model,
                case           = case,
                d              = d,
                n_window       = config.N_WINDOW,
                K_max_crn      = K_max_crn,
                B_crn          = min(B_crn, 500),
                rng            = rng,
            )
        rhos["magnitude"] = d
        rows_diag.append(rhos)

        print(f"    ρ=({rhos.get('rho1',0):.3f}, {rhos.get('rho2',0):.3f}, "
              f"{rhos.get('rho3',0):.3f}, {rhos.get('rho4',0):.3f})"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    df_arl  = pd.DataFrame(rows_arl).set_index("magnitude")
    df_se   = pd.DataFrame(rows_se).set_index("magnitude")
    df_diag = pd.DataFrame(rows_diag).set_index("magnitude")

    df_arl.to_csv(os.path.join(save_dir, f"{case}_arl.csv"))
    df_se.to_csv( os.path.join(save_dir, f"{case}_se.csv"))
    df_se.to_csv( os.path.join(save_dir, f"{case}_std.csv"))
    df_diag.to_csv(os.path.join(save_dir,f"{case}_diag.csv"))
    print(f"  Saved {case}_*.csv → {save_dir}/")
    return df_arl, df_se, df_diag


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all(cases=None, B_crn=None, fast=False,
            seed=None, save_dir="results/tables",
            checkpoint=CHECKPOINT_FILE):

    if cases is None:
        cases = ["case1","case2","case3","case4","case5"]

    B_crn_run    = 50             if fast else (B_crn or config.B_CRN)
    K_max_crn    = 100            if fast else config.K_MAX_CRN

    ic_model, monitors, ucls = load_checkpoint(checkpoint)
    rng = np.random.default_rng(seed or config.SEED + 1)

    print(f"\n{'='*60}")
    print(f"DyPPCA Phase II ARL experiment  (CRN)")
    print(f"  B_crn={B_crn_run}  K_max_crn={K_max_crn}  Cases: {cases}")
    print(f"{'='*60}")

    for case in cases:
        run_one_case(
            case       = case,
            ic_model   = ic_model,
            monitors   = monitors,
            ucls       = ucls,
            shifts     = config.SHIFTS[case],
            B_crn      = B_crn_run,
            K_max_crn  = K_max_crn,
            rng        = rng,
            save_dir   = save_dir,
        )

    print(f"\n{'='*60}")
    print(f"Done.  Results → {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase II ARL experiment (CRN)")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--B",     type=int,  default=10000,
                        help=f"OC sequences (default: {config.B_CRN})")
    parser.add_argument("--fast",  action="store_true",
                        help="Debug: B=50, K_max=100")
    parser.add_argument("--seed",  type=int,  default=None)
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    run_all(cases=args.cases, B_crn=args.B, fast=args.fast,
            seed=args.seed, checkpoint=args.checkpoint)

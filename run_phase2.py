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
from evaluation import run_arl_experiment_crn, OC_COMPARISON_STATS

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
                 B_crn, K_max_crn, rng, save_dir, loc=None, loc_label=""):
    os.makedirs(save_dir, exist_ok=True)

    pairs     = [(m, s) for m, s in OC_COMPARISON_STATS[case] if m in monitors]
    col_names = [f"{m}.{s}" for m, s in pairs]

    loc_str = f"  loc={loc}" if loc is not None else ""
    print(f"\n{'─'*60}")
    print(f"  {case.upper()}  |  shifts={shifts}  B={B_crn}{loc_str}")
    print(f"  Comparing: {col_names}")
    print(f"{'─'*60}")

    rows_arl, rows_se = [], []

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
            loc        = loc,
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


        print(f"  [{time.time()-t0:.0f}s]", flush=True)

    df_arl  = pd.DataFrame(rows_arl).set_index("magnitude")
    df_se   = pd.DataFrame(rows_se).set_index("magnitude")

    suffix = f"_{loc_label}" if loc_label else ""
    df_arl.to_csv(os.path.join(save_dir, f"{case}{suffix}_arl.csv"))
    df_se.to_csv( os.path.join(save_dir, f"{case}{suffix}_se.csv"))
    df_se.to_csv( os.path.join(save_dir, f"{case}{suffix}_std.csv"))
    print(f"  Saved {case}{suffix}_*.csv → {save_dir}/")
    return df_arl, df_se


# ─────────────────────────────────────────────────────────────────────────────
# LSTM-only run  (merges into existing CSVs)
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm_only(cases=None, B_crn=None, fast=False,
                  seed=None, save_dir="results/tables",
                  checkpoint=CHECKPOINT_FILE):
    """
    Run LSTM-AE Phase II only, then merge the new column into existing CSVs.
    Use after  python run_phase2.py --no-lstm  has already produced results.
    """
    if cases is None:
        cases = ["case1","case2","case3","case4","case5"]

    B_crn_run = 50  if fast else (B_crn or config.B_CRN)
    K_max_crn = 100 if fast else config.K_MAX_CRN

    ic_model, monitors, ucls = load_checkpoint(checkpoint)
    if "lstm_ae" not in monitors:
        print("ERROR: checkpoint has no lstm_ae — run run_calibration.py first")
        return

    # Keep only LSTM
    lstm_monitors = {"lstm_ae": monitors["lstm_ae"]}
    lstm_ucls     = {"lstm_ae": ucls["lstm_ae"]}

    rng = np.random.default_rng(seed or config.SEED + 2)

    print(f"\n{'='*60}")
    print(f"LSTM-AE only Phase II  (B={B_crn_run}, K={K_max_crn})")
    print(f"{'='*60}")

    for case in cases:
        locations = config.LOCATIONS[case]
        for i, loc in enumerate(locations):
            loc_label = f"loc{i+1}"
            print(f"\n  {case.upper()} {loc_label}  loc={loc}", flush=True)

            rows_arl, rows_se = [], []
            for d in config.SHIFTS[case]:
                t0 = time.time()
                results = run_arl_experiment_crn(
                    case=case, d=d,
                    monitors=lstm_monitors, ucls=lstm_ucls,
                    model_ic=ic_model,
                    n_window=config.N_WINDOW,
                    K_max_crn=K_max_crn,
                    B_crn=B_crn_run,
                    rng=rng, loc=loc,
                )
                rows_arl.append({"magnitude": d,
                                  **{k: v["arl"] for k, v in results.items()}})
                rows_se.append( {"magnitude": d,
                                  **{k: v["se"]  for k, v in results.items()}})
                for key, val in results.items():
                    print(f"    d={d}  {key}: ARL={val['arl']:.2f}  "
                          f"SE={val['se']:.3f}  [{time.time()-t0:.0f}s]",
                          flush=True)

            df_new_arl = pd.DataFrame(rows_arl).set_index("magnitude")
            df_new_se  = pd.DataFrame(rows_se).set_index("magnitude")

            # Merge into existing CSVs (or create new if missing)
            os.makedirs(save_dir, exist_ok=True)
            for suffix, df_new in [("arl", df_new_arl),
                                    ("se",  df_new_se),
                                    ("std", df_new_se)]:
                path = os.path.join(save_dir, f"{case}_{loc_label}_{suffix}.csv")
                if os.path.exists(path):
                    df_exist = pd.read_csv(path, index_col=0)
                    # Add / overwrite LSTM column
                    for col in df_new.columns:
                        df_exist[col] = df_new[col]
                    df_exist.to_csv(path)
                else:
                    df_new.to_csv(path)
            print(f"  Merged lstm_ae into {case}_{loc_label}_*.csv")

    print(f"\n{'='*60}")
    print(f"Done.  LSTM results merged → {save_dir}/")



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all(cases=None, B_crn=None, fast=False,
            seed=None, save_dir="results/tables",
            checkpoint=CHECKPOINT_FILE, skip_lstm=False):

    if cases is None:
        cases = ["case1","case2","case3","case4","case5"]

    B_crn_run    = 50             if fast else (B_crn or config.B_CRN)
    K_max_crn    = 100            if fast else config.K_MAX_CRN

    ic_model, monitors, ucls = load_checkpoint(checkpoint)
    if skip_lstm and "lstm_ae" in monitors:
        monitors = {k: v for k, v in monitors.items() if k != "lstm_ae"}
        ucls     = {k: v for k, v in ucls.items()     if k != "lstm_ae"}
        print("  (LSTM-AE skipped — run with --lstm-only to add later)")
    rng = np.random.default_rng(seed or config.SEED + 1)

    print(f"\n{'='*60}")
    print(f"DyPPCA Phase II ARL experiment  (CRN)")
    print(f"  B_crn={B_crn_run}  K_max_crn={K_max_crn}  Cases: {cases}")
    print(f"{'='*60}")

    for case in cases:
        locations = config.LOCATIONS[case]
        for i, loc in enumerate(locations):
            loc_label = f"loc{i+1}"
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
                loc        = loc,
                loc_label  = loc_label,
            )

    print(f"\n{'='*60}")
    print(f"Done.  Results → {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase II ARL experiment (CRN)")
    parser.add_argument("--cases",     nargs="+", default=None)
    parser.add_argument("--B",         type=int,  default=None,
                        help=f"OC sequences (default: {config.B_CRN})")
    parser.add_argument("--fast",      action="store_true",
                        help="Debug: B=50, K_max=100")
    parser.add_argument("--no-lstm",   action="store_true",
                        help="Skip LSTM-AE (run all other methods only)")
    parser.add_argument("--lstm-only", action="store_true",
                        help="Run LSTM-AE only and merge into existing CSVs")
    parser.add_argument("--seed",      type=int,  default=None)
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    if args.lstm_only:
        run_lstm_only(cases=args.cases, B_crn=args.B, fast=args.fast,
                      seed=args.seed, checkpoint=args.checkpoint)
    else:
        run_all(cases=args.cases, B_crn=args.B, fast=args.fast,
                seed=args.seed, checkpoint=args.checkpoint,
                skip_lstm=args.no_lstm)

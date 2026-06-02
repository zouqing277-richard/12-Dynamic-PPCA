"""
run_diagnostic.py  —  Section 5.4 Diagnostic Performance Experiment.

Diagnostic rule
───────────────
Detection  : t_k = t1+t2+t3+t4 > h  (overall control limit from checkpoint)

Diagnosis at alarm time τ: compute empirical upper-tail p-values against
a pre-generated IC reference distribution (D windows):

    p_j = (1 + #{t_j^IC >= t_{jτ}}) / (D+1),  j=1,2,3,4

    j* = argmin_j p_j   (always exactly one diagnosis, no inconclusive)

Matching percentage (conditional on alarm):
    MP = #{τ < K_max, j* = j_true} / #{τ < K_max} × 100%

Design
──────
All 3 locations per case.
Shift magnitudes from config.DIAG_SHIFTS (independent from ARL table).

CLI:
  python run_diagnostic.py           # full run (B=10000, D=10000)
  python run_diagnostic.py --fast    # debug  (B=200,   D=500)
  python run_diagnostic.py --cases case1 case2
"""

import os, sys, time, argparse, pickle
import numpy as np
import pandas as pd

import config
from data_generator import (simulate_ic_batch_stateful,
                             simulate_oc_batch_stateful)
from methods.dyppca import DyPPCA

CHECKPOINT_FILE = "results/calibration/checkpoint.pkl"
SAVE_DIR        = "results/tables"

TRUE_COMPONENT = {
    "case1": 0,   # t1
    "case2": 1,   # t2
    "case3": 2,   # t3
    "case4": 2,   # t3
    "case5": 3,   # t4
}
COMP_NAMES = ["t1", "t2", "t3", "t4"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: IC reference distribution  (generated once, shared across all cases)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ic_reference(ic_model, dyppca_mon, D, n_window, rng):
    """
    Generate D IC windows → return (t1,t2,t3,t4) array of shape (D, 4).
    Used to compute empirical p-values at each alarm window.
    """
    WARMUP = 5 * n_window
    q      = ic_model["q"]
    Z = np.zeros((D, q))
    _, Z       = simulate_ic_batch_stateful(ic_model, WARMUP, Z, rng)
    X_last, Z  = simulate_ic_batch_stateful(ic_model, 1,      Z, rng)
    x_lag      = X_last
    X_new, _   = simulate_ic_batch_stateful(ic_model, n_window, Z, rng)
    X_win      = np.concatenate([x_lag, X_new], axis=1)
    stats      = dyppca_mon.monitor_window_batch(X_win)   # (D, 5)
    return stats[:, :4].copy()                            # (D, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: One (case, d, loc) diagnostic experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_one_diagnostic(case, d, loc, ic_model, dyppca_mon,
                       h_total, ic_ref, B, K_max, n_window, rng):
    """
    Parameters
    ----------
    h_total : float       overall control limit for t_total
    ic_ref  : (D, 4)      IC reference statistics t1..t4

    Returns
    -------
    mp, se, n_alarmed
    """
    D      = len(ic_ref)
    j_true = TRUE_COMPONENT[case]
    WARMUP = 5 * n_window
    q      = ic_model["q"]

    Z = np.zeros((B, q))
    _, Z       = simulate_ic_batch_stateful(ic_model, WARMUP, Z, rng)
    X_last, Z  = simulate_ic_batch_stateful(ic_model, 1,      Z, rng)
    x_lag      = X_last

    alarmed     = np.zeros(B, dtype=bool)
    diagnosed   = np.full(B, -1, dtype=int)   # j* for each alarmed sequence

    for k in range(K_max):
        X_new, Z = simulate_oc_batch_stateful(
            ic_model, n_window, case, d, Z, rng, loc=loc)
        X_win = np.concatenate([x_lag, X_new], axis=1)

        stats   = dyppca_mon.monitor_window_batch(X_win)  # (B, 5)
        t_total = stats[:, 4]
        t_comps = stats[:, :4]                            # (B, 4)

        new_alarms = (~alarmed) & (t_total > h_total)

        if new_alarms.any():
            alarm_stats = t_comps[new_alarms]             # (n_new, 4)
            n_new       = alarm_stats.shape[0]

            # Vectorised empirical p-values: p_j = (1+#{IC >= alarm}) / (D+1)
            # ic_ref: (D, 4)  alarm_stats: (n_new, 4)
            if n_new <= 512:
                counts = np.sum(
                    ic_ref[:, np.newaxis, :] >= alarm_stats[np.newaxis, :, :],
                    axis=0)                               # (n_new, 4)
            else:
                counts = np.empty((n_new, 4), dtype=np.int32)
                for j in range(4):
                    counts[:, j] = np.sum(
                        ic_ref[:, j, np.newaxis] >= alarm_stats[:, j],
                        axis=0)

            p_vals             = (1 + counts) / (D + 1)  # (n_new, 4)
            diagnosed[new_alarms] = np.argmin(p_vals, axis=1)
            alarmed |= new_alarms

        x_lag = X_new[:, -1:, :]
        if alarmed.all():
            break

    n_alarmed = int(alarmed.sum())
    if n_alarmed == 0:
        return 0.0, 0.0, 0

    n_match = int((diagnosed[alarmed] == j_true).sum())
    p_hat   = n_match / n_alarmed
    mp      = p_hat * 100.0
    se      = np.sqrt(p_hat * (1 - p_hat) / n_alarmed) * 100.0
    return mp, se, n_alarmed


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(cases=None, B=10_000, D=10_000, K_max=None,
                   fast=False, seed=config.SEED,
                   save_dir=SAVE_DIR, checkpoint=CHECKPOINT_FILE):

    if cases is None:
        cases = ["case1","case2","case3","case4","case5"]
    if fast:
        B, D, K_max = 200, 500, 200
    if K_max is None:
        K_max = config.K_MAX_CRN

    if not os.path.exists(checkpoint):
        print(f"ERROR: {checkpoint} not found")
        sys.exit(1)
    with open(checkpoint, "rb") as f:
        ck = pickle.load(f)

    ic_model = ck["ic_model"]
    h_total  = ck["ucls"]["dyppca"]["t_total"]
    dyppca_mon = DyPPCA.from_true_model(ic_model)
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("DyPPCA Diagnostic Experiment  (Section 5.4)")
    print(f"  B={B}  D={D}  K_max={K_max}  h_total={h_total:.4f}")
    print(f"  Rule: j* = argmin_j p_j  (empirical upper-tail p-value)")
    print(f"  All 3 locations per case")
    print("=" * 60)

    # IC reference — generated ONCE, shared across all cases
    print(f"\nGenerating IC reference  (D={D}) ...", flush=True)
    t0     = time.time()
    ic_ref = generate_ic_reference(dyppca_mon=dyppca_mon,
                                    ic_model=ic_model,
                                    D=D, n_window=config.N_WINDOW, rng=rng)
    print(f"  Done in {time.time()-t0:.1f}s\n", flush=True)

    all_rows = []

    for case in cases:
        j_true  = TRUE_COMPONENT[case]
        shifts  = config.DIAG_SHIFTS[case]
        tc_name = COMP_NAMES[j_true]

        for loc_idx, loc in enumerate(config.LOCATIONS[case]):
            loc_label = f"loc{loc_idx+1}"
            print(f"{'─'*60}")
            print(f"  {case.upper()} {loc_label}  loc={loc}  "
                  f"j_true={j_true} ({tc_name})  shifts={shifts}")
            print(f"{'─'*60}")

            for d in shifts:
                t0 = time.time()
                mp, se, n_al = run_one_diagnostic(
                    case=case, d=d, loc=loc,
                    ic_model=ic_model, dyppca_mon=dyppca_mon,
                    h_total=h_total, ic_ref=ic_ref,
                    B=B, K_max=K_max,
                    n_window=config.N_WINDOW, rng=rng)
                print(f"  d={d}:  MP={mp:6.2f}% (SE={se:.2f}%)  "
                      f"alarmed={n_al}/{B}  [{time.time()-t0:.0f}s]",
                      flush=True)
                all_rows.append({
                    "case":      case,
                    "loc_label": loc_label,
                    "loc":       str(loc),
                    "true_comp": tc_name,
                    "magnitude": d,
                    "MP":        round(mp, 2),
                    "SE":        round(se, 3),
                    "n_alarmed": n_al,
                    "n_total":   B,
                })

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(save_dir, "diagnostic_mp.csv"), index=False)
    print(f"\nSaved → {save_dir}/diagnostic_mp.csv")
    print_latex_table(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(df):
    print("\n" + "=" * 60)
    print("LaTeX table")
    print("=" * 60)

    case_labels = {"case1":"I","case2":"II","case3":"III",
                   "case4":"IV","case5":"V"}

    n_mag   = max(len(v) for v in config.DIAG_SHIFTS.values())
    mag_hdr = " & ".join(f"$d_{i+1}$" for i in range(n_mag))

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Diagnostic matching percentages (\%) across all shift"
          r" locations (standard errors in parentheses).}")
    print(r"\label{tab:diagnostic_mp}")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\begin{tabular}{c c c " + "c " * n_mag + r"}")
    print(r"\hline")
    print(f"Case & Location & True comp. & {mag_hdr} \\\\")
    print(r"\hline")

    for case in ["case1","case2","case3","case4","case5"]:
        sub = df[df["case"] == case]
        if sub.empty:
            continue
        tc = sub["true_comp"].iloc[0]
        for loc_label in sub["loc_label"].unique():
            sub_loc = sub[sub["loc_label"] == loc_label]
            loc_str = sub_loc["loc"].iloc[0]
            mps     = sub_loc["MP"].values
            ses     = sub_loc["SE"].values
            cells   = " & ".join(
                f"{mp:.1f} ({se:.1f})" for mp, se in zip(mps, ses))
            print(f"{case_labels[case]} ({loc_label}) & "
                  f"{loc_str} & ${tc}$ & {cells} \\\\")
        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases",  nargs="+", default=None)
    parser.add_argument("--B",      type=int,  default=10_000)
    parser.add_argument("--D",      type=int,  default=10_000)
    parser.add_argument("--K_max",  type=int,  default=None)
    parser.add_argument("--fast",   action="store_true",
                        help="Debug: B=200, D=500, K_max=200")
    parser.add_argument("--seed",   type=int,  default=config.SEED)
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    run_diagnostic(cases=args.cases, B=args.B, D=args.D, K_max=args.K_max,
                   fast=args.fast, seed=args.seed,
                   checkpoint=args.checkpoint)

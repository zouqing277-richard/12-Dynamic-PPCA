"""
run_diagnostic.py  —  Section 5.4 Diagnostic Performance Experiment.

For each OC case, the second representative location is used.
At each alarm window, the empirical diagnostic rule selects the component
with the smallest upper-tail p-value against an IC reference distribution.
Matching percentage (MP) measures how often the diagnosed component equals
the true structural source.

Method
------
1. Pre-generate D IC windows → IC reference distribution of (t1,t2,t3,t4).
2. For each (case, d):
   a. Simulate B OC sequences in parallel.
   b. At alarm window τ, compute empirical p-values:
        p_j = (1 + #{IC: t_j^IC ≥ t_j^alarm}) / (D+1)
   c. Diagnose: j* = argmin_j p_j
   d. MP = #{j*=j_true, alarmed} / #{alarmed} × 100%

CLI:
  python run_diagnostic.py                  # full run
  python run_diagnostic.py --fast           # debug (B=100, D=500)
  python run_diagnostic.py --cases case1    # one case only
"""

import os, sys, time, argparse, pickle
import numpy as np
import pandas as pd

import config
from data_generator import (build_ic_model, simulate_ic,
                             simulate_ic_batch_stateful,
                             simulate_oc_batch_stateful)
from methods.dyppca import DyPPCA

CHECKPOINT_FILE = "results/calibration/checkpoint.pkl"
SAVE_DIR        = "results/tables"

# ── Diagnostic settings ───────────────────────────────────────────────────────
# Second location (index 1) from LOCATIONS for each case
REP_LOC_IDX = 1

# True diagnostic component (0-indexed: t1=0, t2=1, t3=2, t4=3)
TRUE_COMPONENT = {
    "case1": 0,   # t1
    "case2": 1,   # t2
    "case3": 2,   # t3
    "case4": 2,   # t3
    "case5": 3,   # t4
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: IC reference distribution
# ─────────────────────────────────────────────────────────────────────────────

def generate_ic_reference(ic_model, dyppca_mon, D, n_window, rng):
    """
    Generate D IC windows and return (t1,t2,t3,t4) values: shape (D, 4).
    Used as the empirical IC reference for p-value computation.
    """
    WARMUP = 5 * n_window
    q      = ic_model["q"]

    Z = np.zeros((D, q))
    _, Z       = simulate_ic_batch_stateful(ic_model, WARMUP, Z, rng)
    X_last, Z  = simulate_ic_batch_stateful(ic_model, 1,      Z, rng)
    x_lag      = X_last
    X_new, _   = simulate_ic_batch_stateful(ic_model, n_window, Z, rng)
    X_win      = np.concatenate([x_lag, X_new], axis=1)   # (D, n+1, p)

    stats = dyppca_mon.monitor_window_batch(X_win)         # (D, 5)
    return stats[:, :4].copy()                             # (D, 4)  t1..t4


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: One (case, d) diagnostic experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_one_diagnostic(case, d, loc, ic_model, dyppca_mon, h_total,
                       ic_ref, B, K_max, n_window, rng):
    """
    Returns
    -------
    mp          : float  matching percentage (conditional on alarm)
    n_alarmed   : int    number of sequences that alarmed before K_max
    """
    D      = len(ic_ref)
    j_true = TRUE_COMPONENT[case]
    WARMUP = 5 * n_window
    q      = ic_model["q"]

    # Warm up B OC sequences from stationary IC, then switch to OC
    Z = np.zeros((B, q))
    _, Z       = simulate_ic_batch_stateful(ic_model, WARMUP, Z, rng)
    X_last, Z  = simulate_ic_batch_stateful(ic_model, 1,      Z, rng)
    x_lag      = X_last

    alarmed          = np.zeros(B, dtype=bool)
    alarm_diagnosed  = np.full(B, -1, dtype=int)

    for k in range(K_max):
        X_new, Z = simulate_oc_batch_stateful(
            ic_model, n_window, case, d, Z, rng, loc=loc)
        X_win = np.concatenate([x_lag, X_new], axis=1)    # (B, n+1, p)

        stats   = dyppca_mon.monitor_window_batch(X_win)  # (B, 5)
        t_total = stats[:, 4]
        t_comps = stats[:, :4]                            # (B, 4)

        # Sequences that alarm THIS window (first alarm only)
        new_alarms = (~alarmed) & (t_total > h_total)

        if new_alarms.any():
            alarm_stats = t_comps[new_alarms]             # (n_new, 4)
            n_new       = alarm_stats.shape[0]

            # Vectorised empirical upper-tail p-values
            # ic_ref: (D, 4)  alarm_stats: (n_new, 4)
            # counts[b, j] = #{IC: t_j^IC >= t_j^alarm_b}
            if n_new <= 512:
                # Fully vectorised: (D, n_new, 4)
                counts = np.sum(
                    ic_ref[:, np.newaxis, :] >= alarm_stats[np.newaxis, :, :],
                    axis=0)                               # (n_new, 4)
            else:
                # Chunk to avoid large memory when many alarms at once
                counts = np.empty((n_new, 4), dtype=np.int32)
                for j in range(4):
                    counts[:, j] = np.sum(
                        ic_ref[:, j, np.newaxis] >= alarm_stats[:, j],
                        axis=0)

            p_vals   = (1 + counts) / (D + 1)            # (n_new, 4)
            diagnosed = np.argmin(p_vals, axis=1)         # (n_new,)
            alarm_diagnosed[new_alarms] = diagnosed
            alarmed |= new_alarms

        x_lag = X_new[:, -1:, :]

        if alarmed.all():
            break

    # Matching percentage (conditional on alarm)
    alarmed_mask = alarm_diagnosed >= 0
    n_alarmed    = int(alarmed_mask.sum())
    if n_alarmed == 0:
        return 0.0, 0

    n_match = int(np.sum(alarm_diagnosed[alarmed_mask] == j_true))
    p_hat   = n_match / n_alarmed
    mp      = p_hat * 100.0
    se      = np.sqrt(p_hat * (1 - p_hat) / n_alarmed) * 100.0
    return mp, se, n_alarmed


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(cases=None, B=10_000, D=10_000, K_max=None,
                   fast=False, seed=config.SEED, save_dir=SAVE_DIR,
                   checkpoint=CHECKPOINT_FILE):

    if cases is None:
        cases = ["case1", "case2", "case3", "case4", "case5"]

    if fast:
        B, D, K_max = 100, 500, 200
    if K_max is None:
        K_max = config.K_MAX_CRN

    # Load checkpoint
    if not os.path.exists(checkpoint):
        print(f"ERROR: {checkpoint} not found — run run_calibration.py first")
        sys.exit(1)
    with open(checkpoint, "rb") as f:
        ck = pickle.load(f)

    ic_model = ck["ic_model"]
    ucls     = ck["ucls"]
    h_total  = ucls["dyppca"]["t_total"]

    # Always use oracle DyPPCA for diagnostic experiment
    dyppca_mon = DyPPCA.from_true_model(ic_model)

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("DyPPCA Diagnostic Experiment  (Section 5.4)")
    print(f"  B={B}  D={D}  K_max={K_max}  h_total={h_total:.4f}")
    print(f"  Representative: 2nd location per case (loc index {REP_LOC_IDX})")
    print("=" * 60)

    # Generate IC reference distribution ONCE (shared across all cases)
    print(f"\nGenerating IC reference  (D={D}) ...", flush=True)
    t0     = time.time()
    ic_ref = generate_ic_reference(ic_model, dyppca_mon, D,
                                    config.N_WINDOW, rng)
    print(f"  Done in {time.time()-t0:.1f}s  shape={ic_ref.shape}\n",
          flush=True)

    all_rows = []

    for case in cases:
        loc     = config.LOCATIONS[case][REP_LOC_IDX]
        j_true  = TRUE_COMPONENT[case]
        shifts  = config.SHIFTS[case]
        comp_name = ["t1", "t2", "t3", "t4"][j_true]

        print(f"{'─'*60}")
        print(f"  {case.upper()}  loc={loc}  j_true={j_true} ({comp_name})")
        print(f"{'─'*60}")

        rows = []
        for d in shifts:
            t0 = time.time()
            mp, se, n_alarm = run_one_diagnostic(
                case=case, d=d, loc=loc,
                ic_model=ic_model, dyppca_mon=dyppca_mon,
                h_total=h_total, ic_ref=ic_ref,
                B=B, K_max=K_max,
                n_window=config.N_WINDOW, rng=rng)
            elapsed = time.time() - t0
            print(f"  d={d}:  MP={mp:6.2f}%  SE={se:.2f}%  "
                  f"alarmed={n_alarm}/{B}  [{elapsed:.0f}s]", flush=True)
            rows.append({
                "case":      case,
                "loc":       str(loc),
                "true_comp": comp_name,
                "magnitude": d,
                "MP":        round(mp, 2),
                "SE":        round(se, 3),
                "n_alarmed": n_alarm,
                "n_total":   B,
            })
        all_rows.extend(rows)

    # Save full results
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(save_dir, "diagnostic_mp.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Print LaTeX-ready table
    print_latex_table(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_table(df):
    print("\n" + "=" * 60)
    print("LaTeX table (matching percentages)")
    print("=" * 60)

    cases_order = ["case1","case2","case3","case4","case5"]
    case_labels = {
        "case1": r"I",  "case2": r"II",
        "case3": r"III","case4": r"IV", "case5": r"V",
    }
    loc_labels = {
        "case1": r"$\delta_2$",
        "case2": r"$\tilde\delta_5$",
        "case3": r"$\Delta_{23}$",
        "case4": r"$\Delta_{23}$",
        "case5": r"$\Delta^e_{36}$",
    }

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Diagnostic matching percentages (\%) for representative shift locations.}")
    print(r"\label{tab:diagnostic_mp}")
    print(r"\renewcommand{\arraystretch}{1.2}")

    # Determine number of magnitudes
    n_mag = df.groupby("case").size().max() // 1
    mag_cols = " & ".join([f"$d_{i+1}$" for i in range(n_mag)])
    print(r"\begin{tabular}{c c c " + "c " * n_mag + r"}")
    print(r"\hline")
    print(f"Case & Location & True comp. & {mag_cols} \\\\")
    print(r"\hline")

    for case in cases_order:
        sub = df[df["case"] == case]
        if sub.empty:
            continue
        mps     = sub["MP"].values
        tc      = sub["true_comp"].iloc[0]
        ses     = sub["SE"].values if "SE" in sub.columns else [0]*len(mps)
        mp_cols = " & ".join(
            f"{v:.1f}" + r"\scriptsize{{({s:.2f})}}".format(s=s)
            for v, s in zip(mps, ses))
        print(f"{case_labels[case]} & {loc_labels[case]} & "
              f"${tc}$ & {mp_cols} \\\\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic MP experiment")
    parser.add_argument("--cases",  nargs="+", default=None)
    parser.add_argument("--B",      type=int,  default=10_000,
                        help="OC replications (default 10000)")
    parser.add_argument("--D",      type=int,  default=10_000,
                        help="IC reference size (default 10000)")
    parser.add_argument("--K_max",  type=int,  default=None)
    parser.add_argument("--fast",   action="store_true",
                        help="Debug: B=100, D=500, K_max=200")
    parser.add_argument("--seed",   type=int,  default=config.SEED)
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    args = parser.parse_args()

    run_diagnostic(
        cases      = args.cases,
        B          = args.B,
        D          = args.D,
        K_max      = args.K_max,
        fast       = args.fast,
        seed       = args.seed,
        checkpoint = args.checkpoint,
    )

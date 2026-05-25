"""
plot_results.py
Reads saved CSV results and produces:
  1. results/report.html  — self-contained HTML report (open in VSCode Preview
                            or any browser; no WebView/ServiceWorker needed)
  2. LaTeX table code printed to stdout

Usage:
    python plot_results.py
Then in VSCode: right-click report.html → "Open Preview"  (or Cmd+Shift+V)
"""

import os
import io
import base64
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

CASES = ["case1", "case2", "case3", "case4", "case5"]

CASE_LABELS = {
    "case1": "Case I – Latent mean shift",
    "case2": "Case II – Observation noise mean shift",
    "case3": "Case III – Latent AR matrix shift",
    "case4": "Case IV – Latent covariance shift",
    "case5": "Case V – Observation noise covariance shift",
}

CASE_DESC = {
    "case1": "δ = d · e₁,  E(z_t) = δ",
    "case2": "δ̃ = d · u_{e,1}  (residual subspace direction)",
    "case3": "B₁ = B₀ + d·E₁₁  (first latent autocorrelation ↑)",
    "case4": "Cov(z_t) → I_q + d·e₁e₁ᵀ,  B₀ fixed",
    "case5": "Cov(ε_t) = σ₀(1+d) I_p  (global noise increase)",
}

METHOD_LABELS = {
    "dyppca":       "DyPPCA (proposed)",
    "static_ppca":  "Static PPCA",
    "dpca":         "DPCA",
    "var_residual": "VAR-residual",
    "lstm_ae":      "LSTM-AE",
}

COLORS = {
    "dyppca":       "#d62728",   # red
    "static_ppca":  "#1f77b4",   # blue
    "dpca":         "#2ca02c",   # green
    "var_residual": "#ff7f0e",   # orange
    "lstm_ae":      "#9467bd",   # purple
}

DIAG_COLS = ["rho1", "rho2", "rho3", "rho4"]
DIAG_LABELS = ["ρ₁ latent mean", "ρ₂ noise mean",
               "ρ₃ latent dyn.", "ρ₄ noise cov."]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(table_dir="results/tables"):
    results = {}
    for case in CASES:
        arl_path = os.path.join(table_dir, f"{case}_arl.csv")
        if not os.path.exists(arl_path):
            continue
        results[case] = {
            "arl":  pd.read_csv(arl_path, index_col=0),
            "std":  pd.read_csv(os.path.join(table_dir, f"{case}_std.csv"),
                                 index_col=0),
            "diag": pd.read_csv(os.path.join(table_dir, f"{case}_diag.csv"),
                                 index_col=0),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers  (return base64 PNG strings for inline HTML embedding)
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


def make_arl_curve(case: str, df_arl: pd.DataFrame,
                   df_std: pd.DataFrame) -> str:
    methods = df_arl.columns.tolist()
    fig, ax = plt.subplots(figsize=(6, 3.8))

    for m in methods:
        if m not in df_arl.columns:
            continue
        y    = df_arl[m].values.astype(float)
        yerr = df_std[m].values.astype(float) if m in df_std.columns else None
        ax.errorbar(df_arl.index, y, yerr=yerr,
                    label=METHOD_LABELS.get(m, m),
                    color=COLORS.get(m, "grey"),
                    marker="o", linewidth=1.8,
                    capsize=3, elinewidth=0.8, alpha=0.85)

    ax.set_xlabel("Shift magnitude  d", fontsize=10)
    ax.set_ylabel("ARL₁  (windows)", fontsize=10)
    ax.set_title(f"{CASE_LABELS[case]}\n{CASE_DESC[case]}", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


def make_diagnostic_heatmap(results: dict) -> str:
    available = [c for c in CASES if c in results]
    n = len(available)
    rho_matrix = np.full((n, 4), np.nan)
    row_labels  = []

    for i, case in enumerate(available):
        df   = results[case]["diag"]
        row  = df.iloc[-1]          # largest magnitude
        vals = [row.get(c, np.nan) for c in DIAG_COLS]
        rho_matrix[i] = vals
        row_labels.append(f"Case {case[-1]}")

    fig, ax = plt.subplots(figsize=(6, 0.7 * n + 1.5))
    masked  = np.ma.masked_invalid(rho_matrix)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(DIAG_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Component ratio  ρⱼ", fraction=0.046)
    ax.set_title("DyPPCA diagnostic component ratios\n"
                 "(at largest shift magnitude per case)", fontsize=9)

    for i in range(n):
        for j in range(4):
            v = rho_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if v > 0.6 else "black")
    fig.tight_layout()
    return _fig_to_b64(fig)


def make_phase1_fig(phase1_results: dict) -> str:
    """Re-draw Phase I box plots from saved data (if available)."""
    if not phase1_results:
        return ""
    N_values = sorted(phase1_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    titles = ["EU  (subspace error)", "Eσ  (noise variance)", "EΣx  (covariance)"]
    keys   = ["EU", "Esigma", "ESigmax"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        data = [phase1_results[N][key] for N in N_values]
        bp = ax.boxplot(data, labels=N_values, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Phase I sample size  N", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Phase I Estimation Accuracy", fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table output (to stdout)
# ─────────────────────────────────────────────────────────────────────────────

def print_latex_tables(results: dict):
    for case, data in results.items():
        df      = data["arl"]
        methods = df.columns.tolist()
        cap     = CASE_LABELS[case]
        cols    = "r" + "r" * len(methods)
        header  = "Magnitude & " + " & ".join(
            METHOD_LABELS.get(m, m) for m in methods) + r" \\"

        print(f"\n% {'─'*55}")
        print(f"% {cap}")
        print(f"% {'─'*55}")
        print(r"\begin{table}[htbp]")
        print(f"  \\caption{{{cap}}}")
        print(r"  \centering")
        print(f"  \\begin{{tabular}}{{{cols}}}")
        print(r"    \toprule")
        print(f"    {header}")
        print(r"    \midrule")
        for mag, row in df.iterrows():
            vals = " & ".join(f"{v:.1f}" for v in row.values)
            print(f"    {mag:.2f} & {vals} \\\\")
        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML ARL table  (colour-coded: lowest ARL = green, highest = red)
# ─────────────────────────────────────────────────────────────────────────────

def _cell_style(val: float, row_min: float, row_max: float) -> str:
    """Background colour from green (best) to red (worst)."""
    if row_max == row_min:
        t = 0.5
    else:
        t = (val - row_min) / (row_max - row_min)
    # green → yellow → red
    r = int(60  + t * 195)
    g = int(180 - t * 100)
    b = int(60)
    return f"background-color: rgb({r},{g},{b}); color: #fff;"


def html_arl_table(df_arl: pd.DataFrame, df_std: pd.DataFrame) -> str:
    methods  = df_arl.columns.tolist()
    headers  = "".join(
        f"<th>{METHOD_LABELS.get(m, m)}</th>" for m in methods
    )
    rows_html = ""
    for mag, row in df_arl.iterrows():
        vals     = row.values.astype(float)
        row_min  = vals.min()
        row_max  = vals.max()
        cells    = ""
        for m, v in zip(methods, vals):
            style = _cell_style(v, row_min, row_max)
            std_v = df_std.loc[mag, m] if m in df_std.columns else 0.0
            cells += f'<td style="{style}">{v:.1f}<span class="std">±{std_v:.1f}</span></td>'
        rows_html += f"<tr><td><b>{mag}</b></td>{cells}</tr>\n"
    return f"""
<table>
  <thead><tr><th>d</th>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML diagnostic table
# ─────────────────────────────────────────────────────────────────────────────

def html_diag_table(df_diag: pd.DataFrame) -> str:
    headers = "".join(f"<th>{lb}</th>" for lb in DIAG_LABELS)
    rows_html = ""
    for mag, row in df_diag.iterrows():
        cells = ""
        for col in DIAG_COLS:
            v     = row.get(col, float("nan"))
            alpha = min(1.0, max(0.0, float(v))) if not np.isnan(v) else 0
            style = f"background-color: rgba(214,39,40,{alpha:.2f}); color: #fff;"
            txt   = f"{v:.2f}" if not np.isnan(v) else "–"
            cells += f'<td style="{style}">{txt}</td>'
        rows_html += f"<tr><td><b>{mag}</b></td>{cells}</tr>\n"
    return f"""
<table>
  <thead><tr><th>d</th>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main HTML assembly
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: -apple-system, sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 24px; color: #222; }
h1   { font-size: 1.5rem; border-bottom: 2px solid #d62728;
       padding-bottom: 6px; }
h2   { font-size: 1.15rem; margin-top: 2.5rem; color: #d62728; }
h3   { font-size: 1rem; margin-top: 1.4rem; color: #444; }
table { border-collapse: collapse; font-size: 0.85rem;
        margin: 10px 0 18px 0; }
th, td { border: 1px solid #ccc; padding: 5px 10px; text-align: right; }
thead  { background: #333; color: #fff; }
.std   { font-size: 0.72rem; opacity: 0.85; margin-left: 2px; }
.grid  { display: flex; flex-wrap: wrap; gap: 24px; }
.card  { background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px;
         padding: 14px; flex: 1 1 460px; }
img    { max-width: 100%; border-radius: 4px; }
.note  { font-size: 0.8rem; color: #666; font-style: italic; }
pre    { background: #f4f4f4; padding: 12px; border-radius: 4px;
         font-size: 0.8rem; overflow-x: auto; }
"""

def build_html(results: dict, phase1_b64: str = "") -> str:
    # ── Phase I section ───────────────────────────────────────────────────────
    phase1_html = ""
    if phase1_b64:
        phase1_html = f"""
<h2>1. Phase I – Estimation Accuracy</h2>
<img src="data:image/png;base64,{phase1_b64}" alt="Phase I estimation">
"""

    # ── Phase II section ──────────────────────────────────────────────────────
    case_sections = ""
    for case in CASES:
        if case not in results:
            continue
        data   = results[case]
        b64    = make_arl_curve(case, data["arl"], data["std"])
        at     = html_arl_table(data["arl"], data["std"])
        dt     = html_diag_table(data["diag"])
        cn     = case[-1]

        case_sections += f"""
<h2>Case {cn} – {CASE_LABELS[case]}</h2>
<p class="note">{CASE_DESC[case]}</p>
<div class="grid">
  <div class="card">
    <h3>ARL₁ curve</h3>
    <img src="data:image/png;base64,{b64}" alt="{case} ARL curve">
  </div>
  <div class="card">
    <h3>ARL₁ table  <span class="note">(green = fastest detection)</span></h3>
    {at}
    <h3 style="margin-top:16px">DyPPCA diagnostic ratios  ρⱼ at alarm</h3>
    {dt}
  </div>
</div>
"""

    # ── Heatmap ───────────────────────────────────────────────────────────────
    hm_b64 = make_diagnostic_heatmap(results) if results else ""
    heatmap_html = f"""
<h2>Diagnostic decomposition summary</h2>
<img src="data:image/png;base64,{hm_b64}"
     style="max-width:540px" alt="Diagnostic heatmap">
<p class="note">
  Each cell shows the mean fraction of the combined statistic
  t<sub>k</sub> = t₁+t₂+t₃+t₄ attributable to component j
  at the alarm window (largest shift magnitude per case).
</p>
""" if hm_b64 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DyPPCA Experiment Results</title>
<style>{CSS}</style>
</head>
<body>
<h1>DyPPCA Numerical Experiment Results</h1>
<p class="note">
  Generated by <code>plot_results.py</code> &nbsp;|&nbsp;
  Open in VSCode: right-click → <b>Open Preview</b>
  &nbsp;(Cmd+Shift+V)
</p>

{phase1_html}

<h2>2. Phase II – Monitoring Performance</h2>
<p class="note">
  ARL₀ = 200 &nbsp;|&nbsp; Window size n = 20 &nbsp;|&nbsp;
  All UCLs calibrated from 50 000 IC observations.
</p>
{case_sections}

{heatmap_html}

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = load_results()
    if not results:
        print("No CSV results found in results/tables/.")
        print("Run  python run_phase2.py  first.")
        return

    # Try to load Phase I data if run_phase1 saved it
    p1_path = "results/tables/phase1_results.npz"
    phase1_data = {}
    if os.path.exists(p1_path):
        npz = np.load(p1_path, allow_pickle=True)
        phase1_data = npz["results"].item()
    phase1_b64 = make_phase1_fig(phase1_data)

    html = build_html(results, phase1_b64)

    out_path = "results/report.html"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved → {out_path}")
    print("In VSCode: right-click the file → Open Preview  (or Cmd+Shift+V)")

    # Also print LaTeX tables to stdout
    print("\n" + "=" * 60)
    print("LaTeX tables")
    print("=" * 60)
    print_latex_tables(results)


if __name__ == "__main__":
    main()

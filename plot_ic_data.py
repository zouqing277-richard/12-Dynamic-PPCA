"""
plot_ic_data.py
───────────────
Visualise IC data generated from the DyPPCA model.

For one or more selected variables (x_j), shows:
  1. Time series plot
  2. Sample ACF  (autocorrelation function)
  3. Histogram + fitted normal density
  4. Lag-1 scatter plot  (x_t vs x_{t-1})

Output: results/figures/ic_data_report.html
        (open with VSCode right-click → Open Preview, or in any browser)

CLI examples:
  python plot_ic_data.py                    # plots variables 1, 5, 10, 15, 20
  python plot_ic_data.py --vars 1 2 3       # plots x1, x2, x3
  python plot_ic_data.py --vars 1 --T 500   # 500 time steps, only x1
  python plot_ic_data.py --vars all         # all 20 variables (overview only)
"""

import argparse, io, base64, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from data_generator import build_ic_model, simulate_ic


# ─────────────────────────────────────────────────────────────────────────────
# ACF helper
# ─────────────────────────────────────────────────────────────────────────────

def sample_acf(x, max_lag=40):
    """Return ACF values for lags 0..max_lag."""
    x  = x - x.mean()
    c0 = np.dot(x, x) / len(x)
    acf = np.array([np.dot(x[k:], x[:len(x)-k]) / (len(x) * c0)
                    for k in range(max_lag + 1)])
    return acf


# ─────────────────────────────────────────────────────────────────────────────
# Figure for ONE variable  →  base64 PNG
# ─────────────────────────────────────────────────────────────────────────────

def plot_one_variable(X, j, T_show=300, max_lag=40, color="#1f77b4"):
    """
    4-panel figure for variable x_{j+1}  (j is 0-indexed).

    Parameters
    ----------
    X      : (T, p) IC data array
    j      : 0-based column index
    T_show : how many time steps to show in the time series plot
    """
    x     = X[:, j]
    T     = len(x)
    t_end = min(T_show, T)
    acf   = sample_acf(x, max_lag)
    ci    = 1.96 / np.sqrt(T)         # 95% confidence band

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(f"IC Data  —  Variable $x_{{{j+1}}}$  "
                 f"(p={config.P}, q={config.Q})",
                 fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # ── Panel 1: Time series ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])           # span full row
    ax1.plot(np.arange(t_end), x[:t_end],
             color=color, linewidth=0.9, alpha=0.85)
    ax1.axhline(x.mean(), color="crimson", linewidth=1.2,
                linestyle="--", label=f"mean = {x.mean():.4f}")
    ax1.axhline(x.mean() + 2*x.std(), color="grey", linewidth=0.8,
                linestyle=":", alpha=0.7)
    ax1.axhline(x.mean() - 2*x.std(), color="grey", linewidth=0.8,
                linestyle=":", alpha=0.7, label="±2 std")
    ax1.set_xlabel("Time  $t$", fontsize=10)
    ax1.set_ylabel(f"$x_{{{j+1},t}}$", fontsize=10)
    ax1.set_title(f"Time Series  (first {t_end} of {T} steps)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25)

    # ── Panel 2: ACF ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    lags = np.arange(max_lag + 1)
    ax2.bar(lags, acf, color=color, alpha=0.75, width=0.7)
    ax2.axhline( ci, color="crimson", linewidth=1.0, linestyle="--",
                label=f"95% CI  (±{ci:.3f})")
    ax2.axhline(-ci, color="crimson", linewidth=1.0, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("Lag", fontsize=10)
    ax2.set_ylabel("ACF", fontsize=10)
    ax2.set_title("Sample ACF", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylim(-0.35, 1.05)
    ax2.grid(axis="y", alpha=0.25)

    # ── Panel 3: Histogram ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    counts, bins, _ = ax3.hist(x, bins=40, density=True,
                               color=color, alpha=0.65,
                               edgecolor="white", linewidth=0.4)
    # Fitted normal
    mu, sigma = x.mean(), x.std()
    xs = np.linspace(bins[0], bins[-1], 300)
    ax3.plot(xs, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((xs-mu)/sigma)**2),
             color="crimson", linewidth=1.8, label=f"N({mu:.3f}, {sigma:.3f}²)")
    ax3.set_xlabel(f"$x_{{{j+1}}}$", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Histogram + Fitted Normal", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ─────────────────────────────────────────────────────────────────────────────
# Overview heatmap: all variables
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_overview(X, T_show=200):
    """
    Compact heatmap: rows = variables, columns = time steps.
    Useful for seeing the overall correlation structure.
    """
    T_end = min(T_show, X.shape[0])
    data  = X[:T_end].T               # (p, T_end)

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                   vmin=-3*X.std(), vmax=3*X.std())
    ax.set_xlabel("Time  $t$", fontsize=10)
    ax.set_ylabel("Variable index", fontsize=10)
    ax.set_yticks(np.arange(config.P))
    ax.set_yticklabels([f"$x_{{{j+1}}}$" for j in range(config.P)],
                       fontsize=7)
    ax.set_title(f"All {config.P} IC Variables  (first {T_end} steps)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="Value (standardised scale)")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_corr_matrix(X):
    C   = np.corrcoef(X.T)             # (p, p)
    fig, ax = plt.subplots(figsize=(7, 6))
    im  = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(config.P))
    ax.set_yticks(np.arange(config.P))
    ax.set_xticklabels([f"{j+1}" for j in range(config.P)], fontsize=7)
    ax.set_yticklabels([f"{j+1}" for j in range(config.P)], fontsize=7)
    ax.set_title(f"Sample Correlation Matrix  (p={config.P})", fontsize=11)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ─────────────────────────────────────────────────────────────────────────────
# Summary stats table  (HTML)
# ─────────────────────────────────────────────────────────────────────────────

def summary_table_html(X, var_indices):
    """HTML table: mean, std, min, max, lag-1 ACF for selected variables."""
    header = ("<tr><th>Variable</th><th>Mean</th><th>Std</th>"
              "<th>Min</th><th>Max</th><th>ACF(lag=1)</th></tr>")
    rows   = ""
    for j in var_indices:
        x    = X[:, j]
        acf1 = sample_acf(x, 1)[1]
        rows += (f"<tr><td><b>x<sub>{j+1}</sub></b></td>"
                 f"<td>{x.mean():.4f}</td>"
                 f"<td>{x.std():.4f}</td>"
                 f"<td>{x.min():.4f}</td>"
                 f"<td>{x.max():.4f}</td>"
                 f"<td>{acf1:.4f}</td></tr>")
    return f"<table><thead>{header}</thead><tbody>{rows}</tbody></table>"


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
body { font-family: -apple-system, sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 24px; color: #222; }
h1   { font-size: 1.4rem; border-bottom: 2px solid #1f77b4;
       padding-bottom: 6px; }
h2   { font-size: 1.1rem; color: #1f77b4; margin-top: 2rem; }
p.note { font-size: .83rem; color: #555; font-style: italic; }
img  { max-width: 100%; border-radius: 4px; margin: 8px 0; }
table { border-collapse: collapse; font-size: .86rem; margin: 10px 0; }
th, td { border: 1px solid #ccc; padding: 5px 12px; text-align: right; }
thead  { background: #333; color: #fff; }
td:first-child { text-align: left; }
.param { background: #f4f4f4; padding: 10px 16px; border-radius: 4px;
         font-size: .85rem; font-family: monospace; margin: 10px 0; }
"""

def build_html(X, var_indices, b64_vars, b64_overview, b64_corr, T):
    # Model parameters block
    param_block = (
        f"p = {config.P}  |  q = {config.Q}  |  σ₀ = {config.SIGMA0}<br>"
        f"Λ₀ = diag{tuple(config.LAMBDA0)}<br>"
        f"B₀ =  {config.B0.tolist()}<br>"
        f"T (generated) = {T}"
    )

    stats_tbl = summary_table_html(X, var_indices)

    # Per-variable sections
    var_sections = ""
    for idx, (j, b64) in enumerate(zip(var_indices, b64_vars)):
        var_sections += f"""
<h2>Variable  x<sub>{j+1}</sub>  &nbsp;(column index {j})</h2>
<img src="data:image/png;base64,{b64}" alt="x{j+1}">
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>IC Data Visualisation</title>
<style>{CSS}</style>
</head>
<body>
<h1>IC Data Visualisation  —  DyPPCA Model</h1>
<p class="note">Open in VSCode: right-click → <b>Open Preview</b></p>

<h2>Model Parameters</h2>
<div class="param">{param_block}</div>

<h2>Summary Statistics  (selected variables)</h2>
{stats_tbl}

<h2>All Variables Overview (heatmap)</h2>
<img src="data:image/png;base64,{b64_overview}" alt="overview">
<p class="note">
  Rows = variables x₁ … x₂₀, columns = time steps.
  Colour = value; red/blue = positive/negative deviation.
</p>

<h2>Sample Correlation Matrix</h2>
<img src="data:image/png;base64,{b64_corr}" alt="correlation">
<p class="note">
  Non-zero off-diagonal entries reflect shared latent structure
  induced by the q={config.Q} latent components.
</p>

{var_sections}
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise IC data from the DyPPCA model"
    )
    parser.add_argument(
        "--vars", nargs="+", default=None, metavar="J",
        help=("Variable indices to plot in detail (1-based).\n"
              "Examples:  --vars 1 5 10   or   --vars all\n"
              "Default: 1 5 10 15 20")
    )
    parser.add_argument("--T",    type=int, default=1000,
                        help="Number of IC observations to generate (default: 1000)")
    parser.add_argument("--show", type=int, default=300,
                        help="Time steps shown in the time series panel (default: 300)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out",  default="results/figures/ic_data_report.html",
                        help="Output HTML path")
    args = parser.parse_args()

    # Parse variable indices
    if args.vars is None:
        # Default: 5 representative variables spread across p=20
        var_indices = [0, 4, 9, 14, 19]          # 1, 5, 10, 15, 20
    elif args.vars == ["all"]:
        var_indices = list(range(config.P))
    else:
        var_indices = [int(v) - 1 for v in args.vars]   # convert to 0-based
        # Clamp to valid range
        var_indices = [j for j in var_indices if 0 <= j < config.P]
        if not var_indices:
            print(f"No valid variable indices. Choose 1–{config.P}.")
            return

    print(f"Generating {args.T} IC observations ...", flush=True)
    rng      = np.random.default_rng(args.seed)
    ic_model = build_ic_model(config.P, config.Q, config.SIGMA0,
                              config.LAMBDA0, config.B0, seed=args.seed)
    X        = simulate_ic(ic_model, args.T, rng)

    print(f"Plotting variables: {[j+1 for j in var_indices]} ...", flush=True)
    b64_vars     = [plot_one_variable(X, j, T_show=args.show) for j in var_indices]
    b64_overview = plot_all_overview(X, T_show=args.show)
    b64_corr     = plot_corr_matrix(X)

    html = build_html(X, var_indices, b64_vars,
                      b64_overview, b64_corr, args.T)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport saved → {args.out}")
    print("VSCode: right-click the file → Open Preview")


if __name__ == "__main__":
    main()

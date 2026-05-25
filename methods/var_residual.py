"""
methods/var_residual.py
VAR(1) residual monitoring baseline  (Jarrett & Pan 2007).

Fit:  x_t = c + Φ x_{t-1} + e_t  via OLS (pure numpy, ~10 ms for p=20).
      VARMA is intentionally dropped: at p=20 statsmodels VARMAX requires
      ~90 s per fit and never converges — empirically verified.

Phase II statistics (sliding window of size n):
  T2_resid  — Hotelling T² on window residual mean  (mean-shift)
  W_cov     — trace statistic on window residual covariance  (cov-shift)

Alarm rule: OR (either statistic exceeds its UCL).
Both UCLs are calibrated jointly to achieve target FAR = 1/ARL₀.
"""

import numpy as np


class VARResidual:
    """VAR(1) OLS residual monitor."""

    def __init__(self):
        self.c         = None   # (p,)   intercept
        self.Phi       = None   # (p, p) transition matrix
        self.mu_e      = None   # (p,)   IC residual mean  (≈ 0)
        self.Sig_e     = None   # (p, p) IC residual covariance
        self.Sig_e_inv = None   # (p, p) precision matrix

    # ─────────────────────────────────────────────────────────────────────
    # Phase I
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "VARResidual":
        """
        Fit VAR(1) via OLS.

        Parameters
        ----------
        X : (N+1, p)  Phase I data (row 0 is the lag-initialisation row).
        """
        X1 = X[1:]      # x_t,   shape (N, p)
        X2 = X[:-1]     # x_{t-1}, shape (N, p)
        N, p = X1.shape

        # Design matrix: [1, x_{t-1}]  →  OLS for [c, Φᵀ]
        design = np.hstack([np.ones((N, 1)), X2])       # (N, p+1)
        coef   = np.linalg.lstsq(design, X1, rcond=None)[0]  # (p+1, p)
        self.c   = coef[0]        # (p,)
        self.Phi = coef[1:].T     # (p, p)

        # Spectral-radius check — warn if estimated Φ̂ is explosive
        rho = max(abs(np.linalg.eigvals(self.Phi)))
        if rho >= 1.0:
            import warnings
            warnings.warn(
                f"VAR(1): spectral radius of Φ̂ = {rho:.4f} ≥ 1. "
                "Estimated process may be non-stationary.",
                RuntimeWarning, stacklevel=2,
            )

        # IC residuals and their statistics
        E          = X1 - (self.c + X2 @ self.Phi.T)   # (N, p)
        self.mu_e  = E.mean(axis=0)                      # (p,)
        D          = E - self.mu_e
        Sig        = D.T @ D / N
        # Tikhonov regularisation for numerical stability
        Sig       += 1e-8 * np.eye(p)
        self.Sig_e     = Sig
        self.Sig_e_inv = np.linalg.inv(Sig)
        return self

    def _residuals(self, X_win: np.ndarray) -> np.ndarray:
        """Raw VAR(1) residuals for a window (n+1, p) → (n, p)."""
        return X_win[1:] - (self.c + X_win[:-1] @ self.Phi.T)

    # ─────────────────────────────────────────────────────────────────────
    # Phase II
    # ─────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray):
        """
        Compute T²_resid and W_cov for one monitoring window.

        Parameters
        ----------
        X_win : (n+1, p)

        Returns
        -------
        T2_resid : float   subgroup-mean Hotelling T² (mean shift)
        W_cov    : float   trace(Σ⁻¹ Sₑ − I)² statistic (cov shift)
        """
        E = self._residuals(X_win)          # (n, p)
        n, p = E.shape

        # ── T²: test if window residual mean = IC mean ────────────────────
        e_bar    = E.mean(axis=0) - self.mu_e
        T2_resid = float(n * e_bar @ self.Sig_e_inv @ e_bar)

        # ── W_cov: test if window residual covariance = IC covariance ─────
        Dc    = E - self.mu_e
        Se    = Dc.T @ Dc / n                           # (p, p)
        Dev   = self.Sig_e_inv @ Se - np.eye(p)         # Σ⁻¹ Sₑ − I
        W_cov = float(n * np.trace(Dev @ Dev))

        return T2_resid, W_cov

    def monitor_sequence(self, X: np.ndarray, n: int) -> np.ndarray:
        """
        Slide window over X and collect statistics.

        Returns
        -------
        stats : (K, 2)  columns = [T2_resid, W_cov]
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 2))
        for k in range(K):
            start = k * n
            X_win = X[start : start + n + 1]
            stats[k] = self.monitor_window(X_win)
        return stats

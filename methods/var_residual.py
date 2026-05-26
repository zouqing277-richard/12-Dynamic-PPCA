"""
methods/var_residual.py
VAR(1) residual monitoring baseline  (Jarrett & Pan 2007).

Fit:  x_t = c + Φ x_{t-1} + e_t  via OLS.

Returns a SINGLE combined statistic:
    T = T²_resid + W_cov

where
    T²_resid  = n (ē_k − μ_ε)ᵀ Σ_ε⁻¹ (ē_k − μ_ε)    mean-shift Hotelling T²
    W_cov     ≈ (n/2) · ‖Σ_ε⁻¹ Sεk − I‖²_F            covariance-shift trace stat

This follows Tk,VAR = T²k,VAR + Wk,VAR (Section 4 of the paper).
One UCL is calibrated for this combined statistic.
"""

import numpy as np


class VARResidual:
    """VAR(1) OLS residual monitor — single combined statistic."""

    def __init__(self):
        self.c         = None   # (p,)   intercept
        self.Phi       = None   # (p, p) transition matrix
        self.mu_e      = None   # (p,)   IC residual mean  (≈ 0)
        self.Sig_e     = None   # (p, p) IC residual covariance
        self.Sig_e_inv = None   # (p, p) precision matrix

    # ──────────────────────────────────────────────────────────────────────────
    # Oracle constructor
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_true_model(cls, ic_model: dict) -> "VARResidual":
        """
        Build from TRUE IC model parameters.

        The DyPPCA-generated process x_t has:
          Σ_x     = A₀ A₀ᵀ + σ₀ I
          C₁      = A₀ B₀ A₀ᵀ  (lag-1 cross-cov)
          Φ_true  = C₁ Σ_x⁻¹
          Σ_e     = Σ_x − Φ Σ_x Φᵀ
        """
        A0     = ic_model["A0"]
        B0     = ic_model["B0"]
        sigma0 = ic_model["sigma0"]
        nu0    = ic_model["nu0"]
        p      = A0.shape[0]

        Sigma_x = A0 @ A0.T + sigma0 * np.eye(p)
        C1      = A0 @ B0 @ A0.T
        Phi     = C1 @ np.linalg.inv(Sigma_x)
        c       = nu0 - Phi @ nu0
        Sigma_e = Sigma_x - Phi @ Sigma_x @ Phi.T
        Sigma_e = (Sigma_e + Sigma_e.T) / 2 + 1e-8 * np.eye(p)

        m             = cls()
        m.c           = c
        m.Phi         = Phi
        m.mu_e        = np.zeros(p)
        m.Sig_e       = Sigma_e
        m.Sig_e_inv   = np.linalg.inv(Sigma_e)
        return m

    # ──────────────────────────────────────────────────────────────────────────
    # Phase I
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "VARResidual":
        """
        Fit VAR(1) via OLS.

        Parameters
        ----------
        X : (N+1, p)  Phase I data.
        """
        X1 = X[1:]     # x_t,   shape (N, p)
        X2 = X[:-1]    # x_{t-1}, shape (N, p)
        N, p = X1.shape

        design = np.hstack([np.ones((N, 1)), X2])
        coef   = np.linalg.lstsq(design, X1, rcond=None)[0]
        self.c   = coef[0]
        self.Phi = coef[1:].T

        rho = max(abs(np.linalg.eigvals(self.Phi)))
        if rho >= 1.0:
            import warnings
            warnings.warn(
                f"VAR(1): spectral radius of Φ̂ = {rho:.4f} ≥ 1. "
                "Estimated process may be non-stationary.",
                RuntimeWarning, stacklevel=2,
            )

        E          = X1 - (self.c + X2 @ self.Phi.T)
        self.mu_e  = E.mean(axis=0)
        D          = E - self.mu_e
        Sig        = D.T @ D / N + 1e-8 * np.eye(p)
        self.Sig_e     = Sig
        self.Sig_e_inv = np.linalg.inv(Sig)
        return self

    def _residuals(self, X_win: np.ndarray) -> np.ndarray:
        """Raw VAR(1) residuals for a window (n+1, p) → (n, p)."""
        return X_win[1:] - (self.c + X_win[:-1] @ self.Phi.T)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase II  —  single combined statistic
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray) -> float:
        """
        Compute the combined VAR-residual monitoring statistic.

            T = T²_resid + W_cov

        Parameters
        ----------
        X_win : (n+1, p)

        Returns
        -------
        T_combined : float
        """
        E = self._residuals(X_win)          # (n, p)
        n, p = E.shape

        # T²: mean-shift component
        e_bar    = E.mean(axis=0) - self.mu_e
        T2_resid = float(n * e_bar @ self.Sig_e_inv @ e_bar)

        # W_cov: covariance-shift component (second-order approximation)
        Dc    = E - self.mu_e
        Se    = Dc.T @ Dc / n
        Dev   = self.Sig_e_inv @ Se - np.eye(p)
        W_cov = float(0.5 * n * np.trace(Dev @ Dev))

        return T2_resid + W_cov

    def monitor_sequence(self, X: np.ndarray, n: int) -> np.ndarray:
        """
        Sliding-window monitoring.

        Returns
        -------
        stats : (K, 1)  — column 0 is T_combined.
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 1))
        for k in range(K):
            start = k * n
            stats[k, 0] = self.monitor_window(X[start : start + n + 1])
        return stats

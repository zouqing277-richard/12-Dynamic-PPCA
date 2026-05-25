"""
methods/static_ppca.py
Static PPCA monitoring baseline.

Mean monitoring  :  W = T² + (1/σ)·Q   (combined LRT statistic)
Covariance monitoring:
    R1  – latent covariance shift  (score covariance chart)
    R2  – residual noise shift     (SPE variance chart)
    R   = R1 + R2  (unknown source)

All statistics are computed on a sliding window of size n.
"""

import numpy as np
from scipy.linalg import eigh, inv


class StaticPPCA:
    """
    Static PPCA monitor (ignores temporal dependence).

    Parameters
    ----------
    q : int   number of latent components.
    """

    def __init__(self, q: int):
        self.q = q
        # Phase I
        self.nu0     = None   # (p,)
        self.U       = None   # (p, q)
        self.Ue      = None   # (p, p-q)
        self.Lambda0 = None   # (q,)
        self.sigma0  = None   # float

    # ──────────────────────────────────────────────────────────────────────────
    # Phase I
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "StaticPPCA":
        """
        Fit static PPCA from Phase I data.

        Parameters
        ----------
        X : array (N, p) or (N+1, p)
            Uses only X[1:] (current observations) for consistency with DyPPCA.
        """
        # Use same slice convention as DyPPCA so Phase I sizes match
        Xc = X[1:] if X.shape[0] > 1 else X
        N, p = Xc.shape
        q = self.q

        nu0 = Xc.mean(axis=0)
        D   = Xc - nu0
        S   = D.T @ D / N          # (p, p)

        eigvals, eigvecs = eigh(S)
        idx     = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.nu0     = nu0
        self.U       = eigvecs[:, :q]
        self.Ue      = eigvecs[:, q:]
        self.Lambda0 = eigvals[:q]
        self.sigma0  = eigvals[q:].mean()
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Phase II statistics
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray):
        """
        Compute W, R1, R2, R for a window.

        Parameters
        ----------
        X_win : array (n+1, p) — only X_win[1:] (n rows) are used.

        Returns
        -------
        W  : float  combined mean statistic
        R1 : float  latent covariance chart
        R2 : float  residual noise chart
        R  : float  R1 + R2
        """
        X = X_win[1:]              # (n, p)
        n = X.shape[0]

        U      = self.U
        Ue     = self.Ue
        L0     = self.Lambda0      # (q,)
        sigma0 = self.sigma0
        nu0    = self.nu0
        q      = self.q
        p      = U.shape[0]

        # ── Mean statistic W ──────────────────────────────────────────────────
        xbar = X.mean(axis=0)
        d    = xbar - nu0

        scores    = U.T @ d                                  # (q,)
        resid_mu  = d - U @ scores                          # residual in obs space
        T2 = n * np.sum(scores ** 2 / L0)
        Q  = n * np.sum(resid_mu ** 2)
        W  = float(T2 + Q / sigma0)

        # ── Covariance statistics R1, R2 ─────────────────────────────────────
        D  = X - xbar                                         # (n, p)
        Sk = D.T @ D / n                                      # (p, p)

        # Score covariance  Sₛ = UᵀSₖU  (should ≈ Λ₀ under IC)
        Ss   = U.T @ Sk @ U                                   # (q, q)
        Dev1 = Ss / L0[:, None] - np.eye(q)                  # (q, q)  element-wise Λ⁻¹ Sₛ - I
        # Properly: (Λ₀⁻¹ Sₛ - I)
        Dev1 = np.diag(1.0 / L0) @ Ss - np.eye(q)
        R1   = float(n * np.trace(Dev1 @ Dev1))

        # Residual variance  Sₑ = UₑᵀSₖUₑ  (should ≈ σ₀ I under IC)
        Se   = Ue.T @ Sk @ Ue                                 # (p-q, p-q)
        Dev2 = Se / sigma0 - np.eye(p - q)
        R2   = float(n * np.trace(Dev2 @ Dev2))

        R = R1 + R2
        return W, R1, R2, R

    def monitor_sequence(self, X: np.ndarray, n: int):
        """
        Apply sliding-window monitoring.

        Returns
        -------
        stats : array (K, 4) columns = [W, R1, R2, R]
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 4))
        for k in range(K):
            start = k * n
            end   = start + n + 1
            X_win = X[start:end]
            stats[k] = self.monitor_window(X_win)
        return stats

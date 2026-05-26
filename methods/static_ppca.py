"""
methods/static_ppca.py
Static PPCA monitoring baseline.

Returns a SINGLE combined statistic:
    T = T² + (1/σ₀)·Q + R1 + R2

where
    T²   = n (x̄k − ν₀)ᵀ U Λ₀⁻¹ Uᵀ (x̄k − ν₀)   latent mean
    Q    = n (x̄k − ν₀)ᵀ Ũₑ Ũₑᵀ (x̄k − ν₀)         residual mean
    R1   = n/2 · ‖Λ₀⁻¹ UᵀSₖU − Iq‖²_F              latent cov
    R2   = n/2 · ‖(1/σ₀) ŨₑᵀSₖŨₑ − I_{p−q}‖²_F    noise cov

This follows Tk,PPCA = T²k,PPCA + (1/σ₀)Qk,PPCA + R1k,PPCA + R2k,PPCA
(Section 4 of the paper).  One UCL is calibrated for this combined statistic.
"""

import numpy as np
from scipy.linalg import eigh


class StaticPPCA:
    """
    Static PPCA monitor (ignores temporal dependence).

    Parameters
    ----------
    q : int   number of latent components.
    """

    def __init__(self, q: int):
        self.q       = q
        self.nu0     = None   # (p,)
        self.U       = None   # (p, q)
        self.Ue      = None   # (p, p-q)
        self.Lambda0 = None   # (q,)
        self.sigma0  = None   # float

    # ──────────────────────────────────────────────────────────────────────────
    # Oracle constructor
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_true_model(cls, ic_model: dict) -> "StaticPPCA":
        """Build from TRUE IC model parameters (no Phase I estimation error)."""
        m         = cls(q=len(ic_model["Lambda0"]))
        m.nu0     = ic_model["nu0"].copy()
        m.U       = ic_model["U0"].copy()
        m.Ue      = ic_model["Ue"].copy()
        m.Lambda0 = ic_model["Lambda0"].copy()
        m.sigma0  = float(ic_model["sigma0"])
        return m

    # ──────────────────────────────────────────────────────────────────────────
    # Phase I
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "StaticPPCA":
        """
        Fit static PPCA from Phase I data.

        Parameters
        ----------
        X : array (N+1, p)  — uses X[1:] for consistency with DyPPCA.
        """
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
    # Phase II  —  single combined statistic
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray) -> float:
        """
        Compute the combined PPCA monitoring statistic for one window.

            T = T² + (1/σ₀)·Q + R1 + R2

        Parameters
        ----------
        X_win : array (n+1, p)  — only X_win[1:] (n rows) are used.

        Returns
        -------
        T_combined : float
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

        # ── Mean components ───────────────────────────────────────────────────
        xbar   = X.mean(axis=0)
        d      = xbar - nu0                                  # (p,)

        T2 = float(n * (U.T @ d) @ np.diag(1.0 / L0) @ (U.T @ d))
        Q  = float(n * (Ue.T @ d) @ (Ue.T @ d))

        # ── Covariance components ─────────────────────────────────────────────
        D  = X - xbar                                        # (n, p)
        Sk = D.T @ D / n                                     # (p, p)

        # R1: latent subspace covariance
        Ss   = U.T @ Sk @ U                                  # (q, q)
        Dev1 = np.diag(1.0 / L0) @ Ss - np.eye(q)           # Λ₀⁻¹ Sₛ − I
        R1   = float(0.5 * n * np.trace(Dev1 @ Dev1))

        # R2: residual noise covariance
        Se   = Ue.T @ Sk @ Ue                                # (p-q, p-q)
        Dev2 = Se / sigma0 - np.eye(p - q)
        R2   = float(0.5 * n * np.trace(Dev2 @ Dev2))

        return T2 + Q / sigma0 + R1 + R2

    def monitor_sequence(self, X: np.ndarray, n: int) -> np.ndarray:
        """
        Sliding-window monitoring.

        Returns
        -------
        stats : array (K, 1)  — column 0 is T_combined.
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 1))
        for k in range(K):
            start = k * n
            stats[k, 0] = self.monitor_window(X[start : start + n + 1])
        return stats

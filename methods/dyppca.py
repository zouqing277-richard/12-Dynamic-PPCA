"""
methods/dyppca.py
DyPPCA: Dynamic Probabilistic PCA monitoring framework.

Phase I  – Algorithm 1 in the paper (structured covariance estimation).
Phase II – Approximate LRT statistic  tk = t1k + t2k + t3k + t4k  (eq. 62).
"""

import numpy as np
from scipy.linalg import inv, eigh


class DyPPCA:
    """
    Dynamic Probabilistic PCA monitor.

    Parameters
    ----------
    q : int
        Number of retained latent components.
    """

    def __init__(self, q: int):
        self.q = q
        # Phase I estimates (set by fit)
        self.nu0     = None   # (p,)   process mean
        self.U       = None   # (p, q) latent loading subspace
        self.Ue      = None   # (p, p-q) residual complement
        self.Lambda0 = None   # (q,)   latent eigenvalues
        self.Gamma0  = None   # (q, q) off-diagonal IC block  Uᵀ Q U
        self.sigma0  = None   # float  residual noise variance

        # Derived matrices (precomputed after fit)
        self.Phi0     = None  # (2q, 2q)
        self.Phi0_inv = None  # (2q, 2q)
        self.Phi1     = None  # (q, q)
        self.Phi2     = None  # (q, q)
        self.Phi3     = None  # (q, q)
        self.M_mean   = None  # Φ₁ + Φ₂ + Φ₃ + Φ₃ᵀ  for t1

    # ──────────────────────────────────────────────────────────────────────────
    # Phase I Estimation  (Algorithm 1)
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "DyPPCA":
        """
        Fit the DyPPCA model from Phase I observations.

        Parameters
        ----------
        X : array (N+1, p)
            Phase I data. Row 0 is the "extra" lagged observation.

        Returns
        -------
        self
        """
        N1, p = X.shape
        N = N1 - 1
        q = self.q

        # ── Step 1: construct augmented observation blocks ───────────────────
        X1 = X[1:]    # xt,   shape (N, p)
        X2 = X[:-1]   # xt-1, shape (N, p)

        xbar1 = X1.mean(axis=0)
        xbar2 = X2.mean(axis=0)

        D1 = X1 - xbar1
        D2 = X2 - xbar2

        S1 = D1.T @ D1 / N        # (p, p)
        S2 = D2.T @ D2 / N        # (p, p)
        Q  = D1.T @ D2 / N        # (p, p) lag-one cross-covariance

        # Stationarity-constrained estimates (eq. 14)
        xbar = (xbar1 + xbar2) / 2
        S    = (S1 + S2) / 2

        # ── Step 2: eigen-decomposition of S (Proposition 1) ─────────────────
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = eigh(S)
        idx      = np.argsort(eigvals)[::-1]   # descending
        eigvals  = eigvals[idx]
        eigvecs  = eigvecs[:, idx]

        U  = eigvecs[:, :q]         # (p, q) loading subspace
        Ue = eigvecs[:, q:]         # (p, p-q) residual complement

        # ── Step 3: estimate σ, Λ, Γ  (eqs. 17–18) ──────────────────────────
        sigma   = eigvals[q:].mean()                 # eq. 17 (scalar)
        Lambda0 = eigvals[:q]                        # (q,) latent eigenvalues
        Gamma0  = U.T @ Q @ U                        # (q, q)  eq. 18 reduces to this

        # Store
        self.nu0     = xbar
        self.U       = U
        self.Ue      = Ue
        self.Lambda0 = Lambda0
        self.Gamma0  = Gamma0
        self.sigma0  = sigma

        # ── Step 4: precompute monitoring matrices ───────────────────────────
        self._precompute()
        return self


    # ──────────────────────────────────────────────────────────────────────────
    # Oracle constructor  (uses true parameters, no Phase I data needed)
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_true_model(cls, ic_model: dict) -> "DyPPCA":
        """
        Build a DyPPCA monitor directly from the TRUE IC model parameters.

        No Phase I data is needed. Use this to measure the theoretical
        upper-bound performance of the monitoring statistic, removing
        Phase I estimation error from the ARL comparison.

        Parameters
        ----------
        ic_model : dict returned by data_generator.build_ic_model()
        """
        U0     = ic_model["U0"]       # (p, q)  true loading subspace
        Ue     = ic_model["Ue"]       # (p, p-q)
        A0     = ic_model["A0"]       # (p, q)
        B0     = ic_model["B0"]       # (q, q)
        sigma0 = ic_model["sigma0"]   # scalar
        Lambda0= ic_model["Lambda0"]  # (q,)

        # True Gamma0 = U0^T Cov(x_t, x_{t-1}) U0
        #             = U0^T A0 B0 A0^T U0
        Gamma0 = U0.T @ A0 @ B0 @ A0.T @ U0   # (q, q)

        m          = cls(q=len(Lambda0))
        m.nu0      = ic_model["nu0"].copy()
        m.U        = U0.copy()
        m.Ue       = Ue.copy()
        m.Lambda0  = Lambda0.copy()
        m.Gamma0   = Gamma0
        m.sigma0   = float(sigma0)
        m._precompute()
        return m
    def _precompute(self):
        """Precompute Φ₀, Φ₀⁻¹, and derived blocks (eqs. 34–35)."""
        L0  = np.diag(self.Lambda0)        # (q, q)
        G0  = self.Gamma0                  # (q, q)
        L0i = np.diag(1.0 / self.Lambda0)  # (q, q)

        # Φ₀ = [[Λ₀, Γ₀], [Γ₀ᵀ, Λ₀]]
        self.Phi0 = np.block([[L0, G0],
                               [G0.T, L0]])

        # Schur-complement blocks of Φ₀⁻¹  (eq. 35)
        Phi1 = inv(L0 - G0 @ L0i @ G0.T)       # (q, q)
        Phi2 = inv(L0 - G0.T @ L0i @ G0)        # (q, q)
        Phi3 = -Phi1 @ G0 @ L0i                 # (q, q)

        self.Phi1 = Phi1
        self.Phi2 = Phi2
        self.Phi3 = Phi3

        self.Phi0_inv = np.block([[Phi1, Phi3],
                                   [Phi3.T, Phi2]])

        # Combined mean matrix for t1  (eq. 37)
        self.M_mean = Phi1 + Phi2 + Phi3 + Phi3.T

    # ──────────────────────────────────────────────────────────────────────────
    # Phase II Monitoring
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray):
        """
        Compute monitoring statistics for one window of observations.

        Parameters
        ----------
        X_win : array (n+1, p)
            Window of n observations plus one leading lag observation.
            Rows: [x_{(k-1)n}, x_{(k-1)n+1}, ..., x_{kn}]

        Returns
        -------
        t1, t2, t3, t4 : floats
            Diagnostic components.
        t_total : float
            Combined statistic  tk = t1 + t2 + t3 + t4.
        """
        n = len(X_win) - 1       # window size
        X1 = X_win[1:]           # xt   shape (n, p)
        X2 = X_win[:-1]          # xt-1 shape (n, p)

        xbar_k1 = X1.mean(axis=0)
        xbar_k2 = X2.mean(axis=0)
        xbar_k  = (xbar_k1 + xbar_k2) / 2

        D1  = X1 - xbar_k1
        D2  = X2 - xbar_k2
        Sk1 = D1.T @ D1 / n
        Sk2 = D2.T @ D2 / n
        Qk  = D1.T @ D2 / n
        Sk  = (Sk1 + Sk2) / 2

        U       = self.U
        Ue      = self.Ue
        sigma0  = self.sigma0
        nu0     = self.nu0
        q       = self.q
        p       = U.shape[0]

        d_mean = xbar_k - nu0    # (p,)

        # ── t1: latent mean shift  (eq. 37) ──────────────────────────────────
        v1 = U.T @ d_mean                  # (q,)
        t1 = float(n * v1 @ self.M_mean @ v1)

        # ── t2: residual mean shift  (eq. 38) ────────────────────────────────
        v2 = Ue.T @ d_mean                 # (p-q,)
        t2 = float((2 * n / sigma0) * v2 @ v2)

        # ── t3: latent dynamics / covariance shift  (eq. 39) ─────────────────
        # Mₖ = ΘᵀΣ̂ₖΘ  where Θ = diag(U,U)
        UtSkU = U.T @ Sk @ U               # (q, q)
        UtQkU = U.T @ Qk @ U               # (q, q)
        Mk = np.block([[UtSkU, UtQkU],
                        [UtQkU.T, UtSkU]])  # (2q, 2q)

        Dev3 = self.Phi0_inv @ Mk - np.eye(2 * q)   # (2q, 2q)
        t3   = float(0.5 * n * np.trace(Dev3 @ Dev3))

        # ── t4: obs noise covariance shift  (eq. 40) ─────────────────────────
        UetSkUe = Ue.T @ Sk @ Ue           # (p-q, p-q)
        UetQkUe = Ue.T @ Qk @ Ue           # (p-q, p-q)
        Dev4_S  = (1.0 / sigma0) * UetSkUe - np.eye(p - q)
        t4 = float(
            n * np.trace(Dev4_S @ Dev4_S)
            + (n / sigma0 ** 2) * np.trace(UetQkUe @ UetQkUe.T)
        )

        t_total = t1 + t2 + t3 + t4
        return t1, t2, t3, t4, t_total

    def monitor_sequence(self, X: np.ndarray, n: int):
        """
        Apply sliding-window monitoring to a sequence.

        Parameters
        ----------
        X : array (T, p)
            Full observation sequence (T ≥ n + 1).
        n : int
            Window size.

        Returns
        -------
        stats : array (K, 5)
            Columns: [t1, t2, t3, t4, t_total] for each of K = (T-1)//n windows.
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 5))
        for k in range(K):
            start = k * n
            end   = start + n + 1          # inclusive of lag
            X_win = X[start:end]
            t1, t2, t3, t4, tt = self.monitor_window(X_win)
            stats[k] = [t1, t2, t3, t4, tt]
        return stats

"""
methods/dpca.py
Dynamic PCA baseline  (Ku et al. 1995).

Lag-1 augmented vector  z_t = [x_t; x_{t-1}] ∈ R^{2p}.
PCA is fitted on Phase I z_t.  Components are selected by CPV threshold.
Phase II: sliding-window T² and Q statistics; alarm on either exceeding UCL.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class DPCA:
    """
    Dynamic PCA monitor.

    Parameters
    ----------
    cpv_threshold : float
        Cumulative proportion of variance for selecting number of components.
    lag : int
        Lag order for augmentation (default 1).
    """

    def __init__(self, cpv_threshold: float = 0.90, lag: int = 1):
        self.cpv = cpv_threshold
        self.lag = lag
        # Fitted attributes
        self.mean_z = None     # (2p,) or (lag*p + p,)  mean of augmented vector
        self.P      = None     # (2p, r) loading matrix (retained components)
        self.eigvals= None     # (r,) retained eigenvalues
        self.n_comp = None     # r
        self.scale  = None     # total variance (for Q normalisation)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase I
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "DPCA":
        """
        Fit DPCA on Phase I data.

        Parameters
        ----------
        X : array (N+1, p)
            Phase I observations (one extra row for lag).
        """
        Z = self._augment(X)          # (N, 2p)
        self.mean_z = Z.mean(axis=0)
        Zc = Z - self.mean_z #centered data (N, 2p)

        # Full covariance of augmented vector
        N, D = Zc.shape
        Cov = Zc.T @ Zc / N          # (2p, 2p)

        eigvals, eigvecs = np.linalg.eigh(Cov)
        idx     = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Select components by CPV
        total_var = eigvals.sum()
        cpv_curve = np.cumsum(eigvals) / total_var
        r = int(np.searchsorted(cpv_curve, self.cpv)) + 1
        r = max(r, 1)

        self.P       = eigvecs[:, :r]    # (2p, r)
        self.eigvals = eigvals[:r]       # (r,)
        self.n_comp  = r
        self.scale   = total_var
        return self

    def _augment(self, X: np.ndarray) -> np.ndarray:
        """Build lag-augmented matrix from (N+1, p) → (N, 2p)."""
        return np.hstack([X[1:], X[:-1]])

    # ──────────────────────────────────────────────────────────────────────────
    # Phase II
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray):
        """
        Compute T² and Q for a window.

        Parameters
        ----------
        X_win : array (n+1, p)

        Returns
        -------
        T2_mean : float   mean T² over window
        Q_mean  : float   mean Q (SPE) over window
        """
        Z   = self._augment(X_win)          # (n, 2p)
        Zc  = Z - self.mean_z               # (n, 2p)

        scores = Zc @ self.P                # (n, r)
        recon  = scores @ self.P.T          # (n, 2p)
        resid  = Zc - recon                 # (n, 2p)

        # T² per observation
        T2_vals = np.sum((scores ** 2) / self.eigvals, axis=1)   # (n,)
        # Q (SPE) per observation
        Q_vals  = np.sum(resid ** 2, axis=1)                       # (n,)

        return float(T2_vals.mean()), float(Q_vals.mean())

    def monitor_sequence(self, X: np.ndarray, n: int):
        """
        Apply sliding-window monitoring.

        Returns
        -------
        stats : array (K, 2)  columns = [T2, Q]
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 2))
        for k in range(K):
            start = k * n
            end   = start + n + 1
            X_win = X[start:end]
            stats[k] = self.monitor_window(X_win)
        return stats

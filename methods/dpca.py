"""
methods/dpca.py  — Dynamic PCA, T2 and Q statistics.
monitor_window_batch uses matmul for Accelerate/AMX on M3.
"""
import numpy as np


class DPCA:
    def __init__(self, cpv_threshold=0.90, lag=1):
        self.cpv=cpv_threshold; self.lag=lag
        self.mean_z=None; self.P=None; self.eigvals=None; self.n_comp=None

    def fit(self, X):
        Z=self._augment(X); self.mean_z=Z.mean(0); Zc=Z-self.mean_z
        N,D=Zc.shape; Cov=Zc.T@Zc/N
        eigvals,eigvecs=np.linalg.eigh(Cov)
        idx=np.argsort(eigvals)[::-1]; eigvals=eigvals[idx]; eigvecs=eigvecs[:,idx]
        total=eigvals.sum(); cpv=np.cumsum(eigvals)/total
        r=int(np.searchsorted(cpv,self.cpv))+1; r=max(r,1)
        self.P=eigvecs[:,:r]; self.eigvals=eigvals[:r]; self.n_comp=r; return self

    def _augment(self, X): return np.hstack([X[1:],X[:-1]])

    def monitor_window(self, X_win):
        Z=self._augment(X_win); Zc=Z-self.mean_z
        scores=Zc@self.P; recon=scores@self.P.T; resid=Zc-recon
        return float(np.mean(np.sum(scores**2/self.eigvals,1))), float(np.mean(np.sum(resid**2,1)))

    def monitor_window_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Batch T² and Q.
        Uses matmul for Accelerate/AMX on M3.

        Parameters
        ----------
        X_batch : (B, n+1, p)

        Returns
        -------
        stats : (B, 2)  columns = [T2, Q]
        """
        Z  = np.concatenate([X_batch[:, 1:, :], X_batch[:, :-1, :]], axis=2)
        Zc = Z - self.mean_z                              # (B, n, 2p)
        # scores: (B, n, r) via batched matmul
        scores = np.matmul(Zc, self.P)                    # (B, n, r)
        recon  = np.matmul(scores, self.P.T)              # (B, n, 2p)
        resid  = Zc - recon
        T2 = np.mean(np.sum(scores**2 / self.eigvals, axis=2), axis=1)
        Q  = np.mean(np.sum(resid**2,  axis=2), axis=1)
        return np.column_stack([T2, Q])

    def monitor_sequence(self, X, n):
        T=X.shape[0]; K=(T-1)//n; stats=np.empty((K,2))
        for k in range(K): stats[k]=self.monitor_window(X[k*n:k*n+n+1])
        return stats

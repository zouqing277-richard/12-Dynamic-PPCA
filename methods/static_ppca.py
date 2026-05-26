"""
methods/static_ppca.py  — Static PPCA, single combined statistic.
monitor_window_batch uses matmul for Accelerate/AMX on M3.
"""
import numpy as np
from scipy.linalg import eigh


class StaticPPCA:
    def __init__(self, q):
        self.q=q; self.nu0=None; self.U=None; self.Ue=None
        self.Lambda0=None; self.sigma0=None

    @classmethod
    def from_true_model(cls, ic_model):
        m=cls(q=len(ic_model["Lambda0"]))
        m.nu0=ic_model["nu0"].copy(); m.U=ic_model["U0"].copy()
        m.Ue=ic_model["Ue"].copy(); m.Lambda0=ic_model["Lambda0"].copy()
        m.sigma0=float(ic_model["sigma0"]); return m

    def fit(self, X):
        Xc=X[1:] if X.shape[0]>1 else X; N,p=Xc.shape; q=self.q
        nu0=Xc.mean(0); D=Xc-nu0; S=D.T@D/N
        eigvals,eigvecs=eigh(S)
        idx=np.argsort(eigvals)[::-1]; eigvals=eigvals[idx]; eigvecs=eigvecs[:,idx]
        self.nu0=nu0; self.U=eigvecs[:,:q]; self.Ue=eigvecs[:,q:]
        self.Lambda0=eigvals[:q]; self.sigma0=eigvals[q:].mean(); return self

    def monitor_window(self, X_win):
        X=X_win[1:]; n=X.shape[0]
        U=self.U; Ue=self.Ue; L0=self.Lambda0; sigma0=self.sigma0; nu0=self.nu0
        q=self.q; p=U.shape[0]
        xbar=X.mean(0); d=xbar-nu0
        T2=float(n*np.sum((U.T@d)**2/L0))
        Q=float(n*np.sum((Ue.T@d)**2))
        D=X-xbar; Sk=D.T@D/n
        Dev1=np.diag(1./L0)@(U.T@Sk@U)-np.eye(q)
        R1=float(0.5*n*np.trace(Dev1@Dev1))
        Se=Ue.T@Sk@Ue; Dev2=Se/sigma0-np.eye(p-q)
        R2=float(0.5*n*np.trace(Dev2@Dev2))
        return T2+Q/sigma0+R1+R2

    def monitor_window_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Batch combined T = T² + Q/σ₀ + R1 + R2.
        Uses matmul for Accelerate/AMX on M3.

        Parameters
        ----------
        X_batch : (B, n+1, p)

        Returns
        -------
        stats : (B, 1)
        """
        n_ = X_batch.shape[1] - 1
        X  = X_batch[:, 1:, :]              # (B, n, p)
        xb = X.mean(axis=1)                  # (B, p)
        d  = xb - self.nu0

        v1 = d @ self.U
        T2 = n_ * np.sum(v1 ** 2 / self.Lambda0, axis=1)
        v2 = d @ self.Ue
        Q  = n_ * np.sum(v2 ** 2, axis=1)

        D  = X - xb[:, None]                 # (B, n, p)
        Sk = np.matmul(D.transpose(0,2,1), D) / n_   # (B, p, p)  ← matmul

        qq = self.q; pq = self.Ue.shape[1]
        Ss   = np.einsum('pi,bpj->bij', self.U, Sk @ self.U)
        Dev1 = np.einsum('ij,bjk->bik', np.diag(1./self.Lambda0), Ss) - np.eye(qq)
        R1   = 0.5 * n_ * np.einsum('bij,bji->b', Dev1, Dev1)

        Se   = np.einsum('pi,bpj->bij', self.Ue, Sk @ self.Ue)
        Dev2 = Se / self.sigma0 - np.eye(pq)
        R2   = 0.5 * n_ * np.einsum('bij,bji->b', Dev2, Dev2)

        return (T2 + Q / self.sigma0 + R1 + R2).reshape(-1, 1)

    def monitor_sequence(self, X, n):
        T=X.shape[0]; K=(T-1)//n; stats=np.empty((K,1))
        for k in range(K): stats[k,0]=self.monitor_window(X[k*n:k*n+n+1])
        return stats

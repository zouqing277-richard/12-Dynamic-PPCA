"""
methods/var_residual.py  — VAR(1) residual monitor, single combined statistic.
monitor_window_batch uses matmul for Accelerate/AMX on M3.
"""
import numpy as np


class VARResidual:
    def __init__(self):
        self.c=None; self.Phi=None; self.mu_e=None
        self.Sig_e=None; self.Sig_e_inv=None

    @classmethod
    def from_true_model(cls, ic_model):
        A0=ic_model["A0"]; B0=ic_model["B0"]
        sigma0=ic_model["sigma0"]; nu0=ic_model["nu0"]; p=A0.shape[0]
        Sigma_x=A0@A0.T+sigma0*np.eye(p); C1=A0@B0@A0.T
        Phi=C1@np.linalg.inv(Sigma_x); c=nu0-Phi@nu0
        Sigma_e=Sigma_x-Phi@Sigma_x@Phi.T
        Sigma_e=(Sigma_e+Sigma_e.T)/2+1e-8*np.eye(p)
        m=cls(); m.c=c; m.Phi=Phi; m.mu_e=np.zeros(p)
        m.Sig_e=Sigma_e; m.Sig_e_inv=np.linalg.inv(Sigma_e); return m

    def fit(self, X):
        X1=X[1:]; X2=X[:-1]; N,p=X1.shape
        design=np.hstack([np.ones((N,1)),X2])
        coef=np.linalg.lstsq(design,X1,rcond=None)[0]
        self.c=coef[0]; self.Phi=coef[1:].T
        rho=max(abs(np.linalg.eigvals(self.Phi)))
        if rho>=1.0:
            import warnings
            warnings.warn(f"VAR(1): spectral radius {rho:.4f}>=1", RuntimeWarning, stacklevel=2)
        E=X1-(self.c+X2@self.Phi.T); self.mu_e=E.mean(0)
        D=E-self.mu_e; Sig=D.T@D/N+1e-8*np.eye(p)
        self.Sig_e=Sig; self.Sig_e_inv=np.linalg.inv(Sig); return self

    def monitor_window(self, X_win):
        E=X_win[1:]-(self.c+X_win[:-1]@self.Phi.T); n,p=E.shape
        e_bar=E.mean(0)-self.mu_e
        T2=float(n*e_bar@self.Sig_e_inv@e_bar)
        Dc=E-self.mu_e; Se=Dc.T@Dc/n
        Dev=self.Sig_e_inv@Se-np.eye(p)
        W=float(0.5*n*np.trace(Dev@Dev))
        return T2+W

    def monitor_window_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Batch combined T = T²_resid + W_cov.
        Uses matmul for Accelerate/AMX on M3.

        Parameters
        ----------
        X_batch : (B, n+1, p)

        Returns
        -------
        stats : (B, 1)
        """
        n_ = X_batch.shape[1] - 1
        p  = X_batch.shape[2]
        E  = X_batch[:, 1:, :] - (self.c + X_batch[:, :-1, :] @ self.Phi.T)

        e_bar = E.mean(axis=1) - self.mu_e
        T2    = n_ * np.einsum('bi,ij,bj->b', e_bar, self.Sig_e_inv, e_bar)

        Dc  = E - self.mu_e
        Se  = np.matmul(Dc.transpose(0,2,1), Dc) / n_   # (B,p,p) ← matmul
        Dev = np.einsum('ij,bjk->bik', self.Sig_e_inv, Se) - np.eye(p)
        W   = 0.5 * n_ * np.einsum('bij,bji->b', Dev, Dev)

        return (T2 + W).reshape(-1, 1)

    def monitor_sequence(self, X, n):
        T=X.shape[0]; K=(T-1)//n; stats=np.empty((K,1))
        for k in range(K): stats[k,0]=self.monitor_window(X[k*n:k*n+n+1])
        return stats

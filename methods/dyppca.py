"""
methods/dyppca.py  — DyPPCA monitor.
monitor_window_batch uses np.matmul chains instead of multi-dim einsum
so Apple Accelerate / AMX can be used on M3.
"""
import numpy as np
from scipy.linalg import inv, eigh


def _proj(U, S):
    """Compute U.T @ S[b] @ U for all b.  S:(B,p,p), U:(p,q) → (B,q,q).
    Uses matmul so Accelerate/AMX is invoked on Apple Silicon."""
    return np.einsum('pi,bpj->bij', U, S @ U)   # (B,p,q) then contract p


class DyPPCA:
    def __init__(self, q):
        self.q=q; self.nu0=None; self.U=None; self.Ue=None
        self.Lambda0=None; self.Gamma0=None; self.sigma0=None
        self.Phi0=None; self.Phi0_inv=None; self.Phi1=None
        self.Phi2=None; self.Phi3=None; self.M_mean=None

    def fit(self, X):
        N1,p=X.shape; N=N1-1; q=self.q
        X1=X[1:]; X2=X[:-1]
        xbar1=X1.mean(0); xbar2=X2.mean(0)
        D1=X1-xbar1; D2=X2-xbar2
        S1=D1.T@D1/N; S2=D2.T@D2/N; Q=D1.T@D2/N
        xbar=(xbar1+xbar2)/2; S=(S1+S2)/2
        eigvals,eigvecs=eigh(S)
        idx=np.argsort(eigvals)[::-1]; eigvals=eigvals[idx]; eigvecs=eigvecs[:,idx]
        U=eigvecs[:,:q]; Ue=eigvecs[:,q:]
        sigma=eigvals[q:].mean(); Lambda0=eigvals[:q]; Gamma0=U.T@Q@U
        self.nu0=xbar; self.U=U; self.Ue=Ue
        self.Lambda0=Lambda0; self.Gamma0=Gamma0; self.sigma0=sigma
        self._precompute(); return self

    @classmethod
    def from_true_model(cls, ic_model):
        U0=ic_model["U0"]; Ue=ic_model["Ue"]; A0=ic_model["A0"]
        B0=ic_model["B0"]; sigma0=ic_model["sigma0"]; Lambda0=ic_model["Lambda0"]
        Gamma0=U0.T@A0@B0@A0.T@U0
        m=cls(q=len(Lambda0))
        m.nu0=ic_model["nu0"].copy(); m.U=U0.copy(); m.Ue=Ue.copy()
        m.Lambda0=Lambda0.copy(); m.Gamma0=Gamma0; m.sigma0=float(sigma0)
        m._precompute(); return m

    def _precompute(self):
        L0=np.diag(self.Lambda0); G0=self.Gamma0; L0i=np.diag(1./self.Lambda0)
        self.Phi0=np.block([[L0,G0],[G0.T,L0]])
        Phi1=inv(L0-G0@L0i@G0.T); Phi2=inv(L0-G0.T@L0i@G0); Phi3=-Phi1@G0@L0i
        self.Phi1=Phi1; self.Phi2=Phi2; self.Phi3=Phi3
        self.Phi0_inv=np.block([[Phi1,Phi3],[Phi3.T,Phi2]])
        self.M_mean=Phi1+Phi2+Phi3+Phi3.T

    def monitor_window(self, X_win):
        n=len(X_win)-1; X1=X_win[1:]; X2=X_win[:-1]
        xbar_k1=X1.mean(0); xbar_k2=X2.mean(0); xbar_k=(xbar_k1+xbar_k2)/2
        D1=X1-xbar_k1; D2=X2-xbar_k2
        Sk1=D1.T@D1/n; Sk2=D2.T@D2/n; Qk=D1.T@D2/n; Sk=(Sk1+Sk2)/2
        U=self.U; Ue=self.Ue; sigma0=self.sigma0; nu0=self.nu0; q=self.q; p=U.shape[0]
        d_mean=xbar_k-nu0
        v1=U.T@d_mean; t1=float(n*v1@self.M_mean@v1)
        v2=Ue.T@d_mean; t2=float((2*n/sigma0)*v2@v2)
        UtSkU=U.T@Sk@U; UtQkU=U.T@Qk@U
        Mk=np.block([[UtSkU,UtQkU],[UtQkU.T,UtSkU]])
        Dev3=self.Phi0_inv@Mk-np.eye(2*q)
        t3=float(0.5*n*np.trace(Dev3@Dev3))
        UetSkUe=Ue.T@Sk@Ue; UetQkUe=Ue.T@Qk@Ue
        Dev4_S=(1./sigma0)*UetSkUe-np.eye(p-q)
        t4=float(n*np.trace(Dev4_S@Dev4_S)+(n/sigma0**2)*np.trace(UetQkUe@UetQkUe.T))
        return t1,t2,t3,t4,t1+t2+t3+t4

    def monitor_window_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Compute DyPPCA statistics for B windows simultaneously.
        Uses np.matmul (triggers Accelerate/AMX on Apple Silicon).

        Parameters
        ----------
        X_batch : (B, n+1, p)

        Returns
        -------
        stats : (B, 5)  columns = [t1, t2, t3, t4, t_total]
        """
        Bsz = X_batch.shape[0]
        n_  = X_batch.shape[1] - 1
        X1  = X_batch[:, 1:, :]    # (B, n, p)
        X2  = X_batch[:, :-1, :]

        xb1 = X1.mean(1); xb2 = X2.mean(1); xb = (xb1 + xb2) / 2
        D1  = X1 - xb1[:, None]; D2 = X2 - xb2[:, None]

        # Covariance blocks via batched matmul (hits Accelerate/AMX on M3)
        Sk1 = np.matmul(D1.transpose(0,2,1), D1) / n_  # (B,p,p)
        Sk2 = np.matmul(D2.transpose(0,2,1), D2) / n_
        Qk  = np.matmul(D1.transpose(0,2,1), D2) / n_
        Sk  = (Sk1 + Sk2) / 2

        d   = xb - self.nu0                              # (B, p)
        v1  = d @ self.U                                  # (B, q)
        t1  = n_ * np.einsum('bi,ij,bj->b', v1, self.M_mean, v1)
        v2  = d @ self.Ue
        t2  = (2 * n_ / self.sigma0) * np.sum(v2 ** 2, axis=1)

        # Project: U.T @ Sk[b] @ U  via matmul chain
        qq      = self.q
        UtSkU   = np.einsum('pi,bpj->bij', self.U, Sk @ self.U)   # (B,q,q)
        UtQkU   = np.einsum('pi,bpj->bij', self.U, Qk @ self.U)
        Mk      = np.zeros((Bsz, 2*qq, 2*qq))
        Mk[:, :qq, :qq] = UtSkU
        Mk[:, :qq, qq:] = UtQkU
        Mk[:, qq:, :qq] = np.swapaxes(UtQkU, 1, 2)
        Mk[:, qq:, qq:] = UtSkU
        Dev3 = np.einsum('ij,bjk->bik', self.Phi0_inv, Mk) - np.eye(2*qq)
        t3   = 0.5 * n_ * np.einsum('bij,bji->b', Dev3, Dev3)

        pq      = self.Ue.shape[1]
        UeSkUe  = np.einsum('pi,bpj->bij', self.Ue, Sk @ self.Ue)  # (B,pq,pq)
        UeQkUe  = np.einsum('pi,bpj->bij', self.Ue, Qk @ self.Ue)
        D4      = (1 / self.sigma0) * UeSkUe - np.eye(pq)
        t4      = (n_  * np.einsum('bij,bji->b', D4, D4) +
                   (n_ / self.sigma0**2) *
                   np.einsum('bij,bji->b', UeQkUe, np.swapaxes(UeQkUe, 1, 2)))

        tt = t1 + t2 + t3 + t4
        return np.column_stack([t1, t2, t3, t4, tt])

    def monitor_sequence(self, X, n):
        T=X.shape[0]; K=(T-1)//n; stats=np.empty((K,5))
        for k in range(K):
            s=k*n; stats[k]=self.monitor_window(X[s:s+n+1])
        return stats

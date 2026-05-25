"""
methods/lstm_ae.py
LSTM Autoencoder residual monitoring baseline.

Architecture: LSTM encoder → latent vector → LSTM decoder.
Input: lag-1 window [x_{t-1}, x_t]  ∈ R^{2 × p}
Reconstructs the same window; monitors pointwise residual
    r_t = x_t − x̂_t   via Hotelling T².
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Neural network definition
# ──────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 num_layers: int = 1):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True)
        self.enc_fc  = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.dec_fc  = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                batch_first=True)
        self.out_fc  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        seq_len = x.shape[1]

        # Encode
        _, (h, _) = self.encoder(x)
        z = self.enc_fc(h[-1])            # (batch, latent_dim)

        # Decode
        h0 = self.dec_fc(z).unsqueeze(0)  # (1, batch, hidden_dim)
        c0 = torch.zeros_like(h0)
        # Repeat z as input to each decoder step
        dec_in = h0.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder(dec_in, (h0, c0))
        recon  = self.out_fc(out)         # (batch, seq_len, input_dim)
        return recon


# ──────────────────────────────────────────────────────────────────────────────
# Monitor wrapper
# ──────────────────────────────────────────────────────────────────────────────

class LSTMAEMonitor:
    """
    LSTM-AE residual monitor.

    Parameters
    ----------
    input_dim   : p
    hidden_dim  : LSTM hidden size
    latent_dim  : bottleneck size
    num_layers  : LSTM depth
    epochs      : max training epochs
    batch_size  : mini-batch size
    lr          : learning rate
    patience    : early stopping patience
    device      : 'cpu', 'cuda', or 'mps'
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32,
                 latent_dim: int = 8, num_layers: int = 1,
                 epochs: int = 100, batch_size: int = 128,
                 lr: float = 1e-3, patience: int = 10,
                 device: str = "auto"):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model    = None
        self.mu_r     = None   # (p,)  IC residual mean
        self.Sig_r_inv= None   # (p,p) IC residual precision

    # ──────────────────────────────────────────────────────────────────────────
    # Phase I
    # ──────────────────────────────────────────────────────────────────────────

    def _build_windows(self, X: np.ndarray):
        """
        Build lag-1 windows from (N+1, p) → tensor (N, 2, p).
        """
        windows = np.stack([X[:-1], X[1:]], axis=1).astype(np.float32)
        return torch.from_numpy(windows)

    def fit(self, X: np.ndarray, seed: int = 0) -> "LSTMAEMonitor":
        """
        Train LSTM-AE on Phase I data.

        Parameters
        ----------
        X : array (N+1, p)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        data   = self._build_windows(X)   # (N, 2, p)
        N      = data.shape[0]
        n_val  = max(1, int(0.1 * N))
        n_tr   = N - n_val

        tr_set  = TensorDataset(data[:n_tr])
        val_set = TensorDataset(data[n_tr:])
        tr_ldr  = DataLoader(tr_set,  batch_size=self.batch_size, shuffle=True)
        val_ldr = DataLoader(val_set, batch_size=self.batch_size)

        self.model = LSTMAutoencoder(
            self.input_dim, self.hidden_dim, self.latent_dim, self.num_layers
        ).to(self.device)

        opt      = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn  = nn.MSELoss()
        best_val = np.inf
        no_impv  = 0
        best_sd  = None

        for ep in range(self.epochs):
            # ── train ─────────────────────────────────────────────────────────
            self.model.train()
            for (batch,) in tr_ldr:
                batch = batch.to(self.device)
                opt.zero_grad()
                recon = self.model(batch)
                loss  = loss_fn(recon, batch)
                loss.backward()
                opt.step()

            # ── validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch,) in val_ldr:
                    batch = batch.to(self.device)
                    recon = self.model(batch)
                    val_losses.append(loss_fn(recon, batch).item())
            val_loss = np.mean(val_losses)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_sd  = {k: v.cpu().clone() for k, v in
                            self.model.state_dict().items()}
                no_impv  = 0
            else:
                no_impv += 1
                if no_impv >= self.patience:
                    break

        if best_sd is not None:
            self.model.load_state_dict(best_sd)

        # ── Calibrate residual statistics ─────────────────────────────────────
        residuals = self._compute_residuals_batch(X)   # (N, p)
        self.mu_r = residuals.mean(axis=0)
        D = residuals - self.mu_r
        Sig_r = D.T @ D / len(D)
        # Regularise for numerical stability
        Sig_r += 1e-6 * np.eye(self.input_dim)
        self.Sig_r_inv = np.linalg.inv(Sig_r)
        return self

    def _compute_residuals_batch(self, X: np.ndarray) -> np.ndarray:
        """Return current-step reconstruction residuals (N, p)."""
        self.model.eval()
        windows = self._build_windows(X)   # (N, 2, p)
        loader  = DataLoader(TensorDataset(windows),
                             batch_size=512, shuffle=False)
        residuals = []
        with torch.no_grad():
            for (batch,) in loader:
                batch  = batch.to(self.device)
                recon  = self.model(batch)                  # (B, 2, p)
                # current-step residual: step index 1
                r = (batch[:, 1, :] - recon[:, 1, :]).cpu().numpy()
                residuals.append(r)
        return np.vstack(residuals)   # (N, p)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase II
    # ──────────────────────────────────────────────────────────────────────────

    def monitor_window(self, X_win: np.ndarray) -> float:
        """
        Compute window-mean Hotelling T² on reconstruction residuals.

        Parameters
        ----------
        X_win : array (n+1, p)

        Returns
        -------
        T2_ae : float
        """
        residuals = self._compute_residuals_batch(X_win)   # (n, p)
        n = residuals.shape[0]
        r_bar = residuals.mean(axis=0) - self.mu_r         # (p,)
        T2_ae = float(n * r_bar @ self.Sig_r_inv @ r_bar)
        return T2_ae

    def monitor_sequence(self, X: np.ndarray, n: int):
        """
        Apply sliding-window monitoring.

        Returns
        -------
        stats : array (K, 1)  column = T2_ae
        """
        T = X.shape[0]
        K = (T - 1) // n
        stats = np.empty((K, 1))
        for k in range(K):
            start = k * n
            end   = start + n + 1
            X_win = X[start:end]
            stats[k, 0] = self.monitor_window(X_win)
        return stats

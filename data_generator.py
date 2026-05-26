"""
data_generator.py
IC data and five OC case generators.

IC model (equations 1–4 of the paper):
    x_t = ν₀ + A₀ z_t + ε_t,   ε_t ~ N(0, σ₀ I_p)
    z_t = B₀ z_{t-1} + ξ_t,    ξ_t ~ N(0, Ψ₀)
    Ψ₀  = I_q − B₀ B₀ᵀ         (stationarity: Cov(z_t) = I_q)

build_ic_model() takes A₀, B₀, Ψ₀ directly from config (equations 63–64).
U₀ and Λ₀ are derived from SVD(A₀) rather than generated randomly.

OC cases (Section 5.3):
    Case I   — latent mean shift:        E(z_t) = δ = d · e₁
    Case II  — obs noise mean shift:     E(ε_t) = d · u_{q+1}
    Case III — latent AR matrix shift:   B₁ = B₀ + d · E₁₂  (off-diagonal)
    Case IV  — latent covariance shift:  Cov(z_t) = I_q + d · e₁e₁ᵀ
    Case V   — obs noise cov shift:      Cov(ε_t) = σ₀ I + d·σ₀·u_{q+1}u_{q+1}ᵀ
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# IC model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_ic_model(p, q, sigma0, A0, B0, Psi0):
    """
    Build and return the IC model parameter dict from the given matrices.

    U₀ and Λ₀ are derived via full SVD of A₀:
        A₀  = U_full · diag(sv) · Vᵀ
        U₀  = U_full[:, :q]          latent loading subspace  (p × q)
        Uₑ  = U_full[:, q:]          residual complement       (p × p-q)
        Λ₀  = sv² + σ₀               top-q eigenvalues of Cov(xₜ)

    Parameters
    ----------
    p, q    : observation and latent dimensions
    sigma0  : scalar noise variance
    A0      : (p, q) loading matrix  — taken directly from the paper
    B0      : (q, q) latent AR matrix
    Psi0    : (q, q) innovation covariance  (must satisfy B0 B0ᵀ + Ψ₀ = I_q)

    Returns
    -------
    dict with keys: nu0, U0, Ue, A0, B0, Psi0, sigma0, Lambda0, p, q
    """
    A0   = np.asarray(A0, dtype=float)
    B0   = np.asarray(B0, dtype=float)
    Psi0 = np.asarray(Psi0, dtype=float)

    # Verify stationarity
    err = np.linalg.norm(B0 @ B0.T + Psi0 - np.eye(q))
    assert err < 1e-8, f"Stationarity violated: ‖B₀B₀ᵀ + Ψ₀ − I‖ = {err:.2e}"

    # Verify Ψ₀ is positive definite
    ev_psi = np.linalg.eigvalsh(Psi0)
    assert ev_psi.min() > 0, f"Ψ₀ is not PD; min eigenvalue = {ev_psi.min():.4f}"

    # Full SVD of A₀ → orthonormal basis of R^p
    U_full, sv, _ = np.linalg.svd(A0, full_matrices=True)  # U_full: (p, p)
    U0 = U_full[:, :q]    # (p, q)  latent loading subspace
    Ue = U_full[:, q:]    # (p, p-q) residual complement

    # Top-q eigenvalues of Cov(xₜ) = A₀A₀ᵀ + σ₀ I_p
    Lambda0 = sv**2 + sigma0   # (q,)

    return dict(
        nu0     = np.zeros(p),
        U0      = U0,
        Ue      = Ue,
        A0      = A0.copy(),
        B0      = B0.copy(),
        Psi0    = Psi0.copy(),
        sigma0  = float(sigma0),
        Lambda0 = Lambda0,
        p       = int(p),
        q       = int(q),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation engine
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ic(model, T, rng=None):
    """Generate T IC observations starting from z = 0."""
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    p, q = A0.shape
    L    = np.linalg.cholesky(Psi0)
    X    = np.empty((T, p))
    z    = np.zeros(q)
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case I — latent mean shift  E(z_t) = δ = d · e₁
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case1(model, T, d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    p, q = A0.shape
    L    = np.linalg.cholesky(Psi0)
    delta = np.zeros(q); delta[0] = d
    X     = np.empty((T, p))
    z     = delta.copy()
    for t in range(T):
        z    = delta + B0 @ (z - delta) + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case II — observation noise mean shift  E(ε_t) = d · u_{q+1}
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case2(model, T, d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    p, q = A0.shape
    L           = np.linalg.cholesky(Psi0)
    delta_tilde = d * model["Ue"][:, 0]   # d · u_{q+1}
    X = np.empty((T, p))
    z = np.zeros(q)
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p) + delta_tilde
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case III — latent AR matrix shift  ΔB = d · E₁₂  (off-diagonal)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case3(model, T, d, rng=None):
    """
    B₁ = B₀ + d·E₁₂  (E₁₂: element [0,1] perturbed by +d).
    Ψ₁ = I_q − B₁B₁ᵀ   preserves Cov(z_t) = I_q.
    """
    if rng is None:
        rng = np.random.default_rng()
    A0, sigma0, nu0 = model["A0"], model["sigma0"], model["nu0"]
    p, q = A0.shape
    B1   = model["B0"].copy()
    B1[0, 1] += d
    Psi1 = np.eye(q) - B1 @ B1.T
    ev   = np.linalg.eigvalsh(Psi1)
    if ev.min() <= 0:
        raise ValueError(f"Case III d={d:.2f}: Ψ₁ not PD (min ev={ev.min():.4f})")
    L    = np.linalg.cholesky(Psi1)
    X    = np.empty((T, p))
    z    = np.zeros(q)
    for t in range(T):
        z    = B1 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case IV — latent covariance shift  Cov(z_t) = I_q + d · e₁e₁ᵀ
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case4(model, T, d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, sigma0, nu0 = (model["A0"], model["B0"],
                            model["sigma0"], model["nu0"])
    p, q = A0.shape
    Delta_z  = np.zeros((q, q)); Delta_z[0, 0] = d
    Sigma_z1 = np.eye(q) + Delta_z
    Psi1     = Sigma_z1 - B0 @ Sigma_z1 @ B0.T
    ev       = np.linalg.eigvalsh(Psi1)
    if ev.min() <= 0:
        raise ValueError(f"Case IV d={d:.2f}: Ψ₁ not PD")
    L = np.linalg.cholesky(Psi1)
    X = np.empty((T, p))
    z = np.zeros(q)
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case V — local obs noise cov shift  Cov(ε_t) = σ₀I + d·σ₀·u_{q+1}u_{q+1}ᵀ
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case5(model, T, d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    p, q = A0.shape
    L    = np.linalg.cholesky(Psi0)
    u_e1 = model["Ue"][:, 0]
    # ε_t ~ N(0, σ₀(I + d u_e1 u_e1ᵀ)):  scale that direction by √(1+d)
    sqrt_extra = np.sqrt(1.0 + d) - 1.0
    X = np.empty((T, p))
    z = np.zeros(q)
    for t in range(T):
        z   = B0 @ z + L @ rng.standard_normal(q)
        xi  = rng.standard_normal(p)
        eps = np.sqrt(sigma0) * (xi + sqrt_extra * (u_e1 @ xi) * u_e1)
        X[t] = nu0 + A0 @ z + eps
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Stateful simulation (carries latent state z across windows)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ic_stateful(model, T, z_init, rng):
    """
    Simulate T IC observations starting from latent state z_init.
    Returns (X, z_final) so consecutive calls form a continuous series.
    """
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    p = model["A0"].shape[0]
    L = np.linalg.cholesky(Psi0)
    X = np.empty((T, p))
    z = z_init.copy()
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(model["q"])
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X, z


def simulate_oc_stateful(model, T, case, d, z_init, rng):
    """
    Simulate T OC observations starting from z_init.
    Returns (X, z_final).
    """
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    Ue           = model["Ue"]
    p, q = model["A0"].shape
    X    = np.empty((T, p))
    z    = z_init.copy()

    if case == "case1":
        L     = np.linalg.cholesky(Psi0)
        delta = np.zeros(q); delta[0] = d
        for t in range(T):
            z    = delta + B0 @ (z - delta) + L @ rng.standard_normal(q)
            X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)

    elif case == "case2":
        L           = np.linalg.cholesky(Psi0)
        delta_tilde = d * Ue[:, 0]
        for t in range(T):
            z    = B0 @ z + L @ rng.standard_normal(q)
            X[t] = nu0 + A0 @ z + np.sqrt(sigma0)*rng.standard_normal(p) + delta_tilde

    elif case == "case3":
        B1   = B0.copy(); B1[0, 1] += d
        Psi1 = np.eye(q) - B1 @ B1.T
        L1   = np.linalg.cholesky(Psi1)
        for t in range(T):
            z    = B1 @ z + L1 @ rng.standard_normal(q)
            X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)

    elif case == "case4":
        Delta_z = np.zeros((q, q)); Delta_z[0, 0] = d
        Sigma_z = np.eye(q) + Delta_z
        Psi1    = Sigma_z - B0 @ Sigma_z @ B0.T
        L1      = np.linalg.cholesky(Psi1)
        for t in range(T):
            z    = B0 @ z + L1 @ rng.standard_normal(q)
            X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)

    elif case == "case5":
        L_z   = np.linalg.cholesky(Psi0)
        u_e1  = Ue[:, 0]
        sqrt_extra = np.sqrt(1.0 + d) - 1.0
        for t in range(T):
            z    = B0 @ z + L_z @ rng.standard_normal(q)
            xi   = rng.standard_normal(p)
            eps  = np.sqrt(sigma0) * (xi + sqrt_extra * (u_e1 @ xi) * u_e1)
            X[t] = nu0 + A0 @ z + eps

    else:
        raise ValueError(f"Unknown case: {case}")

    return X, z


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table for batch generators
# ─────────────────────────────────────────────────────────────────────────────

OC_GENERATORS = {
    "case1": generate_oc_case1,
    "case2": generate_oc_case2,
    "case3": generate_oc_case3,
    "case4": generate_oc_case4,
    "case5": generate_oc_case5,
}

def generate_oc(model, T, case, d, rng=None):
    if case not in OC_GENERATORS:
        raise ValueError(f"Unknown case: {case}")
    return OC_GENERATORS[case](model, T, d, rng=rng)

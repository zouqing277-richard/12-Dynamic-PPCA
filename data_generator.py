"""
data_generator.py
IC 数据与五种 OC 情形的生成函数。

IC 模型:
    x_t = ν₀ + A₀ z_t + ε_t,   ε_t ~ N(0, σ₀ I_p)
    z_t = B₀ z_{t-1} + ξ_t,    ξ_t ~ N(0, Ψ₀)
    Ψ₀  = I_q − B₀ B₀ᵀ         （保证 Cov(z_t) = I_q）

OC 情形（严格按照 Section 5.3）：
    Case I   — latent mean shift:        E(z_t) = δ = d·e₁
    Case II  — obs noise mean shift:     E(ε_t) = d·u_{q+1}
    Case III — latent AR matrix shift:   B₁ = B₀ + d·E₁₂  ← 非对角扰动
    Case IV  — latent covariance shift:  Cov(z_t) = I_q + d·e₁e₁ᵀ
    Case V   — obs noise cov shift:      Cov(ε_t) = σ₀I + d·σ₀·u_{q+1}u_{q+1}ᵀ  ← 局部
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# IC 模型构建
# ─────────────────────────────────────────────────────────────────────────────

def build_ic_model(p, q, sigma0, Lambda0, B0, seed=42):
    """
    构造并返回 IC 模型参数字典。
    A₀ = U₀ (Λ₀ − σ₀ I_q)^{1/2},  其中 U₀ 由 QR 分解随机正交矩阵得到。
    """
    rng = np.random.default_rng(seed)
    G   = rng.standard_normal((p, q))
    U0, _ = np.linalg.qr(G)
    U0  = U0[:, :q]
    A0  = U0 @ np.diag(np.sqrt(Lambda0 - sigma0))
    Psi0 = np.eye(q) - B0 @ B0.T
    assert np.linalg.eigvalsh(Psi0).min() > 0, "Ψ₀ 不正定，请调整 B₀"

    # 残差正交补 Uₑ（固定，供 Case II / Case V 使用）
    full_basis = np.eye(p)
    Ue_cols = []
    for v in full_basis.T:
        v = v - U0 @ (U0.T @ v)
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            Ue_cols.append(v / norm)
        if len(Ue_cols) == p - q:
            break
    Ue = np.column_stack(Ue_cols)   # (p, p-q)

    return dict(nu0=np.zeros(p), U0=U0, Ue=Ue, A0=A0,
                B0=B0, Psi0=Psi0, sigma0=sigma0, Lambda0=Lambda0,
                p=p, q=q)


# ─────────────────────────────────────────────────────────────────────────────
# 核心仿真引擎
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ic(model, T, rng=None):
    """生成 T 个 IC 观测（从 z=0 出发，不需烧入期，已验证稳态统计量一致）。"""
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0, sigma0, nu0 = (model["A0"], model["B0"],
                                  model["Psi0"], model["sigma0"], model["nu0"])
    p, q = A0.shape
    L    = np.linalg.cholesky(Psi0)
    X    = np.empty((T, p))
    z    = np.zeros(q)
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case I — latent mean shift
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case1(model, T, d, rng=None):
    """
    E(z_t) = δ = d·e₁。
    过程：z_t = δ + B₀(z_{t-1} − δ) + ξ_t。
    """
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0, sigma0, nu0 = (model["A0"], model["B0"],
                                  model["Psi0"], model["sigma0"], model["nu0"])
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
# OC Case II — observation noise mean shift
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case2(model, T, d, rng=None):
    """
    E(ε_t) = δ̃ = d·u_{q+1}（残差子空间第一个方向）。
    """
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0, sigma0, nu0 = (model["A0"], model["B0"],
                                  model["Psi0"], model["sigma0"], model["nu0"])
    p, q = A0.shape
    L    = np.linalg.cholesky(Psi0)
    # u_{q+1}：Ue 的第一列（对应观测噪声的第一个正交方向）
    delta_tilde = d * model["Ue"][:, 0]
    X = np.empty((T, p))
    z = np.zeros(q)
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p) + delta_tilde
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case III — latent AR matrix shift  ΔB = d·E₁₂ (非对角)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case3(model, T, d, rng=None):
    """
    B₁ = B₀ + d·E₁₂  （E₁₂: 第 1 行第 2 列元素 +d，0-indexed: B[0,1]）。
    Ψ₁ = I_q − B₁ B₁ᵀ  保持 Cov(z_t) = I_q 不变。

    注意：这是 Section 5.3 指定的 off-diagonal 扰动，
    与之前代码的 E₁₁（对角）不同。
    """
    if rng is None:
        rng = np.random.default_rng()
    A0, sigma0, nu0 = model["A0"], model["sigma0"], model["nu0"]
    p, q = A0.shape
    B1   = model["B0"].copy()
    B1[0, 1] += d                         # ← E₁₂ 非对角扰动
    Psi1 = np.eye(q) - B1 @ B1.T
    ev   = np.linalg.eigvalsh(Psi1)
    if ev.min() <= 0:
        raise ValueError(f"Case III d={d:.2f}: Ψ₁ 不正定（最小特征值={ev.min():.4f}）")
    L    = np.linalg.cholesky(Psi1)
    X    = np.empty((T, p))
    z    = np.zeros(q)
    for t in range(T):
        z    = B1 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case IV — latent covariance shift
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case4(model, T, d, rng=None):
    """
    Cov(z_t) = I_q + d·e₁e₁ᵀ，B₀ 不变。
    Ψ₁ = Σ_z − B₀ Σ_z B₀ᵀ  其中 Σ_z = I_q + d·e₁e₁ᵀ。
    """
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
        raise ValueError(f"Case IV d={d:.2f}: Ψ₁ 不正定")
    L = np.linalg.cholesky(Psi1)
    X = np.empty((T, p))
    z = np.zeros(q)
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# OC Case V — observation noise covariance shift (局部，沿 u_{q+1} 方向)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oc_case5(model, T, d, rng=None):
    """
    Cov(ε_t) = σ₀ I_p + d·σ₀·u_{q+1}u_{q+1}ᵀ。
    只有残差子空间第一个方向的噪声方差增加，其余不变。

    注意：之前代码用全局放大 σ₀(1+d)I，现改为局部扰动。
    """
    if rng is None:
        rng = np.random.default_rng()
    A0, B0, Psi0, sigma0, nu0 = (model["A0"], model["B0"],
                                  model["Psi0"], model["sigma0"], model["nu0"])
    p, q = A0.shape
    L    = np.linalg.cholesky(Psi0)
    u_e1 = model["Ue"][:, 0]             # u_{q+1}

    # Cov(ε_t) = σ₀(I + d·u_e1 u_e1ᵀ)，通过 Cholesky 分解采样
    # ε_t = √σ₀ · (ξ + √d · (u_e1ᵀ ξ) u_e1)，其中 ξ ~ N(0,I)
    # 等价构造：ε_t = √σ₀ · (I + (√(1+d)−1) u_e1 u_e1ᵀ) · ξ
    # 用 rank-1 更新更稳定
    sqrt_extra = np.sqrt(1.0 + d) - 1.0   # 使得该方向标准差乘以 √(1+d)

    X = np.empty((T, p))
    z = np.zeros(q)
    for t in range(T):
        z   = B0 @ z + L @ rng.standard_normal(q)
        xi  = rng.standard_normal(p)
        eps = np.sqrt(sigma0) * (xi + sqrt_extra * (u_e1 @ xi) * u_e1)
        X[t] = nu0 + A0 @ z + eps
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 统一调度接口
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Stateful simulation  (preserves latent state z across windows)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ic_stateful(model, T, z_init, rng):
    """
    Simulate T IC observations starting from latent state z_init.

    Unlike simulate_ic(), accepts the current latent state and returns the
    final latent state so consecutive calls produce a truly continuous series.

    Returns
    -------
    X       : (T, p)
    z_final : (q,) latent state after last observation → pass to next call
    """
    A0, B0, Psi0 = model["A0"], model["B0"], model["Psi0"]
    sigma0, nu0  = model["sigma0"], model["nu0"]
    p, q = model["A0"].shape
    L    = np.linalg.cholesky(Psi0)
    X    = np.empty((T, p))
    z    = z_init.copy()
    for t in range(T):
        z    = B0 @ z + L @ rng.standard_normal(q)
        X[t] = nu0 + A0 @ z + np.sqrt(sigma0) * rng.standard_normal(p)
    return X, z


def simulate_oc_stateful(model, T, case, d, z_init, rng):
    """
    Simulate T OC observations starting from latent state z_init.

    Stateful version of generate_oc(): carries z_t forward so consecutive
    windows form a truly continuous time series.

    Returns
    -------
    X       : (T, p)
    z_final : (q,)
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
        Sig_e = sigma0 * np.eye(p) + d * sigma0 * np.outer(u_e1, u_e1)
        L_e   = np.linalg.cholesky(Sig_e)
        for t in range(T):
            z    = B0 @ z + L_z @ rng.standard_normal(q)
            X[t] = nu0 + A0 @ z + L_e @ rng.standard_normal(p)

    else:
        raise ValueError(f"Unknown case: {case}")

    return X, z


OC_GENERATORS = {
    "case1": generate_oc_case1,
    "case2": generate_oc_case2,
    "case3": generate_oc_case3,
    "case4": generate_oc_case4,
    "case5": generate_oc_case5,
}

def generate_oc(model, T, case, d, rng=None):
    if case not in OC_GENERATORS:
        raise ValueError(f"未知 case: {case}")
    return OC_GENERATORS[case](model, T, d, rng=rng)

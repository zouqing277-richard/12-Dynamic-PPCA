"""
evaluation.py
ARL₁ 估计与 DyPPCA 诊断分量分析。

每个 OC replication：
  1. 生成 OC 序列（Phase I 模型固定，只重新生成 OC 数据）
  2. 滑动窗口扫描，记录首次报警窗口编号（截断于 K_max）
  3. 汇总 B₁ 次的均值和 SE = std/√B₁

Case → 每个方法使用哪个统计量报警（单统计量，不用 OR）：
  均值类 (I, II)  : DyPPCA t_total | DPCA OR | PPCA W  | VAR T2  | LSTM T2
  协方差类 (III-V): DyPPCA t_total | DPCA OR | PPCA R  | VAR W   | LSTM T2
"""

import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 报警函数（每个方法 × 每类 case）
# ─────────────────────────────────────────────────────────────────────────────

def _alarm_dyppca(row, ucls):
    return row[4] > ucls["t_total"]

def _alarm_dpca(row, ucls):
    q_T2 = ucls["T2"]; q_Q = ucls["Q"]
    return row[0] > q_T2 or row[1] > q_Q

def _alarm_ppca_mean(row, ucls):     # W
    return row[0] > ucls["W"]

def _alarm_ppca_cov(row, ucls):      # R
    return row[3] > ucls["R"]

def _alarm_var_mean(row, ucls):      # T2
    return row[0] > ucls["T2"]

def _alarm_var_cov(row, ucls):       # W_cov
    return row[1] > ucls["W"]

def _alarm_lstm(row, ucls):
    return row[0] > ucls["T2"]


# 每个 case 对应的报警函数（每方法均用单统计量，不混用 OR+单统计量）
ALARM_RULES = {
    "case1": {                          # latent mean shift
        "dyppca":       _alarm_dyppca,
        "dpca":         _alarm_dpca,
        "static_ppca":  _alarm_ppca_mean,
        "var_residual": _alarm_var_mean,
        "lstm_ae":      _alarm_lstm,
    },
    "case2": {                          # obs noise mean shift
        "dyppca":       _alarm_dyppca,
        "dpca":         _alarm_dpca,
        "static_ppca":  _alarm_ppca_mean,
        "var_residual": _alarm_var_mean,
        "lstm_ae":      _alarm_lstm,
    },
    "case3": {                          # latent AR shift (ΔB = d·E₁₂)
        "dyppca":       _alarm_dyppca,
        "dpca":         _alarm_dpca,
        "static_ppca":  _alarm_ppca_cov,
        "var_residual": _alarm_var_cov,
        "lstm_ae":      _alarm_lstm,
    },
    "case4": {                          # latent covariance shift
        "dyppca":       _alarm_dyppca,
        "dpca":         _alarm_dpca,
        "static_ppca":  _alarm_ppca_cov,
        "var_residual": _alarm_var_cov,
        "lstm_ae":      _alarm_lstm,
    },
    "case5": {                          # local obs noise cov shift
        "dyppca":       _alarm_dyppca,
        "dpca":         _alarm_dpca,
        "static_ppca":  _alarm_ppca_cov,
        "var_residual": _alarm_var_cov,
        "lstm_ae":      _alarm_lstm,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 单次 replication 的 run length（滑动窗口扫描）
# ─────────────────────────────────────────────────────────────────────────────

def _run_length(monitor, ucls, X_oc, n_window, case, method_name, K_max):
    """
    在 X_oc 上滑动窗口，返回首次报警的窗口编号（1-based）。
    超过 K_max 未报警则返回 K_max。
    """
    alarm_fn = ALARM_RULES[case][method_name]
    T        = X_oc.shape[0]
    for k in range(K_max):
        start = k * n_window
        if start + n_window + 1 > T:
            return K_max
        row = monitor.monitor_window(X_oc[start : start + n_window + 1])
        if alarm_fn(row, ucls):
            return k + 1
    return K_max


# ─────────────────────────────────────────────────────────────────────────────
# ARL₁ 估计：B₁ 次 OC replications
# ─────────────────────────────────────────────────────────────────────────────

def run_arl_experiment(method_name, monitor, ucls, model_ic,
                       case, d, n_reps, n_window, K_max=2000, rng=None):
    """
    用 n_reps 次 OC replications 估计 ARL₁。

    Phase I 模型固定（已传入 monitor）；每次只重新生成 OC 序列。

    返回
    ----
    arl_mean : float
    arl_se   : float   SE = std / √n_reps
    rls      : (n_reps,) 原始 run lengths（用于后续分析）
    """
    from data_generator import generate_oc
    if rng is None:
        rng = np.random.default_rng()

    T_oc = K_max * n_window + 1
    rls  = np.empty(n_reps, dtype=np.float64)

    for b in range(n_reps):
        X_oc  = generate_oc(model_ic, T_oc, case, d, rng=rng)
        rls[b] = _run_length(monitor, ucls, X_oc, n_window,
                              case, method_name, K_max)

    arl_mean = float(rls.mean())
    arl_se   = float(rls.std() / np.sqrt(n_reps))
    return arl_mean, arl_se, rls


# ─────────────────────────────────────────────────────────────────────────────
# DyPPCA 诊断分量比例（报警时各分量占比）
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_ratios(dyppca_monitor, ucls, model_ic, case, d,
                      n_reps, n_window, K_max=2000, rng=None):
    """
    计算报警窗口时各分量比例 ρⱼ = tⱼ / (t₁+t₂+t₃+t₄)。
    返回均值字典 {"rho1":..., "rho2":..., "rho3":..., "rho4":...}。
    """
    from data_generator import generate_oc
    if rng is None:
        rng = np.random.default_rng()

    h     = ucls["t_total"]
    T_oc  = K_max * n_window + 1
    rhos  = []

    for _ in range(n_reps):
        X_oc = generate_oc(model_ic, T_oc, case, d, rng=rng)
        for k in range(K_max):
            start = k * n_window
            if start + n_window + 1 > X_oc.shape[0]:
                break
            row = dyppca_monitor.monitor_window(
                X_oc[start : start + n_window + 1]
            )
            if row[4] > h:          # t_total > UCL
                total = row[4]
                if total > 0:
                    rhos.append(np.array(row[:4]) / total)
                break

    if not rhos:
        return {"rho1": np.nan, "rho2": np.nan,
                "rho3": np.nan, "rho4": np.nan}
    rhos = np.array(rhos).mean(axis=0)
    return {"rho1": float(rhos[0]), "rho2": float(rhos[1]),
            "rho3": float(rhos[2]), "rho4": float(rhos[3])}

"""速度定数 k(T) の温度依存性モデル.

Arrhenius と非 Arrhenius (modified Arrhenius、温度依存 Ea) を提供.
速度論モデルの k は本モジュールが温度から決定する.
"""
from __future__ import annotations

import numpy as np

R_GAS = 8.314  # J/(mol·K)


def arrhenius(temp_c: float | np.ndarray, ln_a: float, ea_kj: float) -> np.ndarray:
    """標準 Arrhenius: k = exp(ln_a - Ea/(R T)).  Ea は kJ/mol."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    return np.exp(ln_a - ea_kj * 1000.0 / (R_GAS * t_k))


def modified_arrhenius_convex(temp_c, ln_a: float, ea_kj: float, t_ref_k: float = 313.15) -> np.ndarray:
    """T^n × Arrhenius (n=-4): ln k vs 1/T が下に凸 (高温側でより緩く増える)."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    return ((t_k / t_ref_k) ** (-4.0)) * np.exp(ln_a - ea_kj * 1000.0 / (R_GAS * t_k))


def modified_arrhenius_concave(temp_c, ln_a: float, ea_kj: float, t_ref_k: float = 313.15) -> np.ndarray:
    """T^n × Arrhenius (n=+4): ln k vs 1/T が上に凸 (高温側でより速く増える)."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    return ((t_k / t_ref_k) ** (4.0)) * np.exp(ln_a - ea_kj * 1000.0 / (R_GAS * t_k))


K_OF_T_REGISTRY = {
    "arrhenius": arrhenius,
    "modified_arrhenius_convex": modified_arrhenius_convex,
    "modified_arrhenius_concave": modified_arrhenius_concave,
}


def _lna_from_k25(k_25: float, ea_kj: float, k_of_t: str) -> float:
    """25°C での k と Ea, 温度依存性モデルから ln A を逆算."""
    t_25k = 298.15
    if k_of_t == "arrhenius":
        return float(np.log(k_25) + ea_kj * 1000.0 / (R_GAS * t_25k))
    if k_of_t == "modified_arrhenius_convex":
        return float(np.log(k_25) + ea_kj * 1000.0 / (R_GAS * t_25k) + 4.0 * np.log(t_25k / 313.15))
    if k_of_t == "modified_arrhenius_concave":
        return float(np.log(k_25) + ea_kj * 1000.0 / (R_GAS * t_25k) - 4.0 * np.log(t_25k / 313.15))
    raise ValueError(f"未知の k_of_t モデル: {k_of_t}")


def calibrate_lna_for_true_sl(
    target_sl_at_25c_months: float,
    ea_kj: float,
    kinetics: str,
    spec_lower: float = 95.0,
    initial_content: float = 100.0,
    k_of_t: str = "arrhenius",
    extra_params: dict | None = None,
) -> tuple[float, float]:
    """kinetics 種別ごとに k_25 を解いて ln A を返す.

    真値 SL(25°C で真の content が spec_lower に達する時間)が
    target_sl_at_25c_months と一致するように k_25 を kinetics 別に較正する.
    これにより 1 次以外の速度論でも「真値 SL = target」となり、推定器バイアス
    評価の基準点 (仕様書 §4) が崩れない.

    Returns
    -------
    (ln_a, k_25) : tuple[float, float]
    """
    extra = dict(extra_params or {})
    T = float(target_sl_at_25c_months)

    if kinetics == "first_order":
        k_25 = -np.log(spec_lower / initial_content) / T

    elif kinetics == "second_order":
        # C(t) = C0 / (1 + k C0 t) → t* = (C0/spec − 1) / (k C0)
        k_25 = (initial_content / spec_lower - 1.0) / (initial_content * T)

    elif kinetics == "induction":
        # Avrami: C(t) = C0 * exp(-(k t)^n).  C(T) = spec_lower で解く.
        n_avrami = float(extra.get("n_avrami", 2.0))
        ln_ratio = -np.log(spec_lower / initial_content)
        k_25 = (ln_ratio ** (1.0 / n_avrami)) / T

    elif kinetics == "autocatalytic":
        # 数値求解 (kinetics.autocatalytic の数値積分を使う)
        from scipy.optimize import brentq

        from .kinetics import autocatalytic

        alpha = float(extra.get("alpha", 0.05))

        def f(k: float) -> float:
            c = autocatalytic(
                np.array([T]),
                k,
                c0=initial_content,
                alpha=alpha,
            )
            return float(c[0]) - spec_lower

        k_25 = float(brentq(f, 1e-8, 10.0))

    else:
        raise ValueError(f"未知の kinetics: {kinetics}")

    return _lna_from_k25(k_25, ea_kj, k_of_t), float(k_25)


def solve_true_sl(
    k_at_temp: float,
    kinetics: str,
    spec_lower: float = 95.0,
    initial_content: float = 100.0,
    extra_params: dict | None = None,
    t_max_months: float = 1200.0,
) -> float:
    """ある温度の k に対し、真の速度論で content=spec_lower となる時間を求める.

    分解が spec_lower に達しない場合 (k=0 等) は t_max_months を返す.
    """
    extra = dict(extra_params or {})
    if k_at_temp <= 0:
        return float(t_max_months)

    if kinetics == "first_order":
        return float(-np.log(spec_lower / initial_content) / k_at_temp)

    if kinetics == "second_order":
        return float((initial_content / spec_lower - 1.0) / (initial_content * k_at_temp))

    if kinetics == "induction":
        n_avrami = float(extra.get("n_avrami", 2.0))
        ln_ratio = -np.log(spec_lower / initial_content)
        return float((ln_ratio ** (1.0 / n_avrami)) / k_at_temp)

    if kinetics == "autocatalytic":
        from scipy.optimize import brentq

        from .kinetics import autocatalytic

        alpha = float(extra.get("alpha", 0.05))

        def f(t: float) -> float:
            c = autocatalytic(np.array([t]), k_at_temp, c0=initial_content, alpha=alpha)
            return float(c[0]) - spec_lower

        f_lo = f(1e-6)
        f_hi = f(t_max_months)
        if f_lo * f_hi > 0:
            return float(t_max_months)
        return float(brentq(f, 1e-6, t_max_months))

    raise ValueError(f"未知の kinetics: {kinetics}")

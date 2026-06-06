# MIT License — Copyright (c) 2026 Yasushi Arai
#
# Provenance (author's own code, dual-licensed):
#   source:  apo-cyber/cmc-platform (private)
#   path:    apps/navigator/backend/app/calculations/phase5/classical_stability.py
#   commit:  34012d0
#   copied:  2026-06-06
#   scope:   "scientific core only (MIT dual-license of author's own code)"
#
# Vendored verbatim (single self-contained scientific function; no removals).
# `classical_ich_q1e_single_temp` is the entry point used by
# paper_a.analysis.estimators.classical_ich_q1e.
"""古典 ICH Q1E §B.1 単温度線形回帰

長期保存(25°C)の安定性データに対して、ICH Q1E §B.1(線形回帰)
+ §2.6(片側 95% 信頼区間下限が規格下限と交わる時点を有効期間とする)
の規定通りに計算する。

ベイジアン Arrhenius 解析(`run_bayesian_stability`)の補助的な二重チェック
として、同一の有効期間問題を完全に異なる手法で解く。Layer 4 監査 L4.8 で
両者を並列比較した結果、ベイジアンの方が古典より保守的(SL_lo95 が短い)
であることが確認されている。

参考: docs/audit/l4_8_classical_vs_bayesian_comparison.md
      docs/audit/classical_ich_q1e_comparison.py (監査用ベンチマーク)
"""
from typing import List, Dict
import numpy as np
from scipy import stats


def classical_ich_q1e_single_temp(
    times_months: List[float],
    contents: List[float],
    spec_lower: float = 95.0,
    use_log: bool = False,
    ci_side: str = "one-sided",
) -> Dict:
    """ICH Q1E §B.1 単温度線形回帰 → §2.6 信頼区間下限の交差点.

    Parameters
    ----------
    times_months : list[float]
        測定時間 [月]。t=0 を含めることを推奨。
    contents : list[float]
        含量 [%]
    spec_lower : float
        規格下限 [%](デフォルト 95)
    use_log : bool
        True なら ln C を被説明変数(1 次反応近似)。
        False なら C 自体(0 次反応近似、ICH Q1E §B.1 標準)。
    ci_side : "one-sided" | "two-sided"
        ICH Q1E §2.6 は "one-sided 95%"。
        比較用にベイジアン側(z=1.96)と揃える "two-sided" も指定可能。

    Returns
    -------
    dict
        slope, intercept, r_squared, n, df, residual_sd
        shelf_life_mean_months  : mean fitted curve が spec_lower と交わる時点
        recommended_shelf_life_months : ICH Q1E §2.6 推奨値(CI 下限交差点)
        warnings : List[Dict]  (LOW_R_SQUARED_CLASSICAL 等)
    """
    if len(times_months) != len(contents):
        raise ValueError("times_months と contents の長さが一致しません")
    if len(times_months) < 3:
        raise ValueError(
            "古典 ICH §B.1 線形回帰には最低 3 時点のデータが必要です(信頼区間計算のため df ≥ 1)。"
        )

    times = np.array(times_months, dtype=float)
    cs = np.array(contents, dtype=float)

    if use_log:
        y = np.log(cs)
        spec_y = np.log(spec_lower)
    else:
        y = cs
        spec_y = spec_lower

    slope, intercept, r_val, _, _ = stats.linregress(times, y)
    n = len(times)
    df = n - 2
    t_bar = float(np.mean(times))
    Sxx = float(np.sum((times - t_bar) ** 2))
    y_fit = intercept + slope * times
    ss_res = float(np.sum((y - y_fit) ** 2))
    residual_sd = float(np.sqrt(ss_res / df)) if df > 0 else 0.0
    r_squared = float(r_val ** 2)

    if ci_side == "two-sided":
        t_quantile = float(stats.t.ppf(0.975, df))  # two-sided 95% = one-sided 97.5%
    else:
        t_quantile = float(stats.t.ppf(0.95, df))   # one-sided 95% (ICH Q1E §2.6)

    warnings_list: List[Dict] = []

    # 規格下限到達点(mean fitted curve)
    if slope >= 0:
        # 増加 or 平坦 → spec_lower に到達しない、SL は無限大相当
        return {
            "shelf_life_mean_months": float("inf"),
            "shelf_life_lo95_months": float("inf"),
            "recommended_shelf_life_months": float("inf"),
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "n": n,
            "df": df,
            "residual_sd": round(residual_sd, 6),
            "t_quantile": round(t_quantile, 4),
            "ci_side": ci_side,
            "use_log": use_log,
            "spec_lower": spec_lower,
            "warnings": [{
                "code": "NO_SIGNIFICANT_DEGRADATION",
                "level": "warning",
                "message": (
                    "回帰直線が水平または増加方向のため、規格下限に到達しません。"
                    "Significant degradation 不在のため有効期間を定義できません。"
                ),
                "context": {"slope": slope},
            }],
        }

    sl_mean = (spec_y - intercept) / slope

    # 信頼区間下限(または上限)が spec_y と交わる時点を二分法で求める
    def lower_ci(t: float) -> float:
        se_mean = residual_sd * np.sqrt(1.0 / n + (t - t_bar) ** 2 / Sxx)
        return intercept + slope * t - t_quantile * se_mean

    t_lo, t_hi = 0.0, max(sl_mean * 5.0, 12.0)
    f_lo = lower_ci(t_lo) - spec_y
    f_hi = lower_ci(t_hi) - spec_y
    search_exceeded = False
    if f_lo < 0:
        sl_lo95 = 0.0
    elif f_hi > 0:
        # 探索範囲 [0, max(sl_mean*5, 12)] 内で CI 下限が規格下限と交わらない。
        # sl_lo95 を探索上端 t_hi に張り付かせるが、Rule 6 に従い silent failure
        # を回避するため SHELF_LIFE_SEARCH_EXCEEDED 警告を発出する。
        sl_lo95 = t_hi
        search_exceeded = True
    else:
        for _ in range(100):
            t_mid = 0.5 * (t_lo + t_hi)
            f_mid = lower_ci(t_mid) - spec_y
            if abs(f_mid) < 1e-6:
                break
            if f_mid > 0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        sl_lo95 = 0.5 * (t_lo + t_hi)

    if search_exceeded:
        warnings_list.append({
            "code": "SHELF_LIFE_SEARCH_EXCEEDED",
            "level": "info",
            "message": (
                f"探索範囲 [0, {t_hi:.1f}] ヶ月 内で 95%CI 下限が規格下限と交わりませんでした。"
                "推定値は探索上端に張り付いています。データの分解傾向が弱く、"
                "Significant degradation の前提を満たさない可能性があります。"
            ),
            "context": {"search_upper_months": round(float(t_hi), 1)},
        })

    # R² 警告(古典側でも低 R² は問題)
    if r_squared < 0.9:
        warnings_list.append({
            "code": "LOW_R_SQUARED_CLASSICAL",
            "level": "warning",
            "message": (
                f"線形回帰の決定係数が低い(R² = {r_squared:.3f})。"
                "適用する反応次数(0 次 / 1 次)の選択を見直してください。"
            ),
            "context": {"r_squared": round(r_squared, 4)},
        })

    return {
        "shelf_life_mean_months": round(float(sl_mean), 1),
        "shelf_life_lo95_months": round(float(sl_lo95), 1),
        "recommended_shelf_life_months": round(float(sl_lo95), 1),
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "n": n,
        "df": df,
        "residual_sd": round(residual_sd, 6),
        "t_quantile": round(t_quantile, 4),
        "ci_side": ci_side,
        "use_log": use_log,
        "spec_lower": spec_lower,
        "warnings": warnings_list,
    }

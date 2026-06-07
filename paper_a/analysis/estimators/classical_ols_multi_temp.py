"""推定器 #3: 多温度古典 OLS Arrhenius 外挿 (prior なし、新規最小実装).

経路:
    1. 各温度で OLS により k̂_i を推定 (ln(C/C0) = -k * t に対する単純回帰)
    2. ln k̂_i vs 1/T_i (K) の OLS 回帰で Arrhenius パラメータ (ln A, -Ea/R) を推定
    3. 25°C への外挿で k_25 を算出
    4. t90(25°C) = -ln(0.9) / k_25 (1 次仮定、頻度論)
    5. 95% CI は 2 段目 OLS の予測区間 (ln k_25 の SE × t_quantile) を t に伝播

なぜ新規実装か:
    cmc-platform の `run_bayesian_stability` は prior 必須 (共役回帰).
    `classical_ich_q1e_single_temp` は単温度のみ.両者とも本推定器の構造
    (多温度 OLS、prior なし、頻度論 CI) を実装していない.
    Faya 2018 が比較対象として用いた「単純な古典 OLS」が本推定器に対応する.

頑健性層の扱い:
    本推定器は 1 次反応を仮定して見かけ k を抽出する.頑健性層 (2 次/自触媒/
    誘導期) では真の速度論と推定器の前提が乖離するため、点推定 t90 は
    「見かけ t90」となる.評価側は truth.json の `t90_true_25c_months`
    (真の速度論で算出された真値) に対してバイアスを取る.両者は異なる
    物理量である点を本コメントで明示する.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .base import EstimatorResult

R_GAS = 8.314  # J/(mol·K)
T_25K = 298.15
ESTIMATOR_NAME = "classical_ols_multi_temp"


def _estimate_k_per_temp(
    times: np.ndarray,
    contents: np.ndarray,
    initial_content: float,
) -> tuple[float, float, float] | None:
    """1 温度の OLS: ln(C/C0) = -k * t.

    Returns
    -------
    (k_hat, k_se, r_squared) or None if estimation failed (data unusable).
    """
    if len(times) < 2:
        return None
    if np.any(contents <= 0):
        return None
    t0_idx = int(np.argmin(times))
    c0 = float(contents[t0_idx]) if times[t0_idx] == 0 else initial_content
    y = np.log(contents / c0)
    slope, _, r_val, _, se = stats.linregress(times, y)
    k_hat = -float(slope)
    if k_hat <= 0:
        # 非有意分解.以降の ln(k) が定義不能 → 失敗.
        return None
    # SE の floor は cmc-platform 既存運用と同じ k_hat * 0.05.
    k_se = max(float(se), k_hat * 0.05)
    return k_hat, k_se, float(r_val ** 2)


def estimate(
    data_rows: list[dict],
    case_id: str,
    replicate_id: int,
    spec_lower: float = 90.0,
    initial_content: float = 100.0,
    target_temp_c: float = 25.0,
    column_names: dict | None = None,
) -> EstimatorResult:
    """多温度古典 OLS Arrhenius 外挿で t90(target_temp_c) を推定する.

    Parameters
    ----------
    data_rows : list of dict
        DataRow 規約 (temperature, time_months, content_percent).
        case_id / replicate_id 等は無視 (推定器は raw data のみ受ける).
    spec_lower : float
        主指標は 90.0 (t90 = content 90% 到達時間).
    target_temp_c : float
        外挿先温度 (25°C 固定、Faya Fig 8 整合).

    Returns
    -------
    EstimatorResult
        - 成功時: t90 点推定 + 95% CI、converged=True、error_code=None
        - 失敗時: t90=None、error_code に理由 ("INSUFFICIENT_TEMPERATURES"
          または "OTHER")
    """
    cols = column_names or {
        "temperature": "temperature",
        "time": "time_months",
        "response": "content_percent",
    }

    # 温度別グループ化
    groups: dict[float, list] = {}
    for row in data_rows:
        T_c = float(row[cols["temperature"]])
        t = float(row[cols["time"]])
        C = float(row[cols["response"]])
        groups.setdefault(T_c, []).append((t, C))

    n_t = len(groups)

    # 各温度で OLS により k̂_i を取る
    k_hats: list[float] = []
    k_ses: list[float] = []
    T_Ks: list[float] = []
    per_temp_r2: dict[str, float] = {}

    for T_c, points in sorted(groups.items()):
        times = np.array([p[0] for p in points], dtype=float)
        contents = np.array([p[1] for p in points], dtype=float)
        result = _estimate_k_per_temp(times, contents, initial_content)
        if result is None:
            continue
        k_hat, k_se, r2 = result
        k_hats.append(k_hat)
        k_ses.append(k_se)
        T_Ks.append(T_c + 273.15)
        per_temp_r2[f"{T_c:.1f}"] = r2

    diagnostics: dict[str, Any] = {
        "n_t_observed": n_t,
        "n_t_usable": len(k_hats),
        "per_temp_r_squared": per_temp_r2,
    }

    # 2 段目 OLS には少なくとも n=3 必要 (slope/intercept + 残差 df ≥1).
    # n=2 でも slope/intercept は解けるが SE=0 になり CI が縮退する → 失敗扱い.
    if len(k_hats) < 3:
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code="INSUFFICIENT_TEMPERATURES",
            diagnostics=diagnostics,
            spec_lower_used=spec_lower,
        )

    # 2 段目 OLS: ln k vs 1/T
    x = 1.0 / np.array(T_Ks)
    y = np.log(np.array(k_hats))

    slope2, intercept2, r2_val, _, se_slope2 = stats.linregress(x, y)
    n2 = len(y)
    df2 = n2 - 2

    # 残差分散
    y_fit = intercept2 + slope2 * x
    ss_res = float(np.sum((y - y_fit) ** 2))
    x_mean = float(np.mean(x))
    Sxx = float(np.sum((x - x_mean) ** 2))
    sigma2 = ss_res / df2 if df2 > 0 else 0.0
    residual_sd = float(np.sqrt(sigma2))

    # 25°C 外挿
    x_pred = 1.0 / (target_temp_c + 273.15)
    ln_k_pred = float(intercept2 + slope2 * x_pred)
    se_ln_k = float(np.sqrt(sigma2 * (1.0 / n2 + (x_pred - x_mean) ** 2 / Sxx))) if df2 > 0 and Sxx > 0 else 0.0

    t_quantile = float(stats.t.ppf(0.975, df2)) if df2 > 0 else 0.0  # 両側 95%

    # delta 法で t90 の点推定 + CI: t90 = -ln(spec/C0) / k
    log_ratio = -np.log(spec_lower / initial_content)  # 正の値、spec_lower=90 のとき 0.10536
    k_pred = float(np.exp(ln_k_pred))
    t90_point = float(log_ratio / k_pred)

    # ln k_25 の CI (頻度論 prediction interval, mean response):
    #   ln_k_25 ∈ [ln_k_pred - t_q * se_ln_k, ln_k_pred + t_q * se_ln_k]
    # k_25 ∈ [exp(ln_k_pred - t_q * se), exp(ln_k_pred + t_q * se)]
    # t90 ∝ 1/k なので、t90 の CI は k の CI を反転:
    #   t90_lo95 = log_ratio / k_hi95 (k が大きい→ t90 が短い → 下限)
    #   t90_hi95 = log_ratio / k_lo95
    ln_k_hi = ln_k_pred + t_quantile * se_ln_k
    ln_k_lo = ln_k_pred - t_quantile * se_ln_k
    t90_lo = float(log_ratio / np.exp(ln_k_hi))
    t90_hi = float(log_ratio / np.exp(ln_k_lo))

    ea_kj_estimated = float(-slope2 * R_GAS / 1000.0)
    ln_a_estimated = float(intercept2)

    diagnostics.update({
        "ea_kj_estimated": ea_kj_estimated,
        "ln_a_estimated": ln_a_estimated,
        "r_squared_arrhenius": float(r2_val ** 2),
        "se_ln_k_at_target": se_ln_k,
        "residual_sd": residual_sd,
        "df_arrhenius": df2,
        "t_quantile_975": t_quantile,
        "k_pred_at_target": k_pred,
        "ln_k_pred_at_target": ln_k_pred,
        "target_temp_c": target_temp_c,
    })

    return EstimatorResult(
        estimator_name=ESTIMATOR_NAME,
        case_id=case_id,
        replicate_id=replicate_id,
        t90_point_estimate_months=t90_point,
        t90_lo95_months=t90_lo,
        t90_hi95_months=t90_hi,
        converged=True,
        error_code=None,
        diagnostics=diagnostics,
        spec_lower_used=spec_lower,
    )

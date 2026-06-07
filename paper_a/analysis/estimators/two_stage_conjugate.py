"""推定器 #1: 2 段 OLS / 共役回帰 (cmc-platform `run_bayesian_stability` ラッパ).

実装方針:
    `run_bayesian_stability` を spec_lower=90.0, storage_temps=[25.0] で呼び、
    storage_results[0] (=25°C) から t90 を抽出する.再実装禁止.

n_T<3 失敗扱い:
    cmc-platform は ValueError("最低 3 温度") を投げる (規制 hard fail).
    Wrapper はこれを捕捉して `converged=False, error_code="N_CONDS_TOO_LOW"`
    として記録する.これが論文の知見になる:
    「n_T=2 cell では 2 段 OLS/共役回帰は規制上適用不可、MCMC は動く」

case 別 prior:
    truth.json の `prior_ea_kj_mol` / `prior_ea_sd_kj_mol` を関数引数で受け、
    accurate/moderate/strong の 3 軸で生成器の設定を解析にそのまま反映する.
    生成器と解析の prior 値乖離を防ぐため `test_prior_parity` で保証.
"""
from __future__ import annotations

from typing import Any

from paper_a.vendor.bayesian_stability import (
    StabilityHardFailWarning,
    run_bayesian_stability,
)

from .base import EstimatorResult

ESTIMATOR_NAME = "two_stage_conjugate"


def estimate(
    data_rows: list[dict],
    case_id: str,
    replicate_id: int,
    prior_ea_kj: float,
    prior_ea_sd_kj: float,
    spec_lower: float = 90.0,
    initial_content: float = 100.0,
    target_temp_c: float = 25.0,
    column_names: dict | None = None,
) -> EstimatorResult:
    """加速試験データに 2 段 OLS/共役回帰を適用して t90(25°C) を推定.

    Parameters
    ----------
    data_rows : list of dict
        DataRow 規約.加速試験データのみ (25°C 長期は渡さない).
    prior_ea_kj, prior_ea_sd_kj : float
        truth.json から case 別に渡される (Prior 正確性軸を反映).
    spec_lower : float
        90.0 (t90、主指標).
    target_temp_c : float
        外挿先温度.通常 25°C 固定.
    """
    cols = column_names or {
        "temperature": "temperature",
        "time": "time_months",
        "response": "content_percent",
    }

    # cmc-platform が期待する形式に列名を正規化
    norm_rows = [
        {
            "temperature": float(r[cols["temperature"]]),
            "time_months": float(r[cols["time"]]),
            "content_percent": float(r[cols["response"]]),
        }
        for r in data_rows
    ]

    try:
        out = run_bayesian_stability(
            data_rows=norm_rows,
            storage_temps=[target_temp_c],
            spec_lower=spec_lower,
            initial_content=initial_content,
            prior_ea_kj=prior_ea_kj,
            prior_ea_sd_kj=prior_ea_sd_kj,
        )
    except ValueError as e:
        # n_T<3 または k_hat>0 を満たす温度<3 で発生 (cmc-platform L286-289).
        msg = str(e)
        code = "N_CONDS_TOO_LOW" if "最低" in msg or "n_conds" in msg.lower() else "OTHER"
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code=code,
            diagnostics={"exception_class": "ValueError", "message": msg},
            spec_lower_used=spec_lower,
        )
    except StabilityHardFailWarning as e:
        # PQ_N_POINTS_TOO_LOW / PRIOR_DATA_INCONSISTENCY 等 (B4).
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code=e.code,
            diagnostics={"exception_class": "StabilityHardFailWarning", "message": e.message, "detail": e.detail},
            spec_lower_used=spec_lower,
        )

    storage = out.get("storage_results", [])
    target_row = next((s for s in storage if abs(float(s.get("storage_temp_c", -999)) - target_temp_c) < 1e-9), None)

    if target_row is None:
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code="OTHER",
            diagnostics={"message": f"no storage_result for {target_temp_c}°C", "raw": out},
            spec_lower_used=spec_lower,
        )

    diagnostics: dict[str, Any] = {
        "ea_posterior_kj_mol": out.get("ea_posterior_kj_mol"),
        "ea_posterior_sd_kj_mol": out.get("ea_posterior_sd_kj_mol"),
        "ln_a_posterior": out.get("ln_a_posterior"),
        "r_squared_arrhenius": out.get("r_squared_arrhenius"),
        "n_conditions": out.get("n_conditions"),
        "k_mean_per_month": target_row.get("k_mean_per_month"),
        "capped": target_row.get("capped"),
        "warnings": out.get("warnings", []),
        "prior_ea_kj_used": prior_ea_kj,
        "prior_ea_sd_kj_used": prior_ea_sd_kj,
    }

    return EstimatorResult(
        estimator_name=ESTIMATOR_NAME,
        case_id=case_id,
        replicate_id=replicate_id,
        t90_point_estimate_months=float(target_row.get("shelf_life_mean_months", 0.0)),
        t90_lo95_months=float(target_row.get("shelf_life_lo95_months", 0.0)),
        t90_hi95_months=float(target_row.get("shelf_life_hi95_months", 0.0)),
        converged=True,
        error_code=None,
        diagnostics=diagnostics,
        spec_lower_used=spec_lower,
    )

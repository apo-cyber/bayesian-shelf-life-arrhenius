"""推定器 #4: 古典 ICH Q1E §B.1 (単温度線形回帰、25°C 長期試験データ).

実装方針:
    cmc-platform `classical_ich_q1e_single_temp` を薄くラップする (再実装禁止).
    入力は 25°C 長期試験データ (n_points=3/4/6 × 36 ヶ月、generate.py が
    long_term_25c.csv に出力).加速試験データには適用しない.
    これにより他 3 推定器 (25°C への外挿 t90) と物理量が一致し、Faya 2018
    Fig 4.5 形式の apples-to-apples 比較が成立する (再開プロンプト判断 1).

論文での記述:
    ICH Q1E §B.1 は 25°C 長期試験データに対する単温度線形回帰.加速試験
    データを扱う他 3 推定器とは入力データが異なる ("規制実務での標準手法
    を 25°C 長期データに適用したベースライン").Methods に明記すること.

ci_side = "two-sided" を採用 (Faya Fig 4.5 整合の 95% CI 比較).
ICH Q1E §2.6 規定は one-sided 95% だが、本論文の主指標は推定値の 95%
中心区間を 4 推定器で揃えるため two-sided を採用する旨を Discussion で明記.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from paper_a.vendor.classical_stability import (
    classical_ich_q1e_single_temp,
)

from .base import EstimatorResult

ESTIMATOR_NAME = "classical_ich_q1e"


def estimate(
    long_term_25c_rows: list[dict],
    case_id: str,
    replicate_id: int,
    spec_lower: float = 90.0,
    initial_content: float = 100.0,
    use_log: bool = True,
    ci_side: str = "two-sided",
    column_names: dict | None = None,
) -> EstimatorResult:
    """25°C 長期試験データに ICH Q1E §B.1 を適用して t90(25°C) を推定する.

    Parameters
    ----------
    long_term_25c_rows : list of dict
        DataRow 規約 (`temperature`/`time_months`/`content_percent`).
        全行が temperature=25.0 であることを想定 (paper_a/data/*/long_term_25c.csv).
    spec_lower : float
        90.0 (t90 = content 90% 到達時間、主指標).
    use_log : bool
        True で ln(C) 線形回帰 (1 次反応近似).False は 0 次反応相当 (ICH §B.1 標準).
        合成データは 1 次反応で生成しているため True が適切.
    ci_side : str
        "two-sided" で Faya Fig 4.5 整合の 95% CI.

    Returns
    -------
    EstimatorResult
        失敗時は error_code に "NO_SIGNIFICANT_DEGRADATION" 等を入れる.
    """
    cols = column_names or {
        "temperature": "temperature",
        "time": "time_months",
        "response": "content_percent",
    }

    times = [float(r[cols["time"]]) for r in long_term_25c_rows]
    contents = [float(r[cols["response"]]) for r in long_term_25c_rows]

    try:
        out = classical_ich_q1e_single_temp(
            times_months=times,
            contents=contents,
            spec_lower=spec_lower,
            use_log=use_log,
            ci_side=ci_side,
        )
    except ValueError as e:
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code="OTHER",
            diagnostics={"exception_class": type(e).__name__, "message": str(e)},
            spec_lower_used=spec_lower,
        )

    warns = out.get("warnings", []) or []
    error_code: str | None = None
    point = out.get("shelf_life_mean_months")
    lo = out.get("shelf_life_lo95_months")

    # NO_SIGNIFICANT_DEGRADATION → 推定不能 (slope >= 0)
    if any(w.get("code") == "NO_SIGNIFICANT_DEGRADATION" for w in warns):
        error_code = "NO_SIGNIFICANT_DEGRADATION"
        return EstimatorResult(
            estimator_name=ESTIMATOR_NAME,
            case_id=case_id,
            replicate_id=replicate_id,
            t90_point_estimate_months=None,
            t90_lo95_months=None,
            t90_hi95_months=None,
            converged=False,
            error_code=error_code,
            diagnostics={
                "slope": out.get("slope"),
                "intercept": out.get("intercept"),
                "r_squared": out.get("r_squared"),
                "warnings": warns,
            },
            spec_lower_used=spec_lower,
        )

    # 上限 CI は cmc-platform の単温度実装が片側 lower のみ返すため、
    # 同じ漸近性で対称片を構築する: ln k の予測区間を反転して t90 の hi95.
    # ただし実装が hi95 を返していない場合は None.
    # 現 classical_ich_q1e_single_temp の戻り値に hi95 がなく、本推定器の
    # 主指標は lo95 (保守的下限) のみで十分.実装の単純性のため hi95 は None.
    hi: float | None = None

    return EstimatorResult(
        estimator_name=ESTIMATOR_NAME,
        case_id=case_id,
        replicate_id=replicate_id,
        t90_point_estimate_months=float(point) if point is not None else None,
        t90_lo95_months=float(lo) if lo is not None else None,
        t90_hi95_months=hi,
        converged=True,
        error_code=None,
        diagnostics={
            "slope": out.get("slope"),
            "intercept": out.get("intercept"),
            "r_squared": out.get("r_squared"),
            "n": out.get("n"),
            "df": out.get("df"),
            "residual_sd": out.get("residual_sd"),
            "t_quantile": out.get("t_quantile"),
            "ci_side": out.get("ci_side"),
            "use_log": out.get("use_log"),
            "warnings": warns,
        },
        spec_lower_used=spec_lower,
    )

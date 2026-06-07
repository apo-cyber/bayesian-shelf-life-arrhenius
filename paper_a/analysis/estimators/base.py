"""EstimatorResult 共通契約.

論文中・コードの呼称は ESTIMATOR_NAMES で固定.Methods / Results / 図凡例で
表記揺れを起こさないこと.

全推定器は spec_lower=90.0 (t90 = content 90% 到達時間) を主指標として算出する.
これにより truth.json の `t90_true_25c_months` (Faya Fig 8 整合) と直接
バイアス比較が成立する.補助で spec_lower=95.0 も走らせる場合は
`spec_lower_used` フィールドで識別、デフォルト集計・図は 90.0 のみ参照する.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# 4 推定器の固定呼称 (確定済、再実装禁止).
ESTIMATOR_NAMES: dict[str, str] = {
    "two_stage_conjugate": "Two-stage OLS / conjugate regression",
    "mcmc": "Full Bayesian (MCMC)",
    "classical_ols_multi_temp": "Classical multi-temperature OLS (no prior)",
    "classical_ich_q1e": "Classical ICH Q1E §B.1 (single-temperature, 25°C long-term)",
}


@dataclass
class EstimatorResult:
    """1 推定器 × 1 case × 1 replicate の出力 1 行.

    Notes
    -----
    - 主指標は t90 (content 90% 到達時間).評価は truth.json の
      `t90_true_25c_months` に対して行う.
    - error_code:
      * "N_CONDS_TOO_LOW": cmc-platform `run_bayesian_stability` が n_T<3 で
        ValueError を出した場合 (規制 hard fail).two_stage_conjugate で発生.
      * "PRIOR_DATA_INCONSISTENCY": StabilityHardFailWarning.同上.
      * "PQ_N_POINTS_TOO_LOW": 同上.
      * "MCMC_NOT_CONVERGED": R-hat ≥ 1.01 または ESS < 400.
      * "NO_SIGNIFICANT_DEGRADATION": classical_ich_q1e で slope ≥ 0.
      * "INSUFFICIENT_TEMPERATURES": classical_ols_multi_temp で n_T<3.
      * "OTHER": 上記以外の予期しない例外 (diagnostics に詳細).
    - converged は MCMC 用診断.他の推定器では「推定成功 (= error_code is None)」
      と同義として True を入れる.
    """

    estimator_name: str
    case_id: str
    replicate_id: int
    t90_point_estimate_months: float | None
    t90_lo95_months: float | None
    t90_hi95_months: float | None
    converged: bool
    error_code: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    spec_lower_used: float = 90.0

    def __post_init__(self) -> None:
        if self.estimator_name not in ESTIMATOR_NAMES:
            raise ValueError(
                f"estimator_name '{self.estimator_name}' は ESTIMATOR_NAMES 固定 4 種"
                f" {list(ESTIMATOR_NAMES)} のいずれかでなければならない"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

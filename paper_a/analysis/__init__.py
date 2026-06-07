"""4 推定器 × 3 指標の解析パイプライン.

主指標は t90_true_25c_months ベース (Faya 2018 Fig 8 整合).
sl_at_spec_true_25c_months は実務指標として truth に保持されるが、論文評価の
主軸ではない.
"""

from .estimators.base import EstimatorResult, ESTIMATOR_NAMES

__all__ = ["EstimatorResult", "ESTIMATOR_NAMES"]

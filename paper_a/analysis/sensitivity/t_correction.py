"""t 補正版 2 段 OLS (感度解析、Discussion 補遺、本セッション実装しない).

確定方針 (元プロンプト §B、本セッション §A):
    Q1 案 C 採用に伴い t 補正版 2 段 OLS は本番比較から外し、Discussion 補遺
    用に本ファイルで退避保持する.実装は将来必要時に `mcmc_benchmark.py::
    two_stage_fit(use_t_correction=True)` を lift する.

論文での位置づけ:
    Discussion で 「σ² 周辺化版という中間的構成も理論上ありうる」 旨を一文
    触れる素材.主証拠 (本論文中核) ではない.
"""
from __future__ import annotations


def estimate_t_corrected(*args, **kwargs):
    """将来実装用プレースホルダ (mcmc_benchmark.py::two_stage_fit から lift)."""
    raise NotImplementedError(
        "t 補正版 2 段 OLS は本論文本番比較対象外.必要時に "
        "docs/audit/mcmc_benchmark.py::two_stage_fit(use_t_correction=True) を lift."
    )

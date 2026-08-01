"""supp_table_s1.py が Supplementary Table S1 を再現することを固定するテスト.

S1 の 27 値 (n_T 3 水準 × prior 3 水準 × σ 3 水準) と、脚注が主張する
Figure 3 との整合 9 値を逐語で持つ。原稿は小数第 1 位まで表示するため、
丸め後に一致すればよい。

parquet (35 MB) を読むため slow マーカーを付ける。
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from paper_a.analysis.supp_table_s1 import (  # noqa: E402
    PRIOR_LEVELS,
    RESULTS_DIR,
    SIGMA_LEVELS,
    compute_s1,
)

# 原稿 S1: (n_T, prior) -> (σ=0.01, σ=0.02, σ=0.05) の非収束率 [%]
PAPER_S1 = {
    (2, "accurate"): (57.3, 61.7, 79.3),
    (2, "moderate"): (62.0, 63.0, 80.0),
    (2, "strong"): (63.0, 68.0, 82.7),
    (3, "accurate"): (30.3, 29.0, 66.3),
    (3, "moderate"): (29.3, 30.0, 69.3),
    (3, "strong"): (34.0, 37.3, 64.3),
    (4, "accurate"): (17.0, 11.0, 11.7),
    (4, "moderate"): (21.3, 12.0, 10.7),
    (4, "strong"): (24.0, 17.7, 13.7),
}

# 脚注 "Consistency with Figure 3": n_T -> (accurate, moderate, strong)
PAPER_FIGURE3_MARGINALS = {
    2: (66.1, 68.3, 71.2),
    3: (41.9, 42.9, 45.2),
    4: (13.2, 14.7, 18.4),
}


@pytest.fixture(scope="module")
def s1_rows() -> dict:
    if not (RESULTS_DIR / "estimator_results.parquet").exists():
        pytest.skip("estimator_results.parquet がない")
    payload = compute_s1()
    return {(r["n_t"], r["prior_accuracy"]): r for r in payload["rows"]}


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(PAPER_S1))
def test_matches_paper_s1(key: tuple[int, str], s1_rows: dict) -> None:
    row = s1_rows[key]
    for sigma, expected in zip(SIGMA_LEVELS, PAPER_S1[key]):
        got = row["by_sigma"][f"{sigma}"]["nonconvergence_pct"]
        assert round(got, 1) == pytest.approx(expected, abs=0.05), f"σ={sigma}"


@pytest.mark.slow
def test_cell_sizes_are_300(s1_rows: dict) -> None:
    """S1 の説明文が述べる「3 case × 100 replicate = 300 runs」を固定する."""
    for key, row in s1_rows.items():
        for sigma in SIGMA_LEVELS:
            assert row["by_sigma"][f"{sigma}"]["n_replicates"] == 300, f"{key} σ={sigma}"


@pytest.mark.slow
def test_figure3_marginals(s1_rows: dict) -> None:
    """脚注が主張する Figure 3 との整合 (σ 周辺化) を固定する."""
    for n_t, expected_triple in PAPER_FIGURE3_MARGINALS.items():
        got = [
            s1_rows[(n_t, prior)]["marginalised_over_sigma"]["nonconvergence_pct"]
            for prior in PRIOR_LEVELS
        ]
        for g, e in zip(got, expected_triple):
            assert round(g, 1) == pytest.approx(e, abs=0.05), f"n_T={n_t}"

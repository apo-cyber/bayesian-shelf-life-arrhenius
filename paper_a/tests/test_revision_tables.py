"""改訂で加えた表(Table 4 の拡張、Supplementary Table S2・S3・S3b)を固定するテスト.

原稿に印刷された値を逐語で持ち、リポジトリに収めた集計結果
(results/tuning_validation/tuning_bias.json、results/stage1_propagation.json、
results/cap_sensitivity.json)がその値に丸められることを確かめる。

集計を再実行しないのは、各スクリプト(aggregate_bias.py、stage1_propagation.py、
cap_threshold.py)が実行のたびに結果ファイルを書き直すため。再実行で同じ値が
出ることは v1.1.0 の作成時に確かめてある(README の Release history)。

許容幅は各列の表示桁の半単位(丸めの向きによらず一致とみなす)。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load(rel: str) -> dict:
    path = RESULTS / rel
    if not path.exists():
        pytest.skip(f"{rel} がない")
    return json.loads(path.read_text())


def _close(got: float, expected: float, decimals: int) -> bool:
    return abs(got - expected) <= 0.5 * 10 ** (-decimals) + 1e-9


# --------------------------------------------------------------------------
# Table 4: cell, configuration -> (non-conv. %, n conv., median bias, IQR, coverage %)
# "Two-stage" の非収束列は失敗率(0.0 = fail なし)。
PAPER_TABLE4 = {
    ("nt3_strong", "A"): (30.0, 84, 46.2, 125, 82.1),
    ("nt3_strong", "B"): (12.5, 105, 48.6, 126, 82.9),
    ("nt3_strong", "C"): (21.7, 94, 48.9, 130, 81.9),
    ("nt3_strong", "D"): (16.7, 100, 57.8, 118, 83.0),
    ("nt3_strong", "E"): (2.5, 117, 48.2, 123, 82.9),
    ("nt3_strong", "two_stage"): (0.0, 120, -7.0, 64, 84.2),
    ("nt2_strong", "A"): (70.8, 35, 192.1, 279, 91.4),
    ("nt2_strong", "B"): (42.5, 69, 172.1, 179, 85.5),
    ("nt2_strong", "C"): (68.3, 38, 187.6, 385, 94.7),
    ("nt2_strong", "D"): (55.0, 54, 174.8, 446, 81.5),
    ("nt2_strong", "E"): (19.2, 97, 160.9, 202, 87.6),
    ("nt2_accurate", "A"): (65.0, 42, 54.3, 107, 100.0),
    ("nt2_accurate", "E"): (13.3, 104, 52.5, 102, 99.0),
    ("nt4_strong", "A"): (20.0, 96, 6.0, 40, 80.2),
    ("nt4_strong", "E"): (0.0, 120, 5.5, 38, 81.7),
    ("nt4_strong", "two_stage"): (0.0, 120, -6.0, 33, 93.3),
    ("nt3_accurate", "A"): (16.7, 100, 16.2, 64, 91.0),
    ("nt3_accurate", "E"): (0.8, 119, 17.2, 59, 91.6),
    ("nt3_accurate", "two_stage"): (0.0, 120, -17.1, 30, 93.3),
}
# n_T = 2 では二段推定量は全例失敗(表では "100 (fail)")
PAPER_TABLE4_TWO_STAGE_FAIL = ("nt2_strong", "nt2_accurate")


@pytest.mark.parametrize("key", sorted(PAPER_TABLE4))
def test_table4(key: tuple[str, str]) -> None:
    data = _load("tuning_validation/tuning_bias.json")
    cell, conf = key
    nonconv, n, med, iqr, cov = PAPER_TABLE4[key]
    if conf == "two_stage":
        row = data["two_stage_conjugate"][cell]
        rate = row["failure_rate"]
    else:
        row = data["mcmc"][cell][conf]
        rate = row["nonconv_rate"]
    assert _close(100 * rate, nonconv, 1)
    assert row["n"] == n
    assert _close(row["bias_median"], med, 1)
    assert _close(row["bias_iqr"], iqr, 0)
    assert _close(100 * row["coverage_probability"], cov, 1)


@pytest.mark.parametrize("cell", PAPER_TABLE4_TWO_STAGE_FAIL)
def test_table4_two_stage_fails_at_two_temperatures(cell: str) -> None:
    row = _load("tuning_validation/tuning_bias.json")["two_stage_conjugate"][cell]
    assert row["failure_rate"] == 1.0
    assert row["n"] == 0


# --------------------------------------------------------------------------
# Supplementary Table S2: variant -> (n, median bias, IQR, MAD, SD cap 120,
#                                     optim. %, cov. %, log-width, > 120 %)
PAPER_S2 = {
    "published": (8918, -2.66, 47.0, 32.1, 32.46, 47, 90.2, 2.30, 14.5),
    "no_floor": (8918, -2.76, 46.9, 32.4, 32.54, 47, 89.6, 2.31, 14.4),
    "unweighted": (8918, 0.68, 56.0, 37.1, 34.23, 51, 89.7, 2.35, 18.1),
    "mc_propagation": (8918, -1.90, 42.0, 29.6, 30.70, 47, 97.9, 3.10, 12.1),
    "mcmc_joint": (493, 36.42, 129.1, 69.1, 31.41, 73, 82.6, 1.94, 40.4),
}


@pytest.mark.parametrize("variant", sorted(PAPER_S2))
def test_supp_table_s2(variant: str) -> None:
    data = _load("stage1_propagation.json")
    assert data["n_replicates"] == 8918
    assert data["B_draws"] == 400
    row = data["variants"][variant]
    n, med, iqr, mad, sd120, opt, cov, lw, above = PAPER_S2[variant]
    assert row["n"] == n
    assert _close(row["bias_median"], med, 2)
    assert _close(row["bias_iqr"], iqr, 1)
    assert _close(row["bias_mad"], mad, 1)
    assert _close(row["bias_sd_capped120"], sd120, 2)
    assert _close(100 * row["pp_optimism_rate"], opt, 0)
    assert _close(100 * row["coverage_probability"], cov, 1)
    assert _close(row["median_interval_width_log"], lw, 2)
    assert _close(100 * row["frac_above_120"], above, 1)


# 本文の記述: n_pts = 3 で被覆 84.8 %、n_pts = 6 で 95.4 %(published)
@pytest.mark.parametrize("n_pts, cov", [("3", 84.8), ("6", 95.4)])
def test_supp_s2_coverage_by_n_points(n_pts: str, cov: float) -> None:
    row = _load("stage1_propagation.json")["by_n_points"][n_pts]["published"]
    assert _close(100 * row["coverage_probability"], cov, 1)


# --------------------------------------------------------------------------
# Supplementary Table S3 (central cell): estimator -> (n, SD cap 60, SD cap 120,
#   SD cap 240, SD uncapped, median bias, IQR, MAD, P5–P95)
# SD uncapped は原稿で有効数字 2 桁(a × 10^k)、mcmc は倍精度で溢れて ∞。
PAPER_S3 = {
    "two_stage_conjugate": (8918, 14.4, 32.46, 60.5, 6.0e4, -2.66, 47.0, 32.1, 3878),
    "mcmc": (493, 7.8, 31.41, 73.9, math.inf, 36.42, 129.1, 69.1, 823),
    "classical_ols_multi_temp": (8165, 16.5, 36.87, 67.4, 1.5e6, -5.10, 64.9, 40.9, 529),
    "classical_ich_q1e": (8452, 8.5, 23.02, 39.4, 1.4e3, -1.42, 23.8, 16.8, 116),
}

# Supplementary Table S3b: (cell, estimator) -> (n, SD cap 120, SD cap 240,
#   median bias, IQR, MAD, P5–P95)
PAPER_S3B = {
    ("core|n_t=3|prior=accurate", "two_stage_conjugate"): (8941, 32.25, 61.6, -12.85, 39.6, 27.8, 2508),
    ("core|n_t=3|prior=accurate", "mcmc"): (523, 31.80, 61.8, 14.88, 67.9, 43.8, 325),
    ("core|n_t=3|prior=accurate", "classical_ols_multi_temp"): (8108, 36.92, 67.1, -5.14, 65.4, 40.9, 515),
    ("core|n_t=3|prior=accurate", "classical_ich_q1e"): (8400, 23.02, 40.0, -1.12, 22.9, 16.3, 119),
    ("core|n_t=4|prior=strong", "two_stage_conjugate"): (8995, 28.14, 52.0, -3.79, 31.3, 22.9, 328),
    ("core|n_t=4|prior=strong", "mcmc"): (734, 29.13, 62.5, 12.09, 60.0, 32.4, 416),
    ("core|n_t=4|prior=strong", "classical_ols_multi_temp"): (8975, 33.45, 59.8, -3.59, 52.5, 34.2, 319),
    ("core|n_t=4|prior=strong", "classical_ich_q1e"): (8432, 22.99, 40.5, -1.42, 23.7, 16.6, 123),
    ("robustness_all", "two_stage_conjugate"): (17220, 37.09, 60.3, -0.13, 64.2, 41.5, 293),
    ("robustness_all", "mcmc"): (1497, 32.91, 60.1, 44.73, 83.0, 60.2, 293),
    ("robustness_all", "classical_ols_multi_temp"): (19044, 42.96, 89.3, 44.85, 214.6, 107.1, 1247),
    ("robustness_all", "classical_ich_q1e"): (19821, 23.00, 36.8, 5.00, 27.5, 19.2, 104),
}


def _s3_row(cell: str, estimator: str) -> dict:
    data = _load("cap_sensitivity.json")
    return data["cells"][cell][estimator]["uncapped_point_estimate"]


@pytest.mark.parametrize("estimator", sorted(PAPER_S3))
def test_supp_table_s3(estimator: str) -> None:
    row = _s3_row("core|n_t=3|prior=strong", estimator)
    n, sd60, sd120, sd240, sd_inf, med, iqr, mad, p595 = PAPER_S3[estimator]
    assert row["n"] == n
    assert _close(row["bias_sd_capped60"], sd60, 1)
    assert _close(row["bias_sd_capped120"], sd120, 2)
    assert _close(row["bias_sd_capped240"], sd240, 1)
    if math.isinf(sd_inf):
        assert row["bias_sd_uncapped"] is None or math.isinf(row["bias_sd_uncapped"])
    else:
        exponent = math.floor(math.log10(sd_inf))
        assert _close(row["bias_sd_uncapped"] / 10**exponent, sd_inf / 10**exponent, 1)
    assert _close(row["bias_median"], med, 2)
    assert _close(row["bias_iqr"], iqr, 1)
    assert _close(row["bias_mad"], mad, 1)
    assert _close(row["bias_p5_p95_range"], p595, 0)


@pytest.mark.parametrize("key", sorted(PAPER_S3B))
def test_supp_table_s3b(key: tuple[str, str]) -> None:
    row = _s3_row(*key)
    n, sd120, sd240, med, iqr, mad, p595 = PAPER_S3B[key]
    assert row["n"] == n
    assert _close(row["bias_sd_capped120"], sd120, 2)
    assert _close(row["bias_sd_capped240"], sd240, 1)
    assert _close(row["bias_median"], med, 2)
    assert _close(row["bias_iqr"], iqr, 1)
    assert _close(row["bias_mad"], mad, 1)
    assert _close(row["bias_p5_p95_range"], p595, 0)


# 二段推定量が実装上 120 か月で止まっていた割合(S3 の本文: 14.5 %)
def test_supp_s3_two_stage_capped_fraction() -> None:
    row = _load("cap_sensitivity.json")["cells"]["core|n_t=3|prior=strong"]["two_stage_conjugate"]
    assert _close(100 * row["as_stored"]["frac_at_or_above_120"], 14.5, 1)

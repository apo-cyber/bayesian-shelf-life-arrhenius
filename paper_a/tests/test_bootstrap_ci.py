"""bootstrap_ci モジュールのテスト.

事前に `python -m paper_a.analysis.bootstrap_ci` で
paper_a/results/bootstrap_ci.json を生成しておくこと。
"""
import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BOOTSTRAP_PATH = RESULTS_DIR / "bootstrap_ci.json"
CELL_METRICS_PATH = RESULTS_DIR / "cell_metrics.json"

ESTIMATORS = [
    "two_stage_conjugate",
    "mcmc",
    "classical_ols_multi_temp",
    "classical_ich_q1e",
]


@pytest.fixture(scope="module")
def boot():
    if not BOOTSTRAP_PATH.exists():
        pytest.skip("bootstrap_ci.json 未生成。python -m paper_a.analysis.bootstrap_ci を実行。")
    with BOOTSTRAP_PATH.open() as f:
        return json.load(f)


def test_output_exists():
    assert BOOTSTRAP_PATH.exists(), (
        "bootstrap_ci.json が無い。python -m paper_a.analysis.bootstrap_ci を実行。"
    )


def test_structure(boot):
    for key in ("cell", "true_t90_25C_months", "metrics", "bootstrap_config", "case_ids"):
        assert key in boot, f"missing top-level key: {key}"
    assert len(boot["case_ids"]) == 9, "central cell は 9 case (3 grid × 3 noise) のはず"
    for est in ESTIMATORS:
        assert est in boot["metrics"], f"missing estimator: {est}"
        for metric in ("bias_median", "bias_sd_capped120"):
            m = boot["metrics"][est][metric]
            for k in ("point", "ci_low", "ci_high"):
                assert k in m, f"{est}/{metric}: missing {k}"


def test_ci_ordering(boot):
    """CI 下限 ≤ 点推定 ≤ CI 上限 を各推定器・各指標で確認。"""
    for est in ESTIMATORS:
        for metric in ("bias_median", "bias_sd_capped120"):
            m = boot["metrics"][est][metric]
            if m["point"] is None:
                continue
            assert m["ci_low"] <= m["point"] <= m["ci_high"], (
                f"{est}/{metric}: CI 順序が不正 "
                f"({m['ci_low']} ≤ {m['point']} ≤ {m['ci_high']})"
            )


def test_sd_capped_ci_nonnegative(boot):
    """bias_sd_capped120 は標準偏差なので CI 下限が負にならない。"""
    for est in ESTIMATORS:
        m = boot["metrics"][est]["bias_sd_capped120"]
        if m["point"] is None:
            continue
        assert m["ci_low"] >= 0.0, f"{est}: SD の CI 下限が負 ({m['ci_low']})"


def test_two_stage_vs_mcmc_opposite_sides(boot):
    """§3.1 の 'opposite sides' 主張の統計的裏付け。

    two_stage_conjugate の bias_median CI が mcmc の bias_median CI より
    完全に下にある (重ならない) ことを確認する。重なれば主張が弱まる。
    """
    ts = boot["metrics"]["two_stage_conjugate"]["bias_median"]
    mcmc = boot["metrics"]["mcmc"]["bias_median"]
    assert ts["ci_high"] < mcmc["ci_low"], (
        f"CI が重複: two_stage={ts['ci_low']:.2f}–{ts['ci_high']:.2f}, "
        f"mcmc={mcmc['ci_low']:.2f}–{mcmc['ci_high']:.2f}。'opposite sides' 主張が弱まる。"
    )


def test_point_estimates_match_cell_metrics(boot):
    """メタテスト: bootstrap の点推定が cell_metrics.json と完全一致する。

    bias_median / bias_sd_capped120 の点推定は metrics.py と同一式で計算しており、
    集計パイプラインの n_t=3|prior=strong cell と数値が一致しなければならない。
    (Steel-Dwass 教訓系: 説明と実装の数値整合性をメタテストで保証。)
    """
    if not CELL_METRICS_PATH.exists():
        pytest.skip("cell_metrics.json 未生成。")
    with CELL_METRICS_PATH.open() as f:
        cm = json.load(f)
    by_est = {
        c["estimator_name"]: c
        for c in cm["core_n_t_x_prior"]
        if c["cell_key"] == "n_t=3|prior=strong"
    }
    assert by_est, "cell_metrics.json に n_t=3|prior=strong cell が無い"
    for est in ESTIMATORS:
        cell = by_est[est]
        bm = boot["metrics"][est]["bias_median"]["point"]
        bs = boot["metrics"][est]["bias_sd_capped120"]["point"]
        assert bm == pytest.approx(cell["bias_median"], abs=1e-9), (
            f"{est}: bias_median 不一致 boot={bm} vs cell_metrics={cell['bias_median']}"
        )
        assert bs == pytest.approx(cell["bias_sd_capped120"], abs=1e-9), (
            f"{est}: bias_sd_capped120 不一致 boot={bs} vs cell_metrics={cell['bias_sd_capped120']}"
        )

"""MCMC 推定器テスト (numpyro 引数・seed 再現性).

追加要求 (前停止点判断時の指示):
- pm.sample(nuts_sampler="numpyro") の引数指定が PyMC 5.28.5 で正しく動作する
- numpyro backend での seed 再現性 (同 seed で 2 回走らせて bit-identical)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_a.analysis.estimators.mcmc import (
    DEFAULT_NUTS_SAMPLER,
    estimate as mcmc_estimate,
)
from paper_a.datagen import derive_mcmc_seed

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _load_case(layer: str, case_id: str, rep: int) -> tuple[list[dict], dict]:
    truth_path = DATA_ROOT / layer / "truth.json"
    if not truth_path.exists():
        pytest.skip(f"truth.json 不在: {truth_path} (生成器を先に走らせる)")
    truth_all = {c["case_id"]: c for c in json.loads(truth_path.read_text())["cases"]}
    truth = truth_all[case_id]

    data_path = DATA_ROOT / layer / "data.csv"
    if not data_path.exists():
        pytest.skip(f"data.csv 不在: {data_path}")

    rows: list[dict] = []
    with data_path.open() as f:
        for r in csv.DictReader(f):
            if r["case_id"] != case_id:
                continue
            if int(r["replicate_id"]) != rep:
                continue
            rows.append({
                "temperature": float(r["temperature"]),
                "time_months": float(r["time_months"]),
                "content_percent": float(r["content_percent"]),
            })
    return rows, truth


def test_default_sampler_is_numpyro():
    """デフォルト nuts_sampler が numpyro であることを宣言的に確認."""
    assert DEFAULT_NUTS_SAMPLER == "numpyro", (
        "MCMC 本番は numpyro backend を採用 (4.5x speedup、同等性検証済).「default」は fallback."
    )


@pytest.mark.slow
def test_numpyro_sampler_arg_works_in_pymc():
    """PyMC 5.28.5 で nuts_sampler='numpyro' が引数として受理され、最小 draws で
    実行が完了することを保証する (mcmc_benchmark.py 既存 API との互換性確認)."""
    rows, truth = _load_case("core", "core_041", 0)
    result = mcmc_estimate(
        rows,
        case_id="core_041",
        replicate_id=0,
        prior_ea_kj=float(truth["prior_ea_kj_mol"]),
        prior_ea_sd_kj=float(truth["prior_ea_sd_kj_mol"]),
        spec_lower=90.0,
        # 最小化: テスト時間短縮 (本番は draws=2000/tune=1000)
        draws=200,
        tune=200,
        chains=2,
        nuts_sampler="numpyro",
    )
    assert result.t90_point_estimate_months is not None
    assert result.diagnostics["seed_used"] == derive_mcmc_seed("core_041", 0)


@pytest.mark.slow
def test_mcmc_seed_reproducibility_numpyro():
    """numpyro backend で同 seed の 2 回実行が bit-identical (追加要求 (b)).

    seed = derive_mcmc_seed("core_041", 0) を 2 回明示渡しして、t90 と
    R-hat/ESS が完全一致することを確認する.
    """
    rows, truth = _load_case("core", "core_041", 0)
    seed = derive_mcmc_seed("core_041", 0)
    kwargs = dict(
        case_id="core_041",
        replicate_id=0,
        prior_ea_kj=float(truth["prior_ea_kj_mol"]),
        prior_ea_sd_kj=float(truth["prior_ea_sd_kj_mol"]),
        spec_lower=90.0,
        draws=200,
        tune=200,
        chains=2,
        nuts_sampler="numpyro",
        seed=seed,
    )
    r1 = mcmc_estimate(rows, **kwargs)
    r2 = mcmc_estimate(rows, **kwargs)
    assert r1.t90_point_estimate_months == r2.t90_point_estimate_months
    assert r1.t90_lo95_months == r2.t90_lo95_months
    assert r1.t90_hi95_months == r2.t90_hi95_months
    assert r1.diagnostics["rhat_max"] == r2.diagnostics["rhat_max"]
    assert r1.diagnostics["ess_min"] == r2.diagnostics["ess_min"]

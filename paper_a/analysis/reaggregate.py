"""既存 parquet から再集計・図再生成 (MCMC 再走行なし、D.3 追加要求 b 反映時用).

estimator_results.parquet をロード → metrics.py / aggregation.py / figures.py で
再計算 → cell_metrics.json + 図 3 枚を上書き.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import math

import pandas as pd

from paper_a.analysis.aggregation import compute_all_slices
from paper_a.analysis.figures import (
    faya_fig_4_5_draft,
    fig_mcmc_nonconvergence_heatmap,
    fig_zoom_core_cell,
)
from paper_a.analysis.loaders.synthetic import load_truth

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> int:
    parquet_path = RESULTS_DIR / "estimator_results.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} がない.run_paper_a を先に実行.")

    df = pd.read_parquet(parquet_path)
    print(f"loaded {len(df)} 推定行 from {parquet_path}")

    # parquet NaN → None 正規化 (pd.NA / float nan が "is not None" を満たして
    # 集計が崩れるのを防ぐ).error_code / t90_*_months / converged 全 nullable.
    def _norm(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            return None
        return v

    results_by_estimator: dict[str, list[dict]] = {}
    for est_name, sub in df.groupby("estimator_name"):
        rows = sub.to_dict("records")
        normalized: list[dict] = []
        for r in rows:
            nr = {k: _norm(v) for k, v in r.items()}
            if isinstance(nr.get("diagnostics"), str):
                try:
                    nr["diagnostics"] = json.loads(nr["diagnostics"])
                except json.JSONDecodeError:
                    nr["diagnostics"] = {}
            # converged は bool が parquet で int になることがあるため明示的に bool 化
            if "converged" in nr and nr["converged"] is not None:
                nr["converged"] = bool(nr["converged"])
            normalized.append(nr)
        results_by_estimator[str(est_name)] = normalized
        print(f"  {est_name}: {len(normalized)} 推定")

    # truth 合体
    truth_by_case: dict[str, dict] = {}
    for layer in ("core", "robustness"):
        truth_by_case.update(load_truth(layer))

    metrics_by_slice = compute_all_slices(results_by_estimator, truth_by_case)
    metrics_path = RESULTS_DIR / "cell_metrics.json"
    metrics_path.write_text(json.dumps(
        {k: [asdict(c) for c in cells] for k, cells in metrics_by_slice.items()},
        indent=2, ensure_ascii=False,
    ))
    print(f"→ {metrics_path}")

    fig1 = faya_fig_4_5_draft(
        results_by_estimator=results_by_estimator,
        truth_by_case=truth_by_case,
        output_path=FIGURES_DIR / "fig_4_5_draft.png",
    )
    print(f"→ {fig1}")

    fig2 = fig_zoom_core_cell(
        results_by_estimator=results_by_estimator,
        truth_by_case=truth_by_case,
        n_t=3,
        output_path=FIGURES_DIR / "fig_zoom_n_t_3.png",
    )
    print(f"→ {fig2}")

    fig3 = fig_mcmc_nonconvergence_heatmap(
        metrics_by_slice,
        output_path=FIGURES_DIR / "fig_mcmc_nonconvergence.png",
    )
    print(f"→ {fig3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

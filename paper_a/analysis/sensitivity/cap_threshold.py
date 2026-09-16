"""Cap-threshold sensitivity (Referee 2, comment 3): dispersion of the shelf-life
estimator under alternative truncation horizons and robust, cap-free summaries.

Reads only `results/estimator_results.parquet` and the two `truth.json` files;
no estimator is re-run. Writes `results/cap_sensitivity.json` and prints a
Markdown rendering (Supplementary Table S3).

Two things this script makes explicit that the aggregation as published
(`metrics.py`) does not:

1. **The two-stage point estimate is capped at 120 months inside the vendored
   implementation** (`vendor/bayesian_stability.py::_cap`). For
   `two_stage_conjugate` the parquet therefore already contains
   ``min(t90, 120)``. The uncapped value is reconstructed here from
   ``diagnostics.k_mean_per_month`` as ``-ln(0.9) / k_mean``; the
   reconstruction reproduces the stored value exactly wherever the stored value
   is below the cap (asserted below).

2. For `mcmc` and `classical_ols_multi_temp` the raw point estimates reach
   1e45 - 1e203 months on a small fraction of replicates, so any SD-type
   summary that does not truncate is dominated by those replicates. The
   quantile-based summaries (IQR, MAD, 5-95 % range, log-scale SD) are the
   cap-free alternatives; the cap-based SDs are recomputed at 60, 120, 240
   months and uncapped.

Success set and truth field follow `metrics.py` (error_code is None;
`t90_true_25c_months`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_a.analysis.metrics import SHELF_LIFE_CAP_MONTHS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
TRUTH_FIELD = "t90_true_25c_months"
LOG_RATIO = -np.log(0.9)

ESTIMATORS = ("two_stage_conjugate", "mcmc", "classical_ols_multi_temp", "classical_ich_q1e")
CAPS: tuple[float | None, ...] = (60.0, SHELF_LIFE_CAP_MONTHS, 240.0, None)

# Cells reported: the central cell of Table 1, its n_T = 4 counterpart, the
# n_T = 3 accurate cell (opposite prior extreme) and the robustness aggregate
# of Table 2.
CELLS = {
    "core|n_t=3|prior=strong": dict(layer="core", n_t=3, prior_accuracy="strong"),
    "core|n_t=3|prior=accurate": dict(layer="core", n_t=3, prior_accuracy="accurate"),
    "core|n_t=4|prior=strong": dict(layer="core", n_t=4, prior_accuracy="strong"),
    "robustness_all": dict(layer="robustness"),
}


def load() -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / "estimator_results.parquet")
    meta: dict[str, dict] = {}
    for layer in ("core", "robustness"):
        for c in json.load(open(DATA / layer / "truth.json"))["cases"]:
            meta[c["case_id"]] = dict(
                layer=layer,
                n_t=c.get("n_t"),
                prior_accuracy=c.get("prior_accuracy"),
                truth=float(c[TRUTH_FIELD]),
            )
    for k in ("layer", "n_t", "prior_accuracy", "truth"):
        df[k] = df["case_id"].map(lambda cid, k=k: meta[cid][k])
    df = df[df["error_code"].isna()].copy()
    df["t90"] = df["t90_point_estimate_months"].astype(float)

    # Reconstruct the uncapped two-stage point estimate from k_mean.
    ts = df["estimator_name"] == "two_stage_conjugate"
    k_mean = df.loc[ts, "diagnostics"].map(
        lambda d: (json.loads(d) if isinstance(d, str) else d)["k_mean_per_month"]
    )
    uncapped = LOG_RATIO / k_mean.astype(float)
    stored = df.loc[ts, "t90"]
    below = stored < SHELF_LIFE_CAP_MONTHS
    # The stored t90 is rounded to 1 d.p. and k_mean to 8 d.p. in diagnostics,
    # so agreement is checked to +/-0.1 month.
    assert np.allclose(uncapped[below], stored[below], atol=0.1), "uncap reconstruction failed"
    df["t90_uncapped"] = df["t90"]
    df.loc[ts, "t90_uncapped"] = uncapped.values
    df["implementation_capped"] = False
    df.loc[ts, "implementation_capped"] = (stored >= SHELF_LIFE_CAP_MONTHS).values
    return df


def summarise(est: np.ndarray, truth: np.ndarray) -> dict:
    """Cap-based SDs and cap-free robust summaries of the bias distribution."""
    bias = est - truth
    out: dict = {"n": int(len(est))}
    for cap in CAPS:
        key = "uncapped" if cap is None else f"capped{int(cap)}"
        b = (np.minimum(est, cap) if cap is not None else est) - truth
        out[f"bias_sd_{key}"] = float(np.std(b, ddof=1)) if len(b) > 1 else None
        if cap is not None:
            out[f"frac_at_or_above_{int(cap)}"] = float(np.mean(est >= cap))
    q = np.percentile(bias, [5, 25, 50, 75, 95])
    out["bias_median"] = float(q[2])
    out["bias_iqr"] = float(q[3] - q[1])
    out["bias_p5_p95_range"] = float(q[4] - q[0])
    out["bias_mad"] = float(1.4826 * np.median(np.abs(bias - q[2])))
    pos = est > 0
    out["log_t90_sd"] = float(np.std(np.log(est[pos]), ddof=1)) if pos.sum() > 1 else None
    out["log_t90_iqr"] = float(np.subtract(*np.percentile(np.log(est[pos]), [75, 25])))
    return out


def select(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in spec.items():
        m &= (df[k] == v).values
    return df[m]


def main() -> dict:
    df = load()
    results: dict = {"cells": {}, "notes": __doc__}
    for cell, spec in CELLS.items():
        sub = select(df, spec)
        results["cells"][cell] = {}
        for est in ESTIMATORS:
            g = sub[sub["estimator_name"] == est]
            if g.empty:
                continue
            row = {
                "as_stored": summarise(g["t90"].values, g["truth"].values),
                "uncapped_point_estimate": summarise(g["t90_uncapped"].values, g["truth"].values),
                "frac_implementation_capped": float(g["implementation_capped"].mean()),
            }
            results["cells"][cell][est] = row
    out = RESULTS / "cap_sensitivity.json"
    out.write_text(json.dumps(results, indent=2))
    print(render(results))
    return results


def _f(x, nd=2):
    if x is None:
        return "—"
    if abs(x) >= 1e6:
        return f"{x:.1e}"
    return f"{x:.{nd}f}"


def render(results: dict) -> str:
    lines = []
    for cell, ests in results["cells"].items():
        lines.append(f"\n### {cell}\n")
        lines.append(
            "| Estimator | n | impl-capped | SD cap60 | SD cap120 | SD cap240 | SD uncapped "
            "| median | IQR | MAD | P5–P95 | SD(log t90) |"
        )
        lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for est, row in ests.items():
            s = row["uncapped_point_estimate"]
            lines.append(
                f"| `{est}` | {s['n']} | {row['frac_implementation_capped']*100:.1f}% "
                f"| {_f(s['bias_sd_capped60'])} | {_f(s['bias_sd_capped120'])} "
                f"| {_f(s['bias_sd_capped240'])} | {_f(s['bias_sd_uncapped'])} "
                f"| {_f(s['bias_median'])} | {_f(s['bias_iqr'])} | {_f(s['bias_mad'])} "
                f"| {_f(s['bias_p5_p95_range'])} | {_f(s['log_t90_sd'], 3)} |"
            )
    lines.append(
        "\nAll rows use the uncapped point estimate (two-stage reconstructed from k_mean); "
        "'impl-capped' is the fraction of successful replicates whose stored two-stage "
        "estimate sat at the 120-month implementation cap."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

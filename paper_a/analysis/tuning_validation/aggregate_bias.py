"""Bias / dispersion / coverage of the MCMC estimator under the sampler-tuning
matrix (Referee 2, comment 2), read from the raw_*.jsonl files of Table 4.

Table 4 as published reports only the non-convergence rate per (condition,
cell). The referee asks whether the operational conclusion survives under
reasonable tuning; the relevant question is not only how often the tuned
sampler converges but where its converged estimates sit relative to the truth.
This script adds, per (condition, cell), over the converged replicates:

    n_converged, bias_median, bias_iqr, bias_mad, bias_sd_capped120,
    coverage_probability, ci_overflow_rate (hi95 > 120)

and, for the same single sub-case, the corresponding values of
`two_stage_conjugate` from `estimator_results.parquet` (replicates 0-119, so
the two estimators are compared on identical data). The MCMC point estimate is
the posterior mean of the t90 samples in every condition, identical to the
production estimator (`estimators/mcmc.py`).

Output: results/tuning_validation/tuning_bias.json and tuning_bias.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper_a.analysis.metrics import SHELF_LIFE_CAP_MONTHS
from paper_a.analysis.tuning_validation.aggregate_tuning import CELL_ORDER, CONDITION_ORDER

ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "results" / "tuning_validation"
PARQUET = ROOT / "results" / "estimator_results.parquet"
TRUTH_FIELD = "t90_true_25c_months"
N_REPS = 120  # replicates per (condition, cell) in the tuning matrix


def truth_by_case() -> dict[str, float]:
    out = {}
    for layer in ("core", "robustness"):
        for c in json.load(open(ROOT / "data" / layer / "truth.json"))["cases"]:
            out[c["case_id"]] = float(c[TRUTH_FIELD])
    return out


def summarise(est: np.ndarray, lo: np.ndarray, hi: np.ndarray, truth: float) -> dict:
    bias = est - truth
    q = np.percentile(bias, [25, 50, 75])
    return {
        "n": int(len(est)),
        "bias_median": float(q[1]),
        "bias_iqr": float(q[2] - q[0]),
        "bias_mad": float(1.4826 * np.median(np.abs(bias - q[1]))),
        "bias_sd_capped120": float(np.std(np.minimum(est, SHELF_LIFE_CAP_MONTHS) - truth, ddof=1)),
        "pp_optimism_rate": float(np.mean(est > truth)),
        "coverage_probability": float(np.mean((lo <= truth) & (truth <= hi))),
        "ci_overflow_rate": float(np.mean(hi > SHELF_LIFE_CAP_MONTHS)),
    }


def main() -> dict:
    truth = truth_by_case()
    out: dict = {"mcmc": {}, "two_stage_conjugate": {}, "case_by_cell": {}}

    for path in sorted(TUNING.glob("raw_*.jsonl")):
        d = pd.read_json(path, lines=True)
        cond, cell = d["condition"].iloc[0], d["cell"].iloc[0]
        case = d["case_id"].iloc[0]
        out["case_by_cell"][cell] = case
        c = d[d["converged"] & d["t90_point"].notna()]
        s = summarise(c["t90_point"].values, c["t90_lo95"].values, c["t90_hi95"].values, truth[case])
        s["n_total"] = int(len(d))
        s["nonconv_rate"] = float(1 - len(c) / len(d))
        out["mcmc"].setdefault(cell, {})[cond] = s

    pq = pd.read_parquet(PARQUET)
    for cell, case in out["case_by_cell"].items():
        g = pq[
            (pq["estimator_name"] == "two_stage_conjugate")
            & (pq["case_id"] == case)
            & (pq["replicate_id"] < N_REPS)
        ]
        ok = g[g["error_code"].isna()]
        if ok.empty:
            out["two_stage_conjugate"][cell] = {"n": 0, "n_total": int(len(g)), "failure_rate": 1.0}
            continue
        s = summarise(
            ok["t90_point_estimate_months"].values.astype(float),
            ok["t90_lo95_months"].values.astype(float),
            ok["t90_hi95_months"].values.astype(float),
            truth[case],
        )
        s["n_total"] = int(len(g))
        s["failure_rate"] = float(1 - len(ok) / len(g))
        out["two_stage_conjugate"][cell] = s

    (TUNING / "tuning_bias.json").write_text(json.dumps(out, indent=2))
    md = render(out)
    (TUNING / "tuning_bias.md").write_text(md)
    print(md)
    return out


def _f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def render(out: dict) -> str:
    lines = ["# MCMC tuning matrix — bias / dispersion / coverage on converged replicates\n"]
    for cell in CELL_ORDER:
        if cell not in out["mcmc"]:
            continue
        lines.append(f"\n## {cell} (case {out['case_by_cell'][cell]}, N = 120 per condition)\n")
        lines.append(
            "| Condition | non-conv. % | n conv. | bias_median | IQR | MAD | SD cap120 "
            "| optimism % | coverage % | CI overflow % |"
        )
        lines.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for cond in CONDITION_ORDER:
            s = out["mcmc"][cell].get(cond)
            if not s:
                continue
            lines.append(
                f"| MCMC {cond} | {s['nonconv_rate']*100:.1f} | {s['n']} | {_f(s['bias_median'])} "
                f"| {_f(s['bias_iqr'])} | {_f(s['bias_mad'])} | {_f(s['bias_sd_capped120'])} "
                f"| {s['pp_optimism_rate']*100:.0f} | {s['coverage_probability']*100:.1f} "
                f"| {s['ci_overflow_rate']*100:.1f} |"
            )
        t = out["two_stage_conjugate"].get(cell)
        if t and t["n"]:
            lines.append(
                f"| two_stage_conjugate | {t['failure_rate']*100:.1f} (fail) | {t['n']} | {_f(t['bias_median'])} "
                f"| {_f(t['bias_iqr'])} | {_f(t['bias_mad'])} | {_f(t['bias_sd_capped120'])} "
                f"| {t['pp_optimism_rate']*100:.0f} | {t['coverage_probability']*100:.1f} "
                f"| {t['ci_overflow_rate']*100:.1f} |"
            )
        elif t:
            lines.append(f"| two_stage_conjugate | 100.0 (fail, N_CONDS_TOO_LOW) | 0 | — | — | — | — | — | — | — |")
    lines.append(
        "\nMCMC point estimate = posterior mean of t90 samples (as in the production run); "
        "bias relative to t90_true = 61.6224 months. two_stage_conjugate row: replicates 0-119 "
        "of the same case from estimator_results.parquet (stored estimate, implementation-capped at 120)."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

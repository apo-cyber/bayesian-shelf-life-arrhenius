"""Stage-1 -> stage-2 uncertainty propagation in the two-stage conjugate
estimator (Referee 2, comment 1): sensitivity of the central cell to how the
first-stage OLS uncertainty enters the second-stage conjugate regression.

Background — what the vendored implementation does (`vendor/bayesian_stability.py`,
Steps 2-4):

  stage 1   per temperature j: OLS of ln(C/C0) on t  ->  k_j = -slope,
            SE_j = max(SE_slope, 0.05 * k_j)                      (5 % floor)
  stage 2   likelihood  ln k_j ~ N(b0 + b1 / T_j, s_j^2),
            s_j^2 = max((SE_j / k_j)^2, 1e-8)                     (delta method)
            conjugate normal update with precision matrix W = diag(1 / s_j^2)
            -> posterior N(mu_n, Sigma_n) for (b0, b1)
  predict   ln k25 ~ N(x' mu_n, x' Sigma_n x + sigma2_resid),
            sigma2_resid = max(SS_res / (n_T - 2), 1e-6)
            point estimate  t90 = -ln(0.9) / exp(x' mu_n + x' Sigma_n x / 2)
            95 % interval from the total variance (parametric + residual)

So the heterogeneous stage-1 standard errors *are* propagated: they are the
weights of the second-stage likelihood, and under the normal-linear model
this weighted conjugate posterior is the exact marginalisation over the
stage-1 sampling distribution *given* the s_j. What is approximated is
(i) the s_j are plugged in as known although they are estimated from
n_pts - 2 residual degrees of freedom, (ii) Var(ln k_j) is obtained by the
delta method, and (iii) the 5 % floor bounds s_j from below.

Variants computed here on every successful replicate of the central cell
(n_T = 3, prior strong; 9 sub-cases x 1,000 replicates), all from the same
stage-1 fits:

  published        exactly the vendored second stage (validated against
                   estimator_results.parquet below)
  no_floor         SE_j = SE_slope (floor removed): isolates (iii)
  unweighted       homoscedastic weights, s_j^2 replaced by their mean:
                   removes the heterogeneity the referee asks about
  mc_propagation   simulation-based propagation that removes (i): for
                   b = 1..B draw s_j^2(b) = nu_j SE_j^2 / chi2_{nu_j},
                   nu_j = n_pts_j - 2 (posterior of the stage-1 slope
                   variance under the reference prior), run the conjugate
                   update with those weights, draw (b0, b1) from the
                   resulting posterior, and pool the B draws of ln k25.
                   No floor. Point estimate = -ln(0.9) / mean_b exp(ln k25(b))
                   (same c / E[k] functional as the published estimator);
                   interval = pooled 2.5 / 97.5 % quantiles including the
                   residual term, as in the published interval.

The joint model (`mcmc`) is the fully joint propagation and is reported
alongside from the parquet (converged replicates).

All point estimates are reported uncapped (the vendored implementation
clips t90 at 120 months; see `cap_threshold.py`); bias_sd_capped120 is
computed as in `metrics.py` for comparability with Table 1.

Output: results/stage1_propagation.json and .md (Supplementary Table S2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paper_a.analysis.loaders.synthetic import iter_replicates, load_truth
from paper_a.analysis.metrics import SHELF_LIFE_CAP_MONTHS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
R_GAS = 8.314462618
LOG_RATIO = -np.log(0.9)
T25 = 298.15
B_DRAWS = 400
SEED = 20260916

CELL = dict(n_t=3, prior_accuracy="strong")
VARIANTS = ("published", "no_floor", "unweighted", "mc_propagation")


# ---------------------------------------------------------------- stage 1
def stage1(rows: list[dict], initial_content: float = 100.0) -> dict:
    """Per-temperature OLS exactly as in the vendored Step 2 (before the floor)."""
    groups: dict[float, list[tuple[float, float]]] = {}
    for r in rows:
        groups.setdefault(r["temperature"], []).append((r["time_months"], r["content_percent"]))
    T_K, k, se, nu = [], [], [], []
    for T_C, pts in sorted(groups.items()):
        t = np.array([p[0] for p in pts])
        c = np.array([p[1] for p in pts])
        i0 = int(np.argmin(t))
        c0 = float(c[i0]) if t[i0] == 0 else initial_content
        valid = (c > 0) & (t >= 0)
        t, c = t[valid], c[valid]
        if len(t) < 2:
            continue
        y = np.log(c / c0)
        slope, _, _, _, se_slope = stats.linregress(t, y)
        k_hat = float(-slope)
        if k_hat <= 0:  # vendored K_HAT_ZERO_FALLBACK
            k_hat, se_slope = 1e-7, 1e-7
        T_K.append(T_C + 273.15)
        k.append(k_hat)
        se.append(float(se_slope))
        nu.append(max(len(t) - 2, 1))
    return dict(T_K=np.array(T_K), k=np.array(k), se=np.array(se), nu=np.array(nu))


# ---------------------------------------------------------------- stage 2
def conjugate(x: np.ndarray, y: np.ndarray, obs_var: np.ndarray, prior_ea: float, prior_sd: float):
    """Vendored Step 3 for one (or a batch of) observation-variance vectors.

    x, y: (n,), obs_var: (..., n). Returns mu_n (..., 2), Sigma_n (..., 2, 2),
    sigma2_resid (...,).
    """
    X = np.column_stack([np.ones_like(x), x])  # (n, 2)
    mu0 = np.array([20.0, -prior_ea * 1000.0 / R_GAS])
    S0inv = np.diag([1.0 / 100.0**2, 1.0 / (prior_sd * 1000.0 / R_GAS) ** 2])
    w = 1.0 / obs_var  # (..., n)
    XtWX = np.einsum("...n,ni,nj->...ij", w, X, X)
    XtWy = np.einsum("...n,ni,n->...i", w, X, y)
    Sn = np.linalg.inv(S0inv + XtWX)
    mu = np.einsum("...ij,...j->...i", Sn, S0inv @ mu0 + XtWy)
    resid = y - np.einsum("ni,...i->...n", X, mu)
    n = len(x)
    s2 = np.maximum(np.sum(resid**2, axis=-1) / max(n - 2, 1), 1e-6)
    return mu, Sn, s2


def predict(mu, Sn, s2):
    x = np.array([1.0, 1.0 / T25])
    lnk = np.einsum("...i,i->...", mu, x)
    v_par = np.einsum("i,...ij,j->...", x, Sn, x)
    v_tot = v_par + s2
    k_mean = np.exp(lnk + v_par / 2)
    return dict(
        t90=LOG_RATIO / k_mean,
        lo95=LOG_RATIO / np.exp(lnk + 1.960 * np.sqrt(v_tot)),
        hi95=LOG_RATIO / np.exp(lnk - 1.960 * np.sqrt(v_tot)),
        lnk=lnk,
        v_par=v_par,
        s2=s2,
    )


def analytic_variant(s1: dict, prior_ea: float, prior_sd: float, *, floor: bool, weighted: bool) -> dict:
    se = np.maximum(s1["se"], 0.05 * s1["k"]) if floor else s1["se"]
    obs_var = np.maximum((se / s1["k"]) ** 2, 1e-8)
    if not weighted:
        obs_var = np.full_like(obs_var, obs_var.mean())
    mu, Sn, s2 = conjugate(1.0 / s1["T_K"], np.log(s1["k"]), obs_var, prior_ea, prior_sd)
    return predict(mu, Sn, s2)


def mc_variant(s1: dict, prior_ea: float, prior_sd: float, rng: np.random.Generator) -> dict:
    nu = s1["nu"]
    chi2 = rng.chisquare(nu, size=(B_DRAWS, len(nu)))
    s_slope2 = nu * s1["se"] ** 2 / chi2  # (B, n): stage-1 slope variance draws
    obs_var = np.maximum(s_slope2 / s1["k"] ** 2, 1e-8)
    mu, Sn, s2 = conjugate(1.0 / s1["T_K"], np.log(s1["k"]), obs_var, prior_ea, prior_sd)
    x = np.array([1.0, 1.0 / T25])
    L = np.linalg.cholesky(Sn)  # (B, 2, 2)
    z = rng.standard_normal((B_DRAWS, 2))
    beta = mu + np.einsum("bij,bj->bi", L, z)
    lnk_par = beta @ x
    lnk_tot = lnk_par + rng.standard_normal(B_DRAWS) * np.sqrt(s2)
    k_mean = float(np.mean(np.exp(lnk_par)))
    t90_tot = LOG_RATIO / np.exp(lnk_tot)
    return dict(
        t90=LOG_RATIO / k_mean,
        lo95=float(np.percentile(t90_tot, 2.5)),
        hi95=float(np.percentile(t90_tot, 97.5)),
    )


# ---------------------------------------------------------------- driver
def run_cell() -> pd.DataFrame:
    truth = load_truth("core")
    cases = sorted(cid for cid, c in truth.items() if all(c[k] == v for k, v in CELL.items()))
    pq = pd.read_parquet(RESULTS / "estimator_results.parquet")
    pub = pq[(pq["estimator_name"] == "two_stage_conjugate") & pq["case_id"].isin(cases)]
    pub = pub[pub["error_code"].isna()].set_index(["case_id", "replicate_id"])
    success = set(pub.index)
    rng = np.random.default_rng(SEED)
    recs = []
    for cid, rep, rows in iter_replicates("core", case_ids=cases):
        if (cid, rep) not in success:
            continue
        c = truth[cid]
        s1 = stage1(rows, float(c["initial_content"]))
        pe, ps = float(c["prior_ea_kj_mol"]), float(c["prior_ea_sd_kj_mol"])
        rec = dict(case_id=cid, replicate_id=rep, truth=float(c["t90_true_25c_months"]),
                   n_points=int(c["n_points"]), max_se_ratio=float(np.max(s1["se"] / s1["k"])),
                   floor_active=int(np.sum(s1["se"] < 0.05 * s1["k"])))
        for name, kw in (("published", dict(floor=True, weighted=True)),
                         ("no_floor", dict(floor=False, weighted=True)),
                         ("unweighted", dict(floor=True, weighted=False))):
            p = analytic_variant(s1, pe, ps, **kw)
            rec.update({f"{name}_t90": float(p["t90"]), f"{name}_lo": float(p["lo95"]), f"{name}_hi": float(p["hi95"])})
        p = mc_variant(s1, pe, ps, rng)
        rec.update({"mc_propagation_t90": p["t90"], "mc_propagation_lo": p["lo95"], "mc_propagation_hi": p["hi95"]})
        d = pub.loc[(cid, rep), "diagnostics"]
        rec["parquet_k_mean"] = float((json.loads(d) if isinstance(d, str) else d)["k_mean_per_month"])
        recs.append(rec)
    df = pd.DataFrame(recs)
    # Validate the re-implementation against the production output.
    # k_mean is stored rounded to 8 d.p., so replicates with k_mean ~ 1e-7 carry
    # only one significant digit; the check is restricted to k_mean > 1e-5.
    rel = np.abs(LOG_RATIO / df["parquet_k_mean"] - df["published_t90"]) / df["published_t90"]
    rel = rel[df["parquet_k_mean"] > 1e-5]
    assert rel.max() < 1e-3, f"published variant does not reproduce parquet (max rel {rel.max():.2e})"
    return df


def summarise(t90: np.ndarray, lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> dict:
    b = t90 - truth
    q = np.percentile(b, [25, 50, 75])
    return dict(
        n=int(len(t90)),
        bias_median=float(q[1]),
        bias_iqr=float(q[2] - q[0]),
        bias_mad=float(1.4826 * np.median(np.abs(b - q[1]))),
        bias_sd_capped120=float(np.std(np.minimum(t90, SHELF_LIFE_CAP_MONTHS) - truth, ddof=1)),
        pp_optimism_rate=float(np.mean(t90 > truth)),
        coverage_probability=float(np.mean((lo <= truth) & (truth <= hi))),
        median_interval_width_log=float(np.median(np.log(hi) - np.log(lo))),
        frac_above_120=float(np.mean(t90 > SHELF_LIFE_CAP_MONTHS)),
    )


def main() -> dict:
    df = run_cell()
    out: dict = {"cell": CELL, "n_replicates": int(len(df)), "B_draws": B_DRAWS, "variants": {}, "by_n_points": {}}
    for v in VARIANTS:
        out["variants"][v] = summarise(df[f"{v}_t90"].values, df[f"{v}_lo"].values, df[f"{v}_hi"].values, df["truth"].values)
    for npts, g in df.groupby("n_points"):
        out["by_n_points"][int(npts)] = {
            v: summarise(g[f"{v}_t90"].values, g[f"{v}_lo"].values, g[f"{v}_hi"].values, g["truth"].values)
            for v in VARIANTS
        }
    out["stage1_diagnostics"] = dict(
        frac_replicates_floor_active=float(np.mean(df["floor_active"] > 0)),
        mean_temperatures_floor_active=float(df["floor_active"].mean()),
        frac_replicates_se_ratio_gt_015=float(np.mean(df["max_se_ratio"] > 0.15)),
    )
    # Joint model reference (converged MCMC on the same cell).
    pq = pd.read_parquet(RESULTS / "estimator_results.parquet")
    truth = load_truth("core")
    cases = set(df["case_id"])
    m = pq[(pq["estimator_name"] == "mcmc") & pq["case_id"].isin(cases) & pq["error_code"].isna()]
    tr = m["case_id"].map(lambda c: float(truth[c]["t90_true_25c_months"])).values
    out["variants"]["mcmc_joint"] = summarise(
        m["t90_point_estimate_months"].values.astype(float),
        m["t90_lo95_months"].values.astype(float),
        m["t90_hi95_months"].values.astype(float),
        tr,
    )
    # Pairwise agreement of the point estimate with the published variant.
    out["pairwise_vs_published"] = {
        v: dict(
            median_abs_rel_diff=float(np.median(np.abs(df[f"{v}_t90"] - df["published_t90"]) / df["published_t90"])),
            p95_abs_rel_diff=float(np.percentile(np.abs(df[f"{v}_t90"] - df["published_t90"]) / df["published_t90"], 95)),
        )
        for v in VARIANTS[1:]
    }
    (RESULTS / "stage1_propagation.json").write_text(json.dumps(out, indent=2))
    df.to_parquet(RESULTS / "stage1_propagation_replicates.parquet", index=False)
    md = render(out)
    (RESULTS / "stage1_propagation.md").write_text(md)
    print(md)
    return out


def _row(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['n']} | {s['bias_median']:.2f} | {s['bias_iqr']:.1f} | {s['bias_mad']:.1f} "
        f"| {s['bias_sd_capped120']:.2f} | {s['pp_optimism_rate']*100:.0f} | {s['coverage_probability']*100:.1f} "
        f"| {s['median_interval_width_log']:.2f} | {s['frac_above_120']*100:.1f} |"
    )


def render(out: dict) -> str:
    hdr = ("| Second stage | n | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % "
           "| median log-width 95 % | > 120 mo % |\n|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    lines = [f"# Stage-1 uncertainty propagation — central cell (n_T = 3, prior strong), "
             f"{out['n_replicates']} successful replicates, B = {out['B_draws']}\n", hdr]
    for v, s in out["variants"].items():
        lines.append(_row(v, s))
    d = out["stage1_diagnostics"]
    lines.append(
        f"\nStage-1 diagnostics: 5 % SE floor active in {d['frac_replicates_floor_active']*100:.1f} % of replicates "
        f"({d['mean_temperatures_floor_active']:.2f} of 3 temperatures on average); "
        f"max SE_k/k > 0.15 (delta-method warning) in {d['frac_replicates_se_ratio_gt_015']*100:.1f} %."
    )
    lines.append("\nPoint-estimate agreement with `published` (|rel. diff|): " + "; ".join(
        f"{v}: median {p['median_abs_rel_diff']*100:.2f} %, P95 {p['p95_abs_rel_diff']*100:.1f} %"
        for v, p in out["pairwise_vs_published"].items()))
    for npts, vs in out["by_n_points"].items():
        lines.append(f"\n## n_pts = {npts} (stage-1 residual df = {npts - 2})\n\n" + hdr)
        for v, s in vs.items():
            lines.append(_row(v, s))
    return "\n".join(lines)


if __name__ == "__main__":
    main()

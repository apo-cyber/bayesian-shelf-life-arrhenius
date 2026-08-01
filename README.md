# bayesian-shelf-life-arrhenius

Reproducibility package for the study

> **Bayesian multi-temperature shelf-life prediction extending Faya et al. (2018)**
> Yasushi Arai — ORCID [0009-0005-7300-4234](https://orcid.org/0009-0005-7300-4234)
> Production Division, Alfresa Pharma Corporation, Osaka, Japan
> *(This work was conducted independently of the author's employer, on personal
> time and resources. The views expressed are solely those of the author and do
> not represent Alfresa Pharma Corporation.)*

A synthetic-data simulation study comparing four estimators of pharmaceutical
shelf life from accelerated stability data — a two-stage OLS/conjugate Bayesian
procedure, full MCMC (NUTS), classical multi-temperature OLS, and the ICH Q1E
long-term baseline — across 81 core cases and 20 robustness cases.

> **The manuscript is not included in this repository.** The paper text is
> managed separately (in accordance with the publisher agreement) and is
> **under review at a peer-reviewed journal**. This repository contains only the
> code and synthetic data needed to reproduce the figures, tables, and metrics.

## License

- **Code** — MIT (`LICENSE`). This includes `paper_a/vendor/`, which is the
  author's own scientific code dual-licensed from the private
  `apo-cyber/cmc-platform` repository (scientific core only; see each vendored
  file's provenance header).
- **Data** — CC BY 4.0 (`LICENSE-DATA`), covering `paper_a/data/`,
  `paper_a/results/`, and `paper_a/figures/`. All data is **synthetic** (every
  `truth.json` carries `data_class: "synthetic"`); no measured or third-party
  stability data is included.

### About `paper_a/vendor/`

Two of the four estimators (`two_stage_conjugate`, `classical_ich_q1e`) call
scientific functions that live in the author's private `cmc-platform` product.
Rather than re-implement them (which would risk numerical divergence from the
published results), the **scientific core only** of those functions is vendored
under `paper_a/vendor/` and MIT dual-licensed. Product glue (FastAPI
serialization, verdict/commentary layers, routers, auth/DB) is **not** included.
This keeps the package fully self-contained — no access to the proprietary
backend is required.

The full-MCMC estimator (`analysis/estimators/mcmc.py`) also derives from the
author's MCMC benchmark in the private `cmc-platform` project (MIT
dual-licensed); see its module docstring.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # core: numpy/scipy/pandas/pyarrow/matplotlib
# optional, for the full MCMC estimator (heavy: PyMC/JAX):
pip install -e ".[mcmc]"
# optional, for the test suite:
pip install -e ".[dev]"
```

Requires Python ≥ 3.12. Dependency versions are pinned exactly (see
`pyproject.toml`) because the numerics (numpy/scipy/pandas/pyarrow) and the MCMC
stack (PyMC/numpyro/JAX) can change behaviour across versions.

## Reproduction

### Quick (figures only, from committed results — no simulation)

Regenerates the three figures from the committed
`paper_a/results/estimator_results.parquet` without re-running any estimator
(in particular, **no MCMC**). Completes in well under a minute.

```bash
python paper_a/analysis/reaggregate.py
# rewrites paper_a/results/cell_metrics.json and paper_a/figures/*.png
```

### Full (regenerate everything from scratch)

```bash
# 1. Synthetic data (bit-identical; does not overwrite truth.json)
python -m paper_a.datagen

# 2. Run all estimators (writes results/estimator_results.parquet)
python -m paper_a.analysis.run_paper_a          # needs the [mcmc] extra

# 3. Aggregate + figures
python paper_a/analysis/reaggregate.py

# 4. (optional) bootstrap CIs for the central cell
python -m paper_a.analysis.bootstrap_ci
```

### Tables

Each manuscript table has one script. Steps 3 and 4 above cover Tables 1 and 2
(they are read off `results/cell_metrics.json` and `results/bootstrap_ci.json`);
the remaining three are separate because they aggregate differently.

```bash
# Table 3 — variation across temperature-dependence variants (reads cell_metrics.json)
python -m paper_a.analysis.table3_variation

# Table 4 — sampler-tuning matrix (reads results/tuning_validation/*.jsonl)
python -m paper_a.analysis.tuning_validation.aggregate_tuning

# Supplementary Table S1 — MCMC non-convergence stratified by sigma (reads the parquet)
python -m paper_a.analysis.supp_table_s1
```

Table 4's raw runs are committed under `paper_a/results/tuning_validation/`, so
the sampler-tuning matrix does not have to be re-run. To re-execute it from
scratch (hours of NUTS sampling), use
`python -m paper_a.analysis.tuning_validation.run_tuning_matrix`.

**Runtime / asymmetric budget.** The estimators differ in per-replicate cost by
~4 orders of magnitude. The three closed-form estimators
(`two_stage_conjugate`, `classical_ols_multi_temp`, `classical_ich_q1e`) run
**1,000 replicates/case** and finish in a few minutes total. **MCMC is the
bottleneck**: it runs **100 replicates/case** (10,100 NUTS fits) and takes
**~8 hours** on a single CPU workstation via the numpyro backend. Step 2 is the
only overnight step; Steps 1, 3, 4 are fast. Use `--skip-mcmc` on
`run_paper_a` to exercise the three fast estimators alone.

## Figures ↔ manuscript

| Repository file | Manuscript | Content |
|---|---|---|
| `paper_a/figures/fig_t90_estimates_by_cell.png` | Figure 1 | t90 point-estimate distributions (median / IQR / range), 4 estimators × 9 core cells (n_T × prior accuracy), true t90 reference line |
| `paper_a/figures/fig_zoom_n_t_3.png` | Figure 2 | Zoom on the central cells (n_T = 3 × 3 prior levels) |
| `paper_a/figures/fig_mcmc_nonconvergence.png` | Figure 3 | MCMC non-convergence rate heatmap over core cells (R-hat ≥ 1.01 or ESS < 400) |

Each figure is written as PNG (the canonical form, embedded in the manuscript)
and as a 300 dpi LZW-TIFF alongside it for the publisher's artwork requirements.
The TIFFs are build products and are not tracked.

## Tables ↔ manuscript

| Script | Output | Manuscript |
|---|---|---|
| `paper_a/analysis/reaggregate.py` | `results/cell_metrics.json` | Tables 1, 2 (point estimates) |
| `paper_a/analysis/bootstrap_ci.py` | `results/bootstrap_ci.json` | Table 1 (95% CIs) |
| `paper_a/analysis/table3_variation.py` | `results/table3_variation.json` | Table 3 |
| `paper_a/analysis/tuning_validation/aggregate_tuning.py` | `results/tuning_validation/tuning_summary.{csv,md}` | Table 4 |
| `paper_a/analysis/supp_table_s1.py` | `results/supp_table_s1.json` | Supplementary Table S1 |

The published value of every figure and table is pinned in `paper_a/tests/`, so
`pytest` fails if a change to the aggregation moves a number that appears in the
manuscript.

## Data

All synthetic. Two layers (per-case truth in `data/<layer>/truth.json`):

- **core (81 cases)** — first-order Arrhenius kinetics, factorial over number of
  accelerated temperatures `n_T ∈ {2,3,4}`, sampling times, observation noise,
  and prior accuracy. Single true shelf life `t90(25°C) = 61.6224` months.
- **robustness (20 cases)** — 4 kinetic models × 5 temperature dependencies
  (incl. non-Arrhenius concave/convex), fixed experimental design.

Schema: `data.csv` columns `(case_id, replicate_id, temperature, time_months,
content_percent)`; `long_term_25c.csv` holds the 25 °C long-term series used by
the ICH Q1E baseline.

## Citing

Please cite the software via `CITATION.cff` (archival **Zenodo DOI:
[`10.5281/zenodo.20576829`](https://doi.org/10.5281/zenodo.20576829)** — the
concept DOI, which always resolves to the latest version) and the accompanying
manuscript:

> Arai, Y. *Bayesian multi-temperature shelf-life prediction extending Faya et
> al. (2018).* Manuscript under review at a peer-reviewed journal.

and the work it extends:

> Faya, P., Seaman, J. W., & Stamey, J. D. (2018). Using accelerated drug
> stability results to inform long-term studies in shelf life determination.
> *Statistics in Medicine*, 37(17), 2599–2615. https://doi.org/10.1002/sim.7663

## Release history

### v1.0.1 — 2026-08-01

Completes the reproduction path. Every figure and table in the manuscript can
now be regenerated from a clean clone; in v1.0.0 three of them could not be,
because the code that produced them was not part of the release.

Added:

- `paper_a/analysis/tuning_validation/` and `paper_a/results/tuning_validation/`
  — the sampler-tuning matrix behind **Table 4**, code and raw runs. Present in
  the working tree at v1.0.0 but never committed, so Table 4 was not
  reproducible from the archive.
- `paper_a/analysis/table3_variation.py` — **Table 3**. No generating script had
  existed; the aggregation was reconstructed and confirmed against the published
  values. Its docstring records the three alternative readings of "variation"
  that do *not* reproduce the table, so a reader does not have to guess.
- `paper_a/analysis/supp_table_s1.py` — **Supplementary Table S1**. Same
  situation as Table 3. S1 stratifies by observation noise, which no slice of
  `cell_metrics.json` carries, so it is computed from the parquet.
- Tests pinning all of the above to the published values
  (`paper_a/tests/test_table3_variation.py`, `test_supp_table_s1.py`).

Fixed:

- The Zenodo DOI in this README was still a `TBD` placeholder while the
  manuscript cited the assigned DOI.
- `CITATION.cff` declared `version: 0.1.0` while the archived release was 1.0.0.

Unchanged — **no numerical artefact was touched**: `estimator_results.parquet`,
`cell_metrics.json`, `bootstrap_ci.json`, `paper_a/data/`, `paper_a/vendor/`,
and all four estimator implementations are byte-identical to v1.0.0. This was
verified by regenerating everything in a clean clone: steps 1, 3 and 4 of the
Full procedure above reproduce every committed artefact bit-for-bit (`git
status` reports no modification), and Tables 1–4 and S1 match the published
values exactly.

### v1.0.0 — 2026-06-06

Initial archived release. Note that the tag predates the Figure 2 regeneration
(commit `26719e8`, which widened the zoom panel's y-axis to 30–110 months and
relabelled it), so the Figure 2 in the v1.0.0 archive is not the one that
appears in the manuscript. v1.0.1 carries the current figure.

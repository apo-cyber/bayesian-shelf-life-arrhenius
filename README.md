# bayesian-shelf-life-arrhenius

Reproducibility package for the study

> **Two-stage conjugate regression versus MCMC for Bayesian shelf-life prediction
> from accelerated stability data: a simulation study**
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
> managed separately (in accordance with the publisher agreement) and has been
> **accepted for publication in the *Journal of Biopharmaceutical Statistics***.
> This repository contains only the code and synthetic data needed to reproduce
> the figures, tables, and metrics.

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
the remaining ones are separate because they aggregate differently.

```bash
# Table 3 — variation across temperature-dependence variants (reads cell_metrics.json)
python -m paper_a.analysis.table3_variation

# Table 4 — sampler-tuning matrix (reads results/tuning_validation/*.jsonl)
python -m paper_a.analysis.tuning_validation.aggregate_tuning

# Table 4 — bias, IQR and coverage of the converged replicates (reads the same jsonl)
python -m paper_a.analysis.tuning_validation.aggregate_bias

# Supplementary Table S1 — MCMC non-convergence stratified by sigma (reads the parquet)
python -m paper_a.analysis.supp_table_s1

# Supplementary Table S2 — propagation of first-stage uncertainty (reads data + parquet;
# re-fits the second stage on the central cell, seeded Monte Carlo, ~10 s)
python -m paper_a.analysis.sensitivity.stage1_propagation

# Supplementary Tables S3, S3b — truncation horizon and cap-free summaries (reads the parquet)
python -m paper_a.analysis.sensitivity.cap_threshold
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
| `paper_a/analysis/tuning_validation/aggregate_bias.py` | `results/tuning_validation/tuning_bias.{json,md}` | Table 4 (median bias, IQR, coverage) |
| `paper_a/analysis/supp_table_s1.py` | `results/supp_table_s1.json` | Supplementary Table S1 |
| `paper_a/analysis/sensitivity/stage1_propagation.py` | `results/stage1_propagation.{json,md}` | Supplementary Table S2 |
| `paper_a/analysis/sensitivity/cap_threshold.py` | `results/cap_sensitivity.{json,md}` | Supplementary Tables S3, S3b |

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

> Arai, Y. Two-stage conjugate regression versus MCMC for Bayesian shelf-life
> prediction from accelerated stability data: a simulation study. *Journal of
> Biopharmaceutical Statistics*, accepted for publication (2026).

and the work it extends:

> Faya, P., Seaman, J. W., & Stamey, J. D. (2018). Using accelerated drug
> stability results to inform long-term studies in shelf life determination.
> *Statistics in Medicine*, 37(17), 2599–2615. https://doi.org/10.1002/sim.7663

## Release history

### v1.1.1 — 2026-11-16

Metadata only. The `.zenodo.json` archived with v1.1.0 still declared
`"version": "1.0.1"`, so the Zenodo record of v1.1.0 was first published with
that version number (its metadata has since been corrected on Zenodo). This
release sets the version to 1.1.1 in `.zenodo.json`, `CITATION.cff` and
`pyproject.toml`. No code, data, result or test differs from v1.1.0.

### v1.1.0 — 2026-11-16

Adds the analyses made during peer review, so that every table of the accepted
manuscript can be regenerated from the archive. In v1.0.1 the extended Table 4
and Supplementary Tables S2, S3 and S3b could not be, because their code was
written after that release.

Added:

- `paper_a/analysis/tuning_validation/aggregate_bias.py` — the median bias,
  IQR and coverage of the converged replicates in **Table 4**, with the
  two-stage estimator on the same replicates for comparison
  (`results/tuning_validation/tuning_bias.{json,md}`).
- Two further cells of the sampler-tuning matrix (n_T = 4 strong, n_T = 3
  accurate; configurations A and E): four raw jsonl files, bringing
  `results/tuning_validation/` to 16 jsonl files, and the corresponding rows
  of `tuning_summary.{csv,md}`.
- `paper_a/analysis/sensitivity/stage1_propagation.py` — **Supplementary
  Table S2**, the second stage of the two-stage estimator recomputed under four
  treatments of first-stage uncertainty
  (`results/stage1_propagation.{json,md}` and the per-replicate
  `stage1_propagation_replicates.parquet`).
- `paper_a/analysis/sensitivity/cap_threshold.py` — **Supplementary Tables S3
  and S3b**, dispersion under truncation horizons of 60, 120 and 240 months
  and uncapped, with cap-free summaries (`results/cap_sensitivity.{json,md}`).
- `paper_a/tests/test_revision_tables.py` — pins the committed outputs above
  to the values printed in Table 4 and Supplementary Tables S2–S3b.

Fixed:

- `pyproject.toml` still declared `version = "0.1.0"`; it now matches the
  release (1.1.0).

Unchanged: the estimators, `paper_a/vendor/`, `paper_a/data/`,
`estimator_results.parquet`, `cell_metrics.json` and `bootstrap_ci.json`. The
new scripts only read existing outputs (S2 re-fits the second stage from the
same first-stage fits; its Monte Carlo is seeded).

Verified in a clean clone: `aggregate_tuning`, `aggregate_bias` and
`cap_threshold` reproduce their committed outputs byte for byte.
`stage1_propagation` reproduces `stage1_propagation.md` byte for byte; the
full-precision floats in its JSON and parquet differ from the committed ones by
at most 4 × 10⁻⁹ relative, which does not reach any printed digit. The test
suite passes (52 + 45 tests). On a recent macOS command-line toolchain
PyTensor's C compilation can fail to link (`library 'd64' not found`), which
makes the two PyMC tests in `test_mcmc.py` fail for reasons unrelated to this
package; `PYTENSOR_FLAGS='cxx='` runs them in Python mode, where they pass.

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

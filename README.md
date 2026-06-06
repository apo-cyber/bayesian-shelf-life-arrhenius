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
> managed separately (in accordance with the publisher agreement) and has been
> **submitted to *Statistics in Medicine***. This repository contains only the
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

**Runtime / asymmetric budget.** The estimators differ in per-replicate cost by
~4 orders of magnitude. The three closed-form estimators
(`two_stage_conjugate`, `classical_ols_multi_temp`, `classical_ich_q1e`) run
**1,000 replicates/case** and finish in a few minutes total. **MCMC is the
bottleneck**: it runs **100 replicates/case** (10,100 NUTS fits) and takes
**~8 hours** on a single CPU workstation via the numpyro backend. Step 2 is the
only overnight step; Steps 1, 3, 4 are fast. Use `--skip-mcmc` on
`run_paper_a` to exercise the three fast estimators alone.

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
`10.5281/zenodo.TBD`** — to be assigned at release) and the accompanying
manuscript:

> Arai, Y. *Bayesian multi-temperature shelf-life prediction extending Faya et
> al. (2018).* Manuscript submitted to *Statistics in Medicine*.

and the work it extends:

> Faya, P., Seaman, J. W., & Stamey, J. D. (2018). Using accelerated drug
> stability results to inform long-term studies in shelf life determination.
> *Statistics in Medicine*, 37(17), 2599–2615. https://doi.org/10.1002/sim.7663

"""Vendored scientific core from cmc-platform (author's own code, MIT dual-license).

These modules are verbatim copies of the scientific functions that paper_a's
estimators reach in the private `apo-cyber/cmc-platform` repository. They are
dual-licensed under MIT here so that this reproducibility package is fully
self-contained and requires no access to the proprietary backend.

Only the scientific path is vendored; product glue (FastAPI serialization,
verdict/commentary layers, routers, auth/DB) is not included. See each module's
provenance header for the source path and commit.

Exposed entry points:
    - bayesian_stability.run_bayesian_stability / StabilityHardFailWarning
      (used by paper_a.analysis.estimators.two_stage_conjugate)
    - classical_stability.classical_ich_q1e_single_temp
      (used by paper_a.analysis.estimators.classical_ich_q1e)
"""

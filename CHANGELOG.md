# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

## [0.1.5]

### Added
- Exported accessors `clustervars`, `se_type`, and `vcov_type`, defined uniformly
  for all four model types (`TWFEModel`, `SunabModel`, `GardnerModel`, `BJSModel`).
  `clustervars(m)` returns the clustering variable(s) (empty when standard errors
  are robust/unclustered), `se_type(m)` returns `:robust` or `:cluster`, and
  `vcov_type(m)` returns the same `Vcov.*` object FixedEffectModels exposes on its
  `vcov_type` field. This lets table builders and other tooling detect the
  standard-error type and cluster variables generically.

### Fixed
- Multiway (e.g. two-way) clustering now works end to end for `fit_twfe_static`,
  `fit_twfe_dynamic`, and `fit_sunab`. Passing `cluster=(:a, :b)` (or a vector)
  previously computed the correct covariance matrix but errored when constructing
  the model; the result now stores every clustering variable. `fit_bjs` and the
  Gardner estimators remain single-cluster by design.

### Changed
- Each model now stores its clustering variables as a `Vector{Symbol}` (empty =
  robust) instead of a single `Symbol`. This is an internal representation change;
  the `fit_*` signatures, numerical results, and the StatsAPI interface are
  unchanged.
- `cumulative_effects` now propagates the requested confidence level through to
  its reported confidence intervals.

### Documentation
- Expanded the `fit_bjs`, `fit_bjs_static`, and `fit_bjs_dynamic` docstrings to
  describe the `project` (continuous) and `hetby` (discrete) heterogeneity options
  and when to use each, how they behave under `horizons` (per-horizon
  coefficients, e.g. `τ0_cons`/`τ<h>_<value>`), the full layout of the returned
  coefficient vector (pre-trends → treatment effects → controls), and the fact
  that pre-trend coefficients are never reshaped by `project`/`hetby`.

### Notes
- Non-breaking: existing calls produce identical results.

## [0.1.4]

### Performance
- Faster Gardner standard errors via a sparse fixed-effect Gram solve and
  single-pass clustering.
- Faster BJS estimation: hoisted scalar `DataFrame` indexing, function barriers,
  integer-based cluster grouping, fewer redundant copies, and an optional threaded
  influence-weight orthogonalization (identical results, lower wall-clock).
- Added a precompilation workload to cut first-call latency.

### Added
- Stata `did_imputation` and cross-package benchmark harnesses.

### Documentation
- README updates: corrected the stated Stata parity (~1e-7/1e-8), fairer
  idle-machine performance benchmarks, and fixed stale function names.

## [0.1.3]

### Added
- `project` option for `fit_bjs` (and the dynamic and static wrappers). This gives
  continuous treatment effect heterogeneity by reporting an OLS projection of the
  individual imputed effects onto the covariates you pass: an intercept `τ_cons` and
  a slope `τ_<var>` for each covariate, or one set per event time when combined with
  `horizons`. It mirrors the Stata `did_imputation` `project()` option.
- `hetby` option for `fit_bjs`. This gives discrete treatment effect heterogeneity by
  reporting a separate effect for each value of a grouping variable (up to 30 values),
  named `τ_<value>` or `τ<h>_<value>`. It mirrors the Stata `hetby()` option.
- `minn` option for `fit_bjs`. This suppresses coefficients whose effective sample
  size falls below the given threshold, which protects against misleadingly small
  standard errors on thin coefficients. It defaults to `0`, which keeps every
  coefficient. Setting `minn=30` reproduces Stata's default behaviour.

### Notes
- The new options are additive and do not change the results of existing calls. In
  particular `minn` defaults to `0`, so output is identical to previous versions
  unless you opt in.
- `project` and `hetby` cannot be used together, matching Stata.
- Point estimates and standard errors for every supported option combination, on both
  the single threaded and multithreaded backends, were checked against Stata
  `did_imputation` and agree to within a relative tolerance of 1e-4.

## [0.1.2]

- Earlier development releases of the BJS, Gardner, Sun-Abraham, and two-way fixed
  effects estimators.

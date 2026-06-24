module StagDiDModels

using Logging # Turns of the printing of FixedEffectModels.jl 
using DataFrames
using StatsAPI
using StatsAPI: pvalue, dof_residual
using StatsModels
# using StatsModels: cols
using FixedEffectModels
using CovarianceMatrices
using CategoricalArrays
using LinearAlgebra
using Statistics
using Random
using Distributions
using Distributions: TDist, FDist, ccdf, quantile
using SparseArrays
using StatsBase
using StatsBase: CoefTable, NoQuote, PValue, TestStat
using Printf


#structs
include("struct/TWFE.jl")
include("struct/BJSModel.jl")
include("struct/GardnerModel.jl")
include("struct/Sunab.jl")
export TWFEModel, BJSModel, SunabModel, GardnerModel
# reexport:
export responsename, coef, vcov, coefnames, nobs, dof_residual, dof, stderror, pvalue, confint,r2, islinear, adjr2, coeftable,show

# Cluster / SE-type accessors, uniform across all four model types. Mirrors how
# FixedEffectModels exposes its fitted vcov so downstream packages (table
# builders, etc.) can render "Std. errors" and "Cluster" rows generically.
const _DiDModel = Union{TWFEModel, BJSModel, SunabModel, GardnerModel}

"""
    clustervars(m) -> Vector{Symbol}

Clustering variable(s) used for the model's standard errors. Empty when the SEs
are heteroskedasticity-robust (unclustered). BJS and Gardner return at most one
variable; TWFE and Sunab may return several (multiway clustering).
"""
clustervars(m::_DiDModel) = m.cluster

"""
    se_type(m) -> Symbol

Standard-error type: `:cluster` when [`clustervars`](@ref) is non-empty,
otherwise `:robust`.
"""
se_type(m::_DiDModel) = isempty(m.cluster) ? :robust : :cluster

"""
    vcov_type(m) -> Vcov.CovarianceEstimator

The covariance estimator behind the model's standard errors, in the same
vocabulary FixedEffectModels uses on its `vcov_type` field: `Vcov.cluster(vars...)`
when clustered, otherwise `Vcov.robust()`. Lets vcov-aware consumers classify the
SE type exactly as they would a `FixedEffectModel`
(`vt isa Vcov.ClusterCovariance` / `Vcov.RobustCovariance`).
"""
vcov_type(m::_DiDModel) = isempty(m.cluster) ? Vcov.robust() : Vcov.cluster(m.cluster...)

export clustervars, se_type, vcov_type

# utils
include("utils/quiet_reg.jl")
include("utils/preprocess.jl")
include("utils/build_vcov.jl")
include("utils/fe_predict.jl")
include("utils/autosample_utils.jl")

# estimators
include("estimators/twfe.jl")
export fit_twfe_dynamic, fit_twfe_static
include("estimators/gardner.jl")
export fit_gardner_dynamic, fit_gardner_static
# include("estimators/bjs_R.jl")
# export fit_bjs, fit_bjs_dynamic, fit_bjs_static,fit_bjs_wild_bootstrap, fit_bjs_wild_bootstrap_mthreaded
include("estimators/bjs_mt.jl")
include("estimators/bjs.jl")
export fit_bjs, fit_bjs_dynamic, fit_bjs_static
include("estimators/sunab.jl")
export fit_sunab
# post
include("post/cumulative.jl")
include("post/diagnostics.jl")
export cumulative_effects, pre_trends_summary

# Plotting:
"""
    plot_event_study(m::StatisticalModel; kwargs...)

Plot event study coefficients with confidence intervals.
Requires Makie (or CairoMakie/GLMakie) to be loaded.

See extension documentation for full keyword arguments.
"""
function plot_event_study end

"""
    plot_cumulative(m::StatisticalModel; kwargs...)
    plot_cumulative(cum_result::NamedTuple; kwargs...)

Plot cumulative treatment effects.
Requires Makie (or CairoMakie/GLMakie) to be loaded.
"""
function plot_cumulative end

"""
    plot_comparison(df::DataFrame; y, id, t, g, models=[:bjs, :gardner, :sunab, :twfe], kwargs...)

Estimate multiple DiD models and plot their event study coefficients together.
Requires Makie (or CairoMakie/GLMakie) to be loaded.
"""
function plot_comparison end

# Export plotting functions
export plot_event_study, plot_cumulative, plot_comparison

# Precompilation (runs a tiny end-to-end workload at build time to cut TTFX)
include("precompile.jl")

end

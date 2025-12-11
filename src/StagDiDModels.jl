module StagDiDModels

using Logging # Turns of the printing of FixedEffectModels.jl 
using DataFrames
using StatsAPI
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
export responsename, coef, vcov, coefnames, nobs, dof_resiudual, dof, stderror, pvalue, confint,r2, islinear, adjr2, coeftable,show

# utils
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

end

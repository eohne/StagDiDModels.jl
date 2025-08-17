module DiDModels

using DataFrames
using StatsAPI
using StatsModels
using StatsModels: cols
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
include("utils/aggregation.jl")
include("utils/fe_predict.jl")


# estimators
include("estimators/twfe.jl")
include("estimators/did2s.jl")
include("estimators/bjs.jl")
include("estimators/sunab.jl")
export fit_twfe_dynamic, fit_twfe_static, fit_gardner_dynamic, fit_gardner_static, fit_bjs, fit_bjs_dynamic, fit_bjs_static, fit_sunab, fit_bjs_wild_bootstrap, fit_bjs_wild_bootstrap_mthreaded

# post
include("post/cumulative.jl")
include("post/diagnostics.jl")
export cumulative_effects, pre_trends_summary

end

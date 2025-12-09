module StagDiDModels

using Logging # Turns of the printing of FixedEffectModels.jl 
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
include("utils/fe_predict.jl")


# estimators
include("estimators/twfe.jl")
export fit_twfe_dynamic, fit_twfe_static
include("estimators/gardner.jl")
export fit_gardner_dynamic, fit_gardner_static
# include("estimators/bjs_R.jl")
# export fit_bjs, fit_bjs_dynamic, fit_bjs_static,fit_bjs_wild_bootstrap, fit_bjs_wild_bootstrap_mthreaded
include("estimators/bjs.jl")
export fit_bjs, fit_bjs_dynamic, fit_bjs_static
include("estimators/sunab.jl")
export fit_sunab
# post
include("post/cumulative.jl")
include("post/diagnostics.jl")
export cumulative_effects, pre_trends_summary

end

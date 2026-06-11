using Test
using StagDiDModels
using DataFrames, CSV
using Random

# ---------------------------------------------------------------------------
# Test data
#
# The bundled homicide panel is large (1000 units x 31 years). Estimation on
# the full panel is slow, so for CI we build a smaller subset *once* that
# preserves the panel structure (all treatment cohorts kept, fewer units per
# cohort). Every estimator is then fit on this subset, and the bulk of the
# coverage comes from exercising the full StatsAPI / display surface on the
# already-fit models -- which is essentially free (no re-estimation).
# ---------------------------------------------------------------------------
const RAW = CSV.read(joinpath(@__DIR__, "..", "data", "homicide.csv"), DataFrame)

"""
    subsample(df; n_per_cohort, seed)

Stratified, deterministic subsample: keep every treatment cohort (so the
estimators still have never-treated/not-yet-treated donors and treated units),
but only `n_per_cohort` units from each. Also attaches a synthetic numeric
control column `:x1` for exercising the `controls=` code path.
"""
function subsample(df::DataFrame; n_per_cohort::Int = 60, seed::Int = 20240611)
    rng = MersenneTwister(seed)
    keep = Int[]
    for gval in sort(unique(df.g))
        units = unique(df[df.g .== gval, :unit])
        k = min(n_per_cohort, length(units))
        append!(keep, units[randperm(rng, length(units))[1:k]])
    end
    keepset = Set(keep)
    sub = df[in.(df.unit, Ref(keepset)), :]
    sub.x1 = randn(MersenneTwister(seed + 1), nrow(sub))
    return sub
end

const df = subsample(RAW)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

"Exercise the full StatsAPI + display interface on an already-fit model."
function exercise_api(m)
    β  = coef(m)
    k  = length(β)
    @test k > 0
    @test !any(isnan, β)
    @test length(coefnames(m)) == k
    @test size(vcov(m)) == (k, k)

    se = stderror(m)
    @test length(se) == k
    @test all(x -> x >= 0 || isnan(x), se)

    @test nobs(m) > 0
    @test dof(m) == k
    @test dof_residual(m) > 0
    @test responsename(m) isa AbstractString
    @test islinear(m)

    @test length(pvalue(m)) == k

    ci = confint(m)
    @test size(ci) == (k, 2)
    @test all(ci[:, 1] .<= ci[:, 2])
    @test size(confint(m, 0.90)) == (k, 2)

    @test r2(m) isa Real
    @test adjr2(m) isa Real

    # coeftable + show drive the bulk of the struct/ display code
    @test length(sprint(show, coeftable(m))) > 0
    @test length(sprint(show, m)) > 0
    @test length(repr(m)) > 0
    return nothing
end

"Exercise post-estimation helpers that require event-time (dynamic) coefficients."
function exercise_dynamic_post(m)
    cum = cumulative_effects(m)
    @test cum isa NamedTuple
    @test !any(isnan, cum.cumulative)
    @test length(cum.τ) == length(cum.cumulative) == length(cum.std_errors)
    @test all(cum.conf_low .<= cum.conf_high)

    # explicit reference-period branch
    cum_ref = cumulative_effects(m; ref_period = -1)
    @test cum_ref isa NamedTuple

    pre = pre_trends_summary(m)
    @test pre isa NamedTuple
    @test haskey(pre, :joint_test)
    @test haskey(pre, :overall_pass)
    return nothing
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "StagDiDModels" begin

    @testset "TWFE" begin
        @testset "dynamic" begin
            m = fit_twfe_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
            exercise_api(m)
            exercise_dynamic_post(m)
        end
        @testset "static (never-treated donors)" begin
            m = fit_twfe_static(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit,
                                control_type=:never)
            exercise_api(m)
        end
        @testset "controls + weights" begin
            m = fit_twfe_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit,
                                 controls=[:x1], weights=:w)
            exercise_api(m)
        end
    end

    @testset "Gardner" begin
        @testset "dynamic" begin
            m = fit_gardner_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
            exercise_api(m)
            exercise_dynamic_post(m)
        end
        @testset "static" begin
            m = fit_gardner_static(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
            exercise_api(m)
        end
    end

    @testset "Sun-Abraham" begin
        m = fit_sunab(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
        exercise_api(m)
        exercise_dynamic_post(m)
    end

    @testset "BJS" begin
        @testset "dynamic (multithreaded)" begin
            m = fit_bjs_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
            exercise_api(m)
            exercise_dynamic_post(m)
        end
        @testset "dynamic (single-threaded)" begin
            m = fit_bjs_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit,
                                multithreaded=false)
            exercise_api(m)
        end
        @testset "static" begin
            m = fit_bjs_static(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
            exercise_api(m)
        end
    end

end

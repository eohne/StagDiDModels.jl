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

        @testset "with controls (covariate SE path)" begin
            # Exercises the control-coefficient influence/SE path and the
            # pretrend-with-controls branches, on both threading backends.
            for mt in (true, false)
                ms = fit_bjs_static(df; y=:dep_var, id=:unit, t=:year, g=:g,
                                    controls=[:x1], cluster=:unit, multithreaded=mt)
                exercise_api(ms)
                @test "x1" in coefnames(ms)

                md = fit_bjs_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g,
                                     controls=[:x1], cluster=:unit, multithreaded=mt)
                exercise_api(md)
                @test "x1" in coefnames(md)
                mt && exercise_dynamic_post(md)
            end
        end

        @testset "multiple controls + integer pretrends" begin
            # Two controls exercise the multi-covariate residualization branch;
            # integer pretrends exercises that pre-horizon construction branch.
            df2 = copy(df)
            df2.x2 = randn(MersenneTwister(7), nrow(df2))
            for mt in (true, false)
                m = fit_bjs(df2; y=:dep_var, id=:unit, t=:year, g=:g,
                            controls=[:x1, :x2], horizons=[0,1,2], pretrends=3,
                            cluster=:unit, multithreaded=mt)
                exercise_api(m)
                @test "x1" in coefnames(m)
                @test "x2" in coefnames(m)
            end
        end

        @testset "project() heterogeneity vs Stata" begin
            # Reference values from Stata did_imputation (StataNow 19) on the
            # bundled stata/bjs_testdata.csv:
            #   did_imputation y id time ei, project(x_regtrend [x_idio]) cluster(id)
            pdf = CSV.read(joinpath(@__DIR__, "bjs_testdata.csv"),
                           DataFrame; missingstring = ["", "."])

            # P1: static project(X_regtrend)
            for mt in (true, false)
                m = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                            project=[:X_regtrend], cluster=:id, multithreaded=mt)
                exercise_api(m)
                nm = coefnames(m); b = coef(m); se = stderror(m)
                @test nm == ["τ_cons", "τ_X_regtrend"]
                @test b[1]  ≈ -15.55686 rtol=1e-4
                @test se[1] ≈ 1.398945  rtol=1e-4
                @test b[2]  ≈ 0.579143  rtol=1e-4
                @test se[2] ≈ 0.0444729 rtol=1e-4
            end

            # P2: static project(X_regtrend X_idio)
            m2 = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                         project=[:X_regtrend, :X_idio], cluster=:id, multithreaded=false)
            nm2 = coefnames(m2); b2 = coef(m2); se2 = stderror(m2)
            @test nm2 == ["τ_cons", "τ_X_regtrend", "τ_X_idio"]
            @test b2[1] ≈ -15.545127 rtol=1e-4
            @test b2[2] ≈ 0.57912293 rtol=1e-4
            @test b2[3] ≈ 0.27319449 rtol=1e-4
            @test se2[3] ≈ 0.442574  rtol=1e-4

            # P3: per-horizon gradient (τ0..τ4 match Stata; τ5 differs because the
            # package, unlike Stata's minn rule, does not suppress small-sample horizons)
            m3 = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                         project=[:X_regtrend], horizons=true, cluster=:id, multithreaded=false)
            nm3 = coefnames(m3); b3 = coef(m3)
            i0 = findfirst(==("τ0_cons"), nm3)
            @test b3[i0] ≈ -12.200934 rtol=1e-4
            @test b3[findfirst(==("τ0_X_regtrend"), nm3)] ≈ 0.51150923 rtol=1e-4
        end

        @testset "hetby() heterogeneity vs Stata" begin
            # Stata: did_imputation y id time ei, hetby(region5) cluster(id)
            pdf = CSV.read(joinpath(@__DIR__, "bjs_testdata.csv"),
                           DataFrame; missingstring = ["", "."])
            target = Dict("τ_1" => (-13.446820, 0.854609),
                          "τ_2" => (-5.4731711, 0.899868),
                          "τ_3" => ( 1.3214599, 0.851860),
                          "τ_4" => ( 9.3381006, 0.849024),
                          "τ_5" => (16.600868,  0.877086))
            for mt in (true, false)
                m = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                            hetby=:region5, cluster=:id, multithreaded=mt)
                exercise_api(m)
                nm = coefnames(m); b = coef(m); se = stderror(m)
                @test Set(nm) == Set(keys(target))
                for (k, (tb, tse)) in target
                    i = findfirst(==(k), nm)
                    @test b[i]  ≈ tb  rtol=1e-4
                    @test se[i] ≈ tse rtol=1e-4
                end
            end

            # project + hetby may not be combined
            @test_throws ErrorException fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                project=[:X_regtrend], hetby=:region5, cluster=:id)
        end

        @testset "minn effective-sample suppression" begin
            pdf = CSV.read(joinpath(@__DIR__, "bjs_testdata.csv"),
                           DataFrame; missingstring = ["", "."])
            # Default minn=0 keeps every horizon (incl. the thin τ5).
            m0 = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                         project=[:X_regtrend], horizons=true, cluster=:id)
            @test "τ5_cons" in coefnames(m0)
            # minn=30 reproduces Stata's suppression of the τ5 group.
            m30 = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei,
                          project=[:X_regtrend], horizons=true, cluster=:id, minn=30)
            @test !("τ5_cons" in coefnames(m30))
            @test length(coef(m30)) == length(coef(m0)) - 2
            # surviving coefficients are unchanged by the suppression
            i0a = findfirst(==("τ0_cons"), coefnames(m0))
            i0b = findfirst(==("τ0_cons"), coefnames(m30))
            @test coef(m0)[i0a] ≈ coef(m30)[i0b] rtol=1e-8
        end

        @testset "compute_ses=false returns coefficients only" begin
            pdf = CSV.read(joinpath(@__DIR__, "bjs_testdata.csv"),
                           DataFrame; missingstring = ["", "."])
            # Across the static, dynamic, project, and controls paths, skipping the
            # variance estimation must leave the point estimates (incl. pre-trends
            # and control coefficients) bit-identical and suppress SEs to NaN.
            for kw in ((;),
                       (; horizons = true),
                       (; horizons = true, project = [:X_regtrend]),
                       (; horizons = true, controls = [:X_idio]))
                full = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei, cluster=:id, kw...)
                fast = fit_bjs(pdf; y=:Y, id=:id, t=:time, g=:Ei, cluster=:id,
                               compute_ses=false, kw...)
                @test coefnames(fast) == coefnames(full)
                @test coef(fast) == coef(full)
                @test !any(isnan, coef(fast))
                @test all(isnan, stderror(fast))
                @test all(isnan, vcov(fast))
                @test nobs(fast) == nobs(full)
                @test dof_residual(fast) == dof_residual(full)
                # show/coeftable must still render with NaN standard errors
                @test length(sprint(show, fast)) > 0
            end
        end
    end

end

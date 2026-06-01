using Test
using StagDiDModels
using DataFrames, CSV

@testset "StagDiDModels" begin

    df = CSV.read(joinpath(@__DIR__, "..", "data", "homicide.csv"), DataFrame)
    # g is already coded as 0 for never-treated
    # columns: unit, year, g, dep_var, w

    @testset "TWFE" begin
        m = fit_twfe_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
        @test length(coef(m)) > 0
        @test !any(isnan, coef(m))
    end

    @testset "Gardner" begin
        m = fit_gardner_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
        @test length(coef(m)) > 0
        @test !any(isnan, coef(m))
    end

    @testset "Sun-Abraham" begin
        m = fit_sunab(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
        @test length(coef(m)) > 0
        @test !any(isnan, coef(m))
    end

    @testset "BJS" begin
        m = fit_bjs_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
        @test length(coef(m)) > 0
        @test !any(isnan, coef(m))
    end

    @testset "Post-estimation" begin
        m = fit_gardner_dynamic(df; y=:dep_var, id=:unit, t=:year, g=:g, cluster=:unit)
        cum = cumulative_effects(m)
        @test cum isa NamedTuple
        @test !any(isnan, cum.cumulative)
        pre = pre_trends_summary(m)
        @test pre isa NamedTuple
    end

end
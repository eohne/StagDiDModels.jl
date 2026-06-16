# Benchmark: StagDiDModels BJS imputation estimator vs Stata `did_imputation`.
#
# This script (1) generates synthetic staggered-adoption panels, writing each to
# a CSV that BOTH Julia and Stata read (identical data → fair comparison), and
# (2) times our `fit_bjs` estimator on each, for both the optimized default path
# (`multithreaded=true`) and the naive path (`multithreaded=false`). Warm timings
# only (compilation excluded) — Stata's interpreted ado has no JIT, so steady
# per-call runtime is the like-for-like quantity.
#
# Matched spec (see bench_stata.do): dynamic event study, all post horizons,
# 4 pre-trends, clustered on unit, minn=0, tol=1e-6, maxit=100, autosample.

using StagDiDModels, DataFrames, CSV, Random, Printf, Statistics

const BENCHDIR = @__DIR__
const PRETRENDS = 4
const TOL = 1e-6
const MAXIT = 100   # match Stata did_imputation default maxit(100)

"Deterministic staggered panel: ~30% never-treated, rest in cohorts {5,7,9}."
function gen_panel(N::Int, T::Int; seed::Int=1)
    rng = MersenneTwister(seed)
    cohorts = (5, 7, 9)
    α = randn(rng, N)          # unit effects
    γ = randn(rng, T)          # time effects
    ids = Int[]; ts = Int[]
    Ei = Vector{Union{Int,Missing}}(); Y = Float64[]
    sizehint!(ids, N*T); sizehint!(ts, N*T); sizehint!(Ei, N*T); sizehint!(Y, N*T)
    for u in 1:N
        ei = (u % 10 < 3) ? missing : cohorts[(u % 3) + 1]
        for tt in 1:T
            treated = !ismissing(ei) && tt >= ei
            k = ismissing(ei) ? 0 : tt - ei
            y = α[u] + γ[tt] + (treated ? 0.3 * (k + 1) : 0.0) + 0.5 * randn(rng)
            push!(ids, u); push!(ts, tt); push!(Ei, ei); push!(Y, y)
        end
    end
    DataFrame(id = ids, t = ts, Ei = Ei, Y = Y)
end

fit(df; mt) = fit_bjs_dynamic(df; y=:Y, id=:id, t=:t, g=:Ei,
                              horizons=true, pretrends=PRETRENDS, cluster=:id,
                              minn=0, tol=TOL, maxiter=MAXIT, autosample=true,
                              multithreaded=mt)

"Warm up once, then return minimum elapsed over `reps` runs."
function timeit(f; reps::Int=3)
    f()                      # warmup / compile for this data shape
    best = Inf
    for _ in 1:reps
        GC.gc()
        best = min(best, @elapsed f())
    end
    best
end

coefat(m, name) = (i = findfirst(==(name), coefnames(m)); i === nothing ? NaN : coef(m)[i])

# (label, N, T)
const CASES = [("S", 1_000, 12), ("M", 4_000, 12), ("L", 10_000, 12)]

rows = NamedTuple[]
@printf("%-4s %8s %6s %6s %12s %12s %10s %10s\n",
        "case", "nobs", "ncoef", "iters", "opt_s", "naive_s", "speedup", "b(τ0)")
for (label, N, T) in CASES
    df = gen_panel(N, T)
    CSV.write(joinpath(BENCHDIR, "data_$(label).csv"), df)

    m = fit(df; mt=true)
    ncoef = length(coef(m)); b0 = coefat(m, "τ::0")
    opt   = timeit(() -> fit(df; mt=true))
    naive = timeit(() -> fit(df; mt=false))

    push!(rows, (case=label, nobs=nrow(df), ncoef=ncoef,
                 opt_s=opt, naive_s=naive, b0=b0))
    @printf("%-4s %8d %6d %6s %12.4f %12.4f %9.2fx %10.4f\n",
            label, nrow(df), ncoef, "-", opt, naive, naive/opt, b0)
end

CSV.write(joinpath(BENCHDIR, "results_julia.csv"), DataFrame(rows))
println("\nWrote data_*.csv and results_julia.csv to $BENCHDIR")

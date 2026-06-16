# Benchmark: StagDiDModels BJS imputation estimator — serial vs threaded, and
# vs Stata `did_imputation` (timed separately in bench_stata.do on the same CSVs).
#
# Tracks BOTH wall-clock and allocations. The threading parallelizes only the
# influence-weight orthogonalization (FE groups / W columns), which adds no
# allocations; so threaded vs serial bytes should be ~equal — we report both to
# confirm that. Run with multiple threads to see the speedup, e.g.:
#     julia --project=. -t 16 bench/run_bench_julia.jl
#
# Warm timings only (compilation excluded). Matched-to-Stata spec: all post
# horizons, 4 pre-trends, clustered on unit, minn=0, tol=1e-6, maxit=100.

using StagDiDModels, DataFrames, CSV, Random, Printf, Statistics

const BENCHDIR = @__DIR__
const PRETRENDS = 4
const TOL = 1e-6
const MAXIT = 100

"Deterministic staggered panel. `x` is drawn from a SEPARATE rng so it does not
perturb the `Y` stream — the Y column (and thus the Stata comparison) is
identical whether or not the control is present."
function gen_panel(N::Int, T::Int; seed::Int=1)
    rng = MersenneTwister(seed)
    rngx = MersenneTwister(seed + 10_000)
    cohorts = (5, 7, 9)
    α = randn(rng, N); γ = randn(rng, T)
    ids = Int[]; ts = Int[]
    Ei = Vector{Union{Int,Missing}}(); Y = Float64[]; X = Float64[]
    for u in 1:N
        ei = (u % 10 < 3) ? missing : cohorts[(u % 3) + 1]
        for tt in 1:T
            treated = !ismissing(ei) && tt >= ei
            k = ismissing(ei) ? 0 : tt - ei
            push!(ids, u); push!(ts, tt); push!(Ei, ei)
            push!(Y, α[u] + γ[tt] + (treated ? 0.3 * (k + 1) : 0.0) + 0.5 * randn(rng))
            push!(X, randn(rngx))
        end
    end
    DataFrame(id = ids, t = ts, Ei = Ei, Y = Y, x = X)
end

fit(df; mt, controls=Symbol[]) =
    fit_bjs_dynamic(df; y=:Y, id=:id, t=:t, g=:Ei,
                    horizons=true, pretrends=PRETRENDS, cluster=:id,
                    controls=controls, minn=0, tol=TOL, maxiter=MAXIT,
                    autosample=true, multithreaded=mt)

"Warm up, then return (min time over reps, bytes allocated)."
function measure(f; reps::Int=3)
    f()                                  # warmup / compile for this data shape
    best = Inf; bytes = 0
    for _ in 1:reps
        GC.gc()
        r = @timed f()
        best = min(best, r.time)
        bytes = r.bytes
    end
    (time = best, bytes = bytes)
end

coefat(m, name) = (i = findfirst(==(name), coefnames(m)); i === nothing ? NaN : coef(m)[i])
mb(b) = b / 2^20

const CASES = [("S", 1_000, 12), ("M", 4_000, 12), ("L", 10_000, 12)]

println("Julia threads = $(Threads.nthreads())\n")
rows = NamedTuple[]
for ctrls in ([Symbol[], "no controls"], [[:x], "with control x"])
    cols, lbl = ctrls
    @printf("=== %s ===\n", lbl)
    @printf("%-4s %8s %6s %10s %10s %9s %12s %12s\n",
            "case", "nobs", "ncoef", "serial_s", "thread_s", "speedup",
            "serial_MB", "thread_MB")
    for (label, N, T) in CASES
        df = gen_panel(N, T)
        # data_*.csv (Y matches the Stata runs) — written once on the no-controls pass.
        isempty(cols) && CSV.write(joinpath(BENCHDIR, "data_$(label).csv"), df)

        m = fit(df; mt=true, controls=cols)
        ncoef = length(coef(m)); b0 = coefat(m, "τ::0")
        ser = measure(() -> fit(df; mt=false, controls=cols))
        thr = measure(() -> fit(df; mt=true,  controls=cols))

        push!(rows, (scenario=lbl, case=label, nobs=nrow(df), ncoef=ncoef, b0=b0,
                     serial_s=ser.time, thread_s=thr.time,
                     serial_MB=mb(ser.bytes), thread_MB=mb(thr.bytes)))
        @printf("%-4s %8d %6d %10.4f %10.4f %8.2fx %12.1f %12.1f\n",
                label, nrow(df), ncoef, ser.time, thr.time,
                ser.time/thr.time, mb(ser.bytes), mb(thr.bytes))
    end
    println()
end

CSV.write(joinpath(BENCHDIR, "results_julia.csv"), DataFrame(rows))
println("Wrote data_*.csv and results_julia.csv to $BENCHDIR")

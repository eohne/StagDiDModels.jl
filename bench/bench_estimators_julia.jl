# Time Gardner / Sun-Abraham / TWFE (dynamic event study, clustered on unit) on
# the shared data_*.csv panels, for comparison with the R equivalents
# (did2s / fixest::sunab / fixest::feols) timed in bench_estimators_r.R.
# Warm timings (compilation excluded). Reports the horizon-0 effect for parity.

using StagDiDModels, DataFrames, CSV, Printf

const BENCHDIR = @__DIR__

function measure(f; reps::Int=3)
    f()
    best = Inf
    for _ in 1:reps
        GC.gc(); best = min(best, @elapsed f())
    end
    best
end

# horizon-0 coefficient (event time 0) from any of our event-study models
function coef0(m)
    nms = coefnames(m); b = coef(m)
    for (i, nm) in enumerate(nms)
        mm = match(r"::(-?\d+)$", nm)
        if mm !== nothing && parse(Int, mm.captures[1]) == 0
            return b[i]
        end
    end
    return NaN
end

const CASES = [("S", "data_S.csv"), ("M", "data_M.csv"), ("L", "data_L.csv")]
const SPECS = [
    ("gardner", df -> fit_gardner_dynamic(df; y=:Y, id=:id, t=:t, g=:Ei, cluster=:id)),
    ("sunab",   df -> fit_sunab(df;          y=:Y, id=:id, t=:t, g=:Ei, cluster=:id)),
    ("twfe",    df -> fit_twfe_dynamic(df;    y=:Y, id=:id, t=:t, g=:Ei, cluster=:id)),
]

rows = NamedTuple[]
@printf("%-8s %-4s %8s %10s %10s\n", "estim", "case", "nobs", "julia_s", "b(0)")
for (label, file) in CASES
    df = CSV.read(joinpath(BENCHDIR, file), DataFrame)
    for (name, f) in SPECS
        m = f(df)
        s = measure(() -> f(df))
        push!(rows, (estimator=name, case=label, nobs=nrow(df), julia_s=s, b0=coef0(m)))
        @printf("%-8s %-4s %8d %10.4f %10.4f\n", name, label, nrow(df), s, coef0(m))
    end
end
CSV.write(joinpath(BENCHDIR, "results_estimators_julia.csv"), DataFrame(rows))
println("\nWrote results_estimators_julia.csv")

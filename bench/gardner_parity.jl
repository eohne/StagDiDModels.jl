# Golden-baseline parity harness for the Gardner SE optimization.
#   First run (no golden file present): fits the CURRENT Gardner on several
#     panels and serializes (coef, vcov) per spec to gardner_golden.jls.
#   Later runs (golden present): re-fits and reports max |Δcoef| / |Δvcov| vs
#     the golden — the "retain exact same outputs" check.
using StagDiDModels, DataFrames, Random, Serialization, Printf

const GOLDEN = joinpath(@__DIR__, "gardner_golden.jls")

function genpanel(N, T, cohorts; seed=1)
    rng = MersenneTwister(seed); rx = MersenneTwister(seed + 7)
    ids = Int[]; ts = Int[]; Ei = Union{Int,Missing}[]
    Y = Float64[]; X = Float64[]; W = Float64[]; grp = Int[]
    α = randn(rng, N); γ = randn(rng, T)
    for u in 1:N
        ei = (u % 4 == 0) ? missing : cohorts[(u % length(cohorts)) + 1]
        for tt in 1:T
            tr = !ismissing(ei) && tt >= ei
            k = ismissing(ei) ? 0 : tt - ei
            push!(ids, u); push!(ts, tt); push!(Ei, ei)
            push!(Y, α[u] + γ[tt] + (tr ? 0.3 * (k + 1) : 0.0) + 0.4 * randn(rng))
            push!(X, randn(rx)); push!(W, 0.5 + abs(randn(rx))); push!(grp, (u % 10) + 1)
        end
    end
    DataFrame(id=ids, t=ts, Ei=Ei, Y=Y, x=X, w=W, grp=grp)
end

# (label, fit closure) — varied: controls, weights, alt cluster, static
const DF1 = genpanel(100, 8, (3, 5); seed=1)
const DF2 = genpanel(120, 8, (3, 5, 7); seed=2)
const DF3 = genpanel(150, 10, (4, 6); seed=3)
const DF4 = genpanel(120, 9, (3, 5, 7); seed=4)
const DF5 = genpanel(200, 10, (4, 6, 8); seed=5)
const DF6 = genpanel(100, 8, (3, 5); seed=6)
const DF7 = genpanel(700, 10, (4, 6, 8); seed=7)   # larger: p≈710, exercises the sparse solve

specs = [
    ("dyn_plain",     () -> fit_gardner_dynamic(DF1; y=:Y, id=:id, t=:t, g=:Ei, cluster=:id)),
    ("dyn_control",   () -> fit_gardner_dynamic(DF2; y=:Y, id=:id, t=:t, g=:Ei, controls=[:x], cluster=:id)),
    ("dyn_weights",   () -> fit_gardner_dynamic(DF3; y=:Y, id=:id, t=:t, g=:Ei, weights=:w, cluster=:id)),
    ("dyn_cluster",   () -> fit_gardner_dynamic(DF4; y=:Y, id=:id, t=:t, g=:Ei, cluster=:grp)),
    ("dyn_ctrl_wt",   () -> fit_gardner_dynamic(DF5; y=:Y, id=:id, t=:t, g=:Ei, controls=[:x], weights=:w, cluster=:id)),
    ("dyn_large",     () -> fit_gardner_dynamic(DF7; y=:Y, id=:id, t=:t, g=:Ei, cluster=:id)),
    ("static_plain",  () -> fit_gardner_static(DF6;  y=:Y, id=:id, t=:t, g=:Ei, cluster=:id)),
]

results = Dict{String,Any}()
for (lbl, f) in specs
    m = f()
    results[lbl] = (coef=collect(coef(m)), vcov=Matrix(vcov(m)), names=coefnames(m))
end

if !isfile(GOLDEN)
    serialize(GOLDEN, results)
    println("Captured golden baseline for $(length(results)) specs -> $GOLDEN")
else
    let gold = deserialize(GOLDEN), maxdc = 0.0, maxdv = 0.0
        for (lbl, r) in sort(collect(results); by=first)
            g = gold[lbl]
            dc = maximum(abs.(r.coef .- g.coef))
            dv = maximum(abs.(r.vcov .- g.vcov))
            maxdc = max(maxdc, dc); maxdv = max(maxdv, dv)
            @printf("%-14s Δcoef=%.3e  Δvcov=%.3e  (k=%d)\n", lbl, dc, dv, length(r.coef))
        end
        @printf("MAX Δcoef=%.3e  MAX Δvcov=%.3e  => %s\n", maxdc, maxdv,
                (maxdc == 0 && maxdv == 0) ? "BIT-IDENTICAL" :
                (max(maxdc, maxdv) < 1e-9 ? "numerically identical (<1e-9)" : "*** CHANGED ***"))
    end
end

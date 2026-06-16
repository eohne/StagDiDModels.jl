# One BJS fit on data_L.csv, reporting MEMORY (not allocation traffic):
#   - Sys.maxrss(): OS peak resident set of the whole process (incl. runtime +
#     package image + JIT — Julia's fixed overhead Stata doesn't have).
#   - gc_live_bytes(): live Julia heap after the fit (the algorithm's actual
#     working footprint, the closest analogue to "data + matrices in memory").
using StagDiDModels, DataFrames, CSV
df = CSV.read(joinpath(@__DIR__, "data_L.csv"), DataFrame)
spec() = fit_bjs_dynamic(df; y=:Y, id=:id, t=:t, g=:Ei, horizons=true,
                         pretrends=4, cluster=:id, minn=0, tol=1e-6, maxiter=100,
                         multithreaded=true)
spec()                                   # warmup / compile
GC.gc()
live_before = Base.gc_live_bytes()
spec()                                   # the measured fit
GC.gc()
mb(x) = round(Int, x / 2^20)
println("nobs = ", nrow(df), ", threads = ", Threads.nthreads())
println("Julia peak RSS (whole process) = ", mb(Sys.maxrss()), " MB")
println("Julia live heap (data+model)   = ", mb(Base.gc_live_bytes()), " MB",
        "  (baseline before fit: ", mb(live_before), " MB)")

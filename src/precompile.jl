using PrecompileTools: @setup_workload, @compile_workload
using Logging: with_logger, SimpleLogger, Error

@setup_workload begin
    # Tiny synthetic staggered-adoption panel: 16 units × 8 periods.
    #   units  1-5   : treated from t=4
    #   units  6-10  : treated from t=6
    #   units 11-16  : never treated (g=0)
    # Never-treated units span every period, so every time/unit FE is imputable
    # and the autosample / pretrend / control paths all exercise. Columns:
    #   x  – continuous control / projection covariate
    #   hb – discrete subgroup for hetby
    #   w  – observation weights
    ids = Int[]; ts = Int[]; gs = Int[]
    ys = Float64[]; xs = Float64[]; hbs = Int[]; ws = Float64[]
    for u in 1:16
        g = u <= 5 ? 4 : (u <= 10 ? 6 : 0)
        for tt in 1:8
            treated = g > 0 && tt >= g
            x = 0.5 * u - 0.3 * tt
            # Deterministic but non-collinear idiosyncratic term so the
            # regressions have nonzero residuals (keeps cluster vcov well-posed).
            noise = 0.4 * sin(1.3 * u + 0.7 * tt)
            push!(ids, u); push!(ts, tt); push!(gs, g)
            push!(xs, x); push!(hbs, u % 2); push!(ws, 1.0 + 0.1 * (u % 3))
            push!(ys, 0.1 * u + 0.5 * tt + (treated ? 2.0 : 0.0) + 0.3 * x + noise)
        end
    end

    @compile_workload begin
        # Tiny degenerate data makes FixedEffectModels/Vcov emit cosmetic
        # warnings; silence them so package installs stay clean. `quiet_reg`
        # installs its own Warn-level logger internally, so on top of the Error
        # logger we also redirect stderr for the duration of the workload. Each
        # estimator call is isolated so a degenerate run never aborts the rest.
        redirect_stderr(devnull) do; with_logger(SimpleLogger(stderr, Error)) do
            df = DataFrame(id = ids, t = ts, g = gs, y = ys, x = xs, hb = hbs, w = ws)
            models = Any[]
            run!(f) = try push!(models, f()) catch err
                @debug "DiDModels precompile step failed" exception=err
            end

            # ---- BJS: multithreaded path (default) ----
            run!(() -> fit_bjs_dynamic(df; y=:y, id=:id, t=:t, g=:g))
            run!(() -> fit_bjs_dynamic(df; y=:y, id=:id, t=:t, g=:g,
                                       controls=[:x], cluster=:id, weights=:w))
            run!(() -> fit_bjs_dynamic(df; y=:y, id=:id, t=:t, g=:g, project=[:x]))
            run!(() -> fit_bjs_dynamic(df; y=:y, id=:id, t=:t, g=:g, hetby=:hb))
            run!(() -> fit_bjs_static(df;  y=:y, id=:id, t=:t, g=:g, minn=2))

            # ---- Gardner ----
            run!(() -> fit_gardner_dynamic(df; y=:y, id=:id, t=:t, g=:g))
            run!(() -> fit_gardner_static(df;  y=:y, id=:id, t=:t, g=:g, controls=[:x]))

            # ---- Sun-Abraham (both aggregations) ----
            run!(() -> fit_sunab(df; y=:y, id=:id, t=:t, g=:g, cluster=:id))
            run!(() -> fit_sunab(df; y=:y, id=:id, t=:t, g=:g, agg=:att))

            # ---- TWFE (dynamic, binned, static, controls) ----
            run!(() -> fit_twfe_dynamic(df; y=:y, id=:id, t=:t, g=:g, cluster=:id))
            run!(() -> fit_twfe_dynamic(df; y=:y, id=:id, t=:t, g=:g,
                                        bin_leads=2, bin_lags=2))
            run!(() -> fit_twfe_static(df;  y=:y, id=:id, t=:t, g=:g, controls=[:x]))

            # ---- StatsAPI accessors + heavy custom show + post-estimation ----
            for m in models
                try
                    coeftable(m); stderror(m); pvalue(m); confint(m)
                    sprint(show, m)
                catch err
                    @debug "DiDModels precompile accessor failed" exception=err
                end
            end
            # Event-study post-processing (cumulative + pretrend diagnostics).
            for m in models
                try
                    cumulative_effects(m)
                    redirect_stdout(devnull) do
                        pre_trends_summary(m)
                    end
                catch err
                    @debug "DiDModels precompile post step failed" exception=err
                end
            end
        end end
    end
end

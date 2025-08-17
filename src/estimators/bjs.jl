"""
    build_wtr!(df; ttt::Symbol=:ttt, treat::Symbol=:__treat, weights::Union{Nothing,Symbol}=nothing,
               horizons::Union{Nothing,Bool,Vector{Int}}=nothing, prefix::String="wtr")

Creates treatment weights:
- If `horizons === nothing` -> static ATT weights: a single column `\${prefix}treat` = 1{treated}.
- If `horizons === true`    -> dynamic: one column per nonnegative event time present.
- If `horizons::Vector{Int}`-> dynamic for those horizons (keeps τ ≥ 0 only).
Weights are normalized to sum to 1 over treated rows, and (optionally) multiplied by `weights` before normalization.
Returns the vector of created column Symbols, in order.
"""
function build_wtr!(df::DataFrame; ttt::Symbol=:ttt, treat::Symbol=:__treat,
                    weights::Union{Nothing,Symbol}=nothing,
                    horizons::Union{Nothing,Bool,Vector{Int}}=nothing,
                    prefix::String="wtr")

    # base scalar weights
    basew = isnothing(weights) ? ones(Float64, nrow(df)) : Float64.(df[!, weights])

    if horizons === nothing
        # Static case:
        nm = Symbol(prefix * "static")
        df[!, nm] = (df[!, treat] .== 1) .* basew
        s = sum(df[!, nm])
        if s > 0; df[!, nm] ./= s; end
        return [nm]
    else
        # Dynamic case:
        τvals = if horizons === true
            unique(df[!, ttt])
        else
            collect(horizons)
        end
        τkeep = sort!(unique(filter(≥(0), τvals)))
        cols = Symbol[]
        
        for τ in τkeep
            nm = Symbol(prefix * string(τ))
            # R logic: weight = base_weight if ttt == τ, else 0
            df[!, nm] = ifelse.(df[!, ttt] .== τ, basew, 0.0)
            push!(cols, nm)
        end
        
        if isempty(cols)
            @warn "No treatment weight columns created - no valid horizons found"
            return Symbol[]
        end
        
        # normalization: each column sums to 1 over ALL observations
        for c in cols
            s = sum(df[!, c]; init=0.0)
            if s > 0
                df[!, c] ./= s
            end
        end
        
        return cols
    end
end


"""
    fit_bjs(df; y, id, t, g;
                       controls = Symbol[],
                       fe::Tuple = (id, t),
                       weights::Union{Nothing,Symbol} = nothing,
                       cluster::Union{Nothing,Symbol} = nothing,
                       horizons::Union{Nothing,Bool,Vector{Int}} = nothing,
                       pretrends::Union{Nothing,Bool,Vector{Int}} = nothing,
                       control_type::Symbol = :notyet)

BJS **imputation** estimator (static or dynamic) with optional pretrend testing, 
following Borusyak, Jaravel, and Spiess (2021).

# Description

The imputation-based estimator is a method of calculating treatment effects
in a difference-in-differences framework. The method estimates a model for
Y(0) using untreated/not-yet-treated observations and predicts Y(0) for the
treated observations ŷ(0). The difference between treated and
predicted untreated outcomes Y(1) - ŷ(0) serves as an estimate
for the treatment effect for unit i in period t. These are then averaged to
form average treatment effects for groups of {it}.

# Arguments

- `df::DataFrame`: Panel dataset
- `y::Symbol`: Outcome variable name
- `id::Symbol`: Unit identifier variable name  
- `t::Symbol`: Time period variable name
- `g::Symbol`: Treatment timing variable name (when unit first receives treatment)

# Keyword Arguments

- `controls::Vector{Symbol}=Symbol[]`: Control variables to include in first stage
- `fe::Tuple=(id, t)`: Fixed effects to include in first stage (default: unit and time FE)
- `weights::Union{Nothing,Symbol}=nothing`: Observation weights variable name
- `cluster::Union{Nothing,Symbol}=nothing`: Clustering variable (default: clusters by `id`)
- `horizons::Union{Nothing,Bool,Vector{Int}}=nothing`: Event study horizons to estimate
  - `nothing`: Static ATT estimation
  - `true`: All available post-treatment periods (τ ≥ 0)  
  - `Vector{Int}`: Specific horizons to estimate
- `pretrends::Union{Nothing,Bool,Vector{Int}}=nothing`: Pre-treatment periods for testing
  - `nothing` or `false`: No pretrend testing
  - `true`: All available pre-treatment periods (τ < 0)
  - `Vector{Int}`: Specific pre-treatment periods to test
- `control_type::Symbol=:notyet`: Donor sample definition
  - `:notyet`: Not-yet-treated + never-treated units
  - `:never`: Never-treated units only

# Data Conventions

Never-treated units must be coded as `g == 0` or `g == missing`.
Do NOT use `Inf` or out-of-sample dates for never-treated units.

# Implementation Details

1. **First Stage**: Estimates Y(0) using donor observations via fixest-style FE regression
2. **Prediction**: Uses robust `fe_predict` that handles missing FE levels gracefully  
3. **Treatment Effects**: Computes weighted averages of Y(1) - ŷ(0) for treated observations
4. **Standard Errors**: Analytic influence function approach exactly matching R `didimputation`

The implementation produces identical results to the R `didimputation` package.

# Returns

`SimpleModel` containing:
- Point estimates (static ATT or event study coefficients)
- Analytic standard errors with cluster-robust inference
- Compatible with all `StatsAPI` functions (`coef`, `vcov`, `coefnames`, etc.)

# Examples

```julia
# Static ATT estimation
static_model = fit_bjs(df; y=:outcome, id=:unit, t=:year, g=:treatment_year)

# Dynamic event study (all post periods)  
dynamic_model = fit_bjs(df; y=:outcome, id=:unit, t=:year, g=:treatment_year,
                                   horizons=true, cluster=:state)

# Event study with pretrends testing
full_model = fit_bjs(df; y=:outcome, id=:unit, t=:year, g=:treatment_year,
                                horizons=0:5, pretrends=[-3,-2,-1], 
                                controls=[:covariate1, :covariate2],
                                cluster=:state)

# Access results
coef(dynamic_model)                    # Point estimates
sqrt.(diag(vcov(dynamic_model)))       # Standard errors  
coefnames(dynamic_model)               # Coefficient names
```

# References

Borusyak, K., Jaravel, X., & Spiess, J. (2021). Revisiting event study designs: 
Robust and efficient estimation. Working paper.
"""
function fit_bjs(df::DataFrame; 
                            y::Symbol, 
                            id::Symbol, t::Symbol, g::Symbol,
                            controls::Vector{Symbol}=Symbol[],
                            fe::Tuple{Vararg{Symbol}}=(id, t),
                            weights::Union{Nothing,Symbol}=nothing,
                            cluster::Union{Nothing,Symbol}=id,
                            horizons::Union{Nothing,Bool,Vector{Int}}=nothing,
                            pretrends::Union{Nothing,Bool,Vector{Int}}=nothing,
                            control_type::Symbol=:notyet)

    d = copy(df)
    d[ismissing.(d[!, g]), g] .= 0
    
    # Missing data handling
    key_vars = [y, id, t, g]
    if !isnothing(weights); push!(key_vars, weights); end
    if !isnothing(cluster); push!(key_vars, cluster); end
    append!(key_vars, controls)
    append!(key_vars, collect(fe))
    key_vars = unique(key_vars)
    
    missing_mask = falses(nrow(d))
    for var in key_vars
        missing_mask .|= ismissing.(d[!, var])
    end
    
    if any(missing_mask)
        d = d[.!missing_mask, :]
    end

    n = nrow(d)
    if n == 0
        error("No observations remaining after dropping missing values")
    end

    # Treatment indicator and event time
    treat = (.!ismissing.(d[!, g])) .& (d[!, g] .> 0) .& (d[!, t] .>= d[!, g])
    d[!, :__treat] = Int.(treat)
    make_eventtime!(d; t=t, g=g, new=:ttt, ref_p=-1_000_000)

    # Get list of event times
    event_times = sort(unique(d[d[!, :ttt] .!= -1_000_000, :ttt]))

    # First stage on donors only
    donors = donor_mask(d; t=t, g=g, control_type=control_type)

    if sum(donors) == 0
        error("No donor observations found for first stage estimation")
    end

    w_vec = isnothing(weights) ? nothing : Float64.(d[!, weights])

    # Build first stage formula
    if isempty(controls)
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ fe_terms_sum
    else
        control_terms = sum(term.(controls))
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ control_terms + fe_terms_sum
    end
    
    # Fit first stage model
    m1 = reg(d[donors, :], f1, Vcov.robust(); 
             weights = weights === nothing ? nothing : weights,
             save = :fe)
    
    # Y(0) prediction
    y0hat = fe_predict(m1, d)
    
    # Handle any missing predictions
    if any(ismissing, y0hat)
        y0hat = coalesce.(y0hat, 0.0)
    end
    
    y0hat = Float64.(y0hat)

    # Create adjusted outcome
    d[!, :__adj] = Float64.(d[!, y]) .- y0hat
    
    # Drop rows with missing adjusted outcomes
    adj_missing = ismissing.(d[!, :__adj])
    if any(adj_missing)
        d = d[.!adj_missing, :]
        treat = treat[.!adj_missing]
        n = nrow(d)
        if !isnothing(w_vec); w_vec = w_vec[.!adj_missing]; end
    end

    if n == 0
        error("No observations remaining after adjustment")
    end

    # Main BJS estimation (treatment effects)
    main_β = Float64[]
    main_Σ = Matrix{Float64}(undef, 0, 0)
    main_names = String[]
    treat_mask = d[!, :__treat] .== 1
    if horizons !== nothing || isnothing(pretrends)
        # Build treatment weights
        wtr_syms = build_wtr!(d; ttt=:ttt, treat=:__treat, weights=weights, 
                              horizons=horizons, prefix="__wtr")
        k = length(wtr_syms)
        
        if k > 0
            WTR = reduce(hcat, (Float64.(d[!, s]) for s in wtr_syms))  # n × k
            
            # Point estimates - use only treated observations
            treat_mask = d[!, :__treat] .== 1
            
            if sum(treat_mask) == 0
                error("No treated observations found for estimation")
            end
            
            WTR_treated = WTR[treat_mask, :]
            adj_treated = Float64.(d[treat_mask, :__adj])

            # Point estimates using R formula
            main_β = vec(sum(WTR_treated .* adj_treated, dims=1))
            
            # Analytic SEs (BJS influence function)
            idx1 = findall(treat_mask)
            idx0 = findall(.!treat_mask)

            if isempty(idx1)
                error("No treated observations found")
            end
            if isempty(idx0)
                error("No untreated observations found")
            end

            # Build design matrix for first stage
            if isempty(controls)
                Z = sparse_rhs_fe(d; controls=Symbol[], fe=fe)
            else
                Z = sparse_rhs_fe(d; controls=controls, fe=fe)
            end

            # Weight vector for influence function
            wdat = isnothing(w_vec) ? ones(Float64, n) : Float64.(w_vec)
            Zw  = spdiagm(0 => wdat) * Z
            Z1w = Zw[idx1, :]
            Z0w = Zw[idx0, :]

            # Cross-products
            S_Z0Z0 = Matrix(Z0w' * Z0w)

            WTR1 = WTR[idx1, :]
            Z1_wtr = Matrix(Z1w' * WTR1)

            # v* = -Z * (S_Z0Z0 \ (Z1' WTR1))  # match R::solve(S_Z0Z0, Z1_wtr)
            vstar = - Matrix(Zw * (S_Z0Z0 \ Z1_wtr))


            # Fix v* on treated rows: set equal to WTR
            vstar[idx1, :] .= WTR1


            # Add vstar columns to dataframe 
            for j in 1:k
                d[!, Symbol("zz000v$j")] = vstar[:, j]
            end

            vcols = [Symbol("zz000v$j") for j in 1:k]
            tcols = [Symbol("zz000tau_et$j") for j in 1:k]

            # Initialize tau_et columns
            for j in 1:k
                d[!, tcols[j]] = zeros(Float64, nrow(d))
            end

            # Calculate group means BY (g, event_time) and assign properly
            for group_df in groupby(d, [g, :ttt])
                if nrow(group_df) == 0; continue; end
                
                g_val = group_df[1, g]
                τ_val = group_df[1, :ttt]
                group_mask = (d[!, g] .== g_val) .& (d[!, :ttt] .== τ_val)
                
                for j in 1:k
                    x = d[group_mask, vcols[j]]
                    adj_vals = d[group_mask, :__adj]
                    treat_vals = d[group_mask, :__treat]
                    
                    # R formula: sum(x^2 * adj) / sum(x^2) * treat
                    numerator = sum(x.^2 .* adj_vals)
                    denominator = sum(x.^2)
                    
                    if denominator > eps()
                        group_mean_base = numerator / denominator
                        group_values = group_mean_base .* treat_vals
                    else
                        group_values = zeros(sum(group_mask))
                    end
                    
                    d[group_mask, tcols[j]] = group_values
                end
            end

            # Recenter: adj - τ̄_{et}
            for j in 1:k
                tau_vals = d[!, tcols[j]]
                tau_vals[ismissing.(tau_vals)] .= 0.0
                d[!, tcols[j]] = d[!, :__adj] .- tau_vals
            end

            # Cluster variance calculation - EXACT R approach
            cl = isnothing(cluster) ? d[!, id] : d[!, cluster]
            cluster_contributions = zeros(length(unique(cl)), k)
            cluster_ids = unique(cl)

            for (c_idx, c_id) in enumerate(cluster_ids)
                cluster_mask = cl .== c_id
                cluster_data = d[cluster_mask, :]
                
                for j in 1:k
                    v_col = cluster_data[!, vcols[j]]
                    tau_col = cluster_data[!, tcols[j]]
                    cluster_sum = sum(v_col .* tau_col)
                    cluster_contributions[c_idx, j] = cluster_sum^2
                end
            end

            # Standard errors: sqrt(sum(x)) across clusters
            ses = sqrt.(sum(cluster_contributions, dims=1))[:]
            main_Σ = Diagonal(ses.^2)

            # Clean up temporary columns
            for j in 1:k
                select!(d, Not(vcols[j]))
                select!(d, Not(tcols[j]))
            end
            
            # Build main coefficient names
            main_names = if horizons === nothing
                ["_ATT"]
            else
                τkeep = parse.(Int, replace.(String.(wtr_syms), "__wtr" => ""))
                "τ::" .* string.(τkeep)
            end
        end
    end

    # Pretrend estimation (TWFE on untreated)
    pre_β = Float64[]
    pre_Σ = Matrix{Float64}(undef, 0, 0)
    pre_names = String[]

    if !isnothing(pretrends) && pretrends !== false
        # determine which negative τ’s to keep (exactly like R’s keep=…)
        pretrend_periods = pretrends === true ? filter(τ -> τ < 0, event_times) : filter(τ -> τ < 0, pretrends)

        if !isempty(pretrend_periods)
            d_untreated = d[.!treat_mask, :]
            if nrow(d_untreated) > 0
                # 1) build explicit indicator columns ONLY for the requested τ’s
                #    (mimics fixest::i(zz000event_time, keep = pretrend_periods))
                pre_syms = Symbol[]
                for τ in pretrend_periods
                    s = Symbol("__pre_", τ)            # e.g., __pre_-5
                    push!(pre_syms, s)
                    d_untreated[!, s] = Float64.(d_untreated[!, :ttt] .== τ)
                end

                # 2) construct RHS = sum of kept dummies + controls + FE
                control_terms = isempty(controls) ? nothing : sum(term.(controls))
                fe_terms_sum  = length(fe) == 0 ? nothing : reduce(+, [FixedEffectModels.fe(s) for s in fe])
                pre_terms_sum = sum(term.(pre_syms))  # sum of explicit τ dummies

                rhs_list = Any[pre_terms_sum]
                if control_terms !== nothing; push!(rhs_list, control_terms); end
                if fe_terms_sum  !== nothing; push!(rhs_list, fe_terms_sum);  end
                rhs_pre = reduce(+, rhs_list)

                f_pre = Term(y) ~ rhs_pre
                cluster_vcov = build_cluster_vcov(isnothing(cluster) ? id : cluster)
                w_untreated = isnothing(w_vec) ? nothing : w_vec[.!treat_mask]

                pre_est = reg(d_untreated, f_pre, cluster_vcov;weights = weights)

                
                cn       = StatsAPI.coefnames(pre_est)
                coefs    = StatsAPI.coef(pre_est)
                vcov_mat = StatsAPI.vcov(pre_est)

                # map symbol -> column index
                idx_map = Dict{Symbol, Int}()
                for (i, nm) in pairs(cn)
                    try
                        idx_map[Symbol(nm)] = i
                    catch
                        # ignore non-symbol names (e.g., FE)
                    end
                end

                pre_indices = [ get(idx_map, Symbol("__pre_", τ), nothing) for τ in pretrend_periods ]
                # drop τ’s that didn’t appear in untreated data (no column created)
                keep_mask = map(!isnothing, pre_indices)
                pre_indices = collect(skipmissing(pre_indices))

                if !isempty(pre_indices)
                    pre_β = coefs[pre_indices]
                    pre_Σ = vcov_mat[pre_indices, pre_indices]
                    pre_names = ["τ::$(τ)" for (τ, keep) in zip(pretrend_periods, keep_mask) if keep]
                end

            end
        end
    end


    # Combine results and order by τ
    if isempty(pre_β) && isempty(main_β)
        error("No coefficients estimated")
    end
    
    all_β = vcat(pre_β, main_β)
    all_names = vcat(pre_names, main_names)
    
    # Build combined covariance matrix (block diagonal)
    k_pre = length(pre_β)
    k_main = length(main_β)
    k_total = k_pre + k_main
    
    Σ_combined = zeros(k_total, k_total)
    if k_pre > 0
        Σ_combined[1:k_pre, 1:k_pre] = pre_Σ
    end
    if k_main > 0
        Σ_combined[(k_pre+1):k_total, (k_pre+1):k_total] = main_Σ
    end
    
    # Order by τ (ascending)
    τ_values = Float64[]
    if length(all_names) > 1 && any(contains.(all_names, "τ::"))
        
        for name in all_names
            if contains(name, "τ::")
                τ_str = replace(name, "τ::" => "")
                push!(τ_values, parse(Float64, τ_str))
            elseif name == "_ATT"
                push!(τ_values, Inf)
            else
                push!(τ_values, 0.0)
            end
        end
        
        sort_order = sortperm(τ_values)
        all_β = all_β[sort_order]
        all_names = all_names[sort_order]
        Σ_combined = Σ_combined[sort_order, sort_order]
    end
    
    dof = max(n - length(all_β), 1)
     # Return proper BJSModel
    return BJSModel(
        all_β, Σ_combined, all_names, n, dof;
        y_name = y,
        estimator_type = length(all_β) >1 ? :event_study : :static,
        treatment_periods = length(τ_values) > 0 ? Int.(τ_values) : Int[],
        first_stage_r2 = r2(m1),
        n_treated = sum(treat),
        n_donors = sum(donors),
        control_type = control_type,
        boot=false,n_boot=0,cluster=cluster
    )
end



"""
    fit_bjs_static(df; y, id, t, g; kwargs...)

Static ATT alias for `fit_bjs(...; horizons=nothing)`.
Returns a `SimpleModel` with a single `"_ATT"` coefficient.
"""
function fit_bjs_static(df::DataFrame; 
                        y::Symbol, 
                        id::Symbol, t::Symbol, g::Symbol,
                        controls::Vector{Symbol}=Symbol[], 
                        fe::Tuple{Vararg{Symbol}}=(id,t),
                        weights::Union{Nothing,Symbol}=nothing,
                        cluster::Union{Nothing,Symbol}=id,
                        control_type::Symbol=:notyet)
    fit_bjs(df; y=y, id=id, t=t, g=g, controls=controls, fe=fe,
                       weights=weights, cluster=cluster, horizons=nothing,
                       control_type=control_type)
end


"""
    fit_bjs_dynamic(df; y, id, t, g; horizons=true, kwargs...)

Dynamic ES alias for `fit_bjs` with `horizons=true` (all τ ≥ 0).
Returns a `SimpleModel` with coefficients `"τ::<τ>"`.
"""
function fit_bjs_dynamic(df::DataFrame; 
                         y::Symbol, 
                         id::Symbol, t::Symbol, g::Symbol,
                         controls::Vector{Symbol}=Symbol[], 
                         fe::Tuple{Vararg{Symbol}}=(id,t),
                         weights::Union{Nothing,Symbol}=nothing,
                         cluster::Union{Nothing,Symbol}=id,
                         ref::Int=-1,
                         horizons::Union{Nothing,Bool,Vector{Int}}=true,
                         pretrends::Union{Nothing,Bool,Vector{Int}}=nothing,
                         control_type::Symbol=:notyet)
    fit_bjs(df; y=y, id=id, t=t, g=g, controls=controls, fe=fe,
                       weights=weights, cluster=cluster, horizons=horizons,
                       pretrends=pretrends, control_type=control_type)
end








"""
    fit_bjs_boot(df; y, id, t, g, ...)

Streamlined BJS estimation for bootstrap iterations. Only computes point estimates,
skipping expensive standard error calculations for speed.

Returns coefficient vector in same order as main `fit_bjs` function.
"""
function _fit_bjs_boot(df::DataFrame; 
                      y::Symbol, 
                      id::Symbol, t::Symbol, g::Symbol,
                      controls::Vector{Symbol}=Symbol[],
                      fe::Tuple{Vararg{Symbol}}=(id, t),
                      weights::Union{Nothing,Symbol}=nothing,
                      horizons::Union{Nothing,Bool,Vector{Int}}=nothing,
                      pretrends::Union{Nothing,Bool,Vector{Int}}=nothing,
                      control_type::Symbol=:notyet)

    d = copy(df)
    d[ismissing.(d[!, g]), g] .= 0

    n = nrow(d)

    # Treatment indicator and event time
    treat = (.!ismissing.(d[!, g])) .& (d[!, g] .> 0) .& (d[!, t] .>= d[!, g])
    d[!, :__treat] = Int.(treat)
    make_eventtime!(d; t=t, g=g, new=:ttt, ref_p=-1_000_000)

    # Get event times for pretrend estimation
    event_times = sort(unique(d[d[!, :ttt] .!= -1_000_000, :ttt]))

    # First stage on donors only
    donors = donor_mask(d; t=t, g=g, control_type=control_type)

    # Build first stage formula
    if isempty(controls)
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ fe_terms_sum
    else
        control_terms = sum(term.(controls))
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ control_terms + fe_terms_sum
    end
    
    # Fit first stage model
    m1 = reg(d[donors, :], f1, Vcov.robust(); 
             weights = weights === nothing ? nothing : weights,save=:fe)
    
    # Y(0) prediction
    y0hat = fe_predict(m1, d)
    if any(ismissing, y0hat)
        y0hat = coalesce.(y0hat, 0.0)
    end
    y0hat = Float64.(y0hat)

    # Create adjusted outcome
    d[!, :__adj] = Float64.(d[!, y]) .- y0hat
    
    # Treatment effects (main estimation)
    main_β = Float64[]
    main_names = String[]
    treat_mask = d[!, :__treat] .== 1
    
    if horizons !== nothing || isnothing(pretrends)
        # Build treatment weights
        wtr_syms = build_wtr!(d; ttt=:ttt, treat=:__treat, weights=weights, 
                              horizons=horizons, prefix="__wtr")
        k = length(wtr_syms)
        
        if k > 0
            WTR = reduce(hcat, (Float64.(d[!, s]) for s in wtr_syms))
            
            if sum(treat_mask) > 0
                WTR_treated = WTR[treat_mask, :]
                adj_treated = Float64.(d[treat_mask, :__adj])
                
                # Point estimates only
                main_β = vec(sum(WTR_treated .* adj_treated, dims=1))
                
                # Build coefficient names
                if horizons === nothing
                    main_names = ["_ATT"]
                else
                    τkeep = parse.(Int, replace.(String.(wtr_syms), "__wtr" => ""))
                    main_names = "τ::" .* string.(τkeep)
                end
            end
        end
    end

    # Pretrend estimation
    pre_β = Float64[]
    pre_names = String[]

    if !isnothing(pretrends) && pretrends !== false
        pretrend_periods = pretrends === true ? filter(τ -> τ < 0, event_times) : filter(τ -> τ < 0, pretrends)

        if !isempty(pretrend_periods)
            d_untreated = d[.!treat_mask, :]
            if nrow(d_untreated) > 0
                # Build explicit indicator columns
                pre_syms = Symbol[]
                for τ in pretrend_periods
                    s = Symbol("__pre_", τ)
                    push!(pre_syms, s)
                    d_untreated[!, s] = Float64.(d_untreated[!, :ttt] .== τ)
                end

                # Build formula
                control_terms = isempty(controls) ? nothing : sum(term.(controls))
                fe_terms_sum = length(fe) == 0 ? nothing : reduce(+, [FixedEffectModels.fe(s) for s in fe])
                pre_terms_sum = sum(term.(pre_syms))

                rhs_list = Any[pre_terms_sum]
                if control_terms !== nothing; push!(rhs_list, control_terms); end
                if fe_terms_sum !== nothing; push!(rhs_list, fe_terms_sum); end
                rhs_pre = reduce(+, rhs_list)

                f_pre = Term(y) ~ rhs_pre
                
                # Fit pretrend model 
                pre_est = reg(d_untreated, f_pre, Vcov.robust(); weights=weights)

                # Extract coefficients
                cn = StatsAPI.coefnames(pre_est)
                coefs = StatsAPI.coef(pre_est)

                # Map symbols to indices
                idx_map = Dict{Symbol, Int}()
                for (i, nm) in pairs(cn)
                    try
                        idx_map[Symbol(nm)] = i
                    catch
                    end
                end

                pre_indices = [get(idx_map, Symbol("__pre_", τ), nothing) for τ in pretrend_periods]
                keep_mask = map(!isnothing, pre_indices)
                pre_indices = collect(skipmissing(pre_indices))

                if !isempty(pre_indices)
                    pre_β = coefs[pre_indices]
                    pre_names = ["τ::$(τ)" for (τ, keep) in zip(pretrend_periods, keep_mask) if keep]
                end
            end
        end
    end

    # Combine and order coefficients (same logic as main function)
    all_β = vcat(pre_β, main_β)
    all_names = vcat(pre_names, main_names)
    
    # Order by τ (same as main function)
    if length(all_names) > 1 && any(contains.(all_names, "τ::"))
        τ_values = Float64[]
        for name in all_names
            if contains(name, "τ::")
                τ_str = replace(name, "τ::" => "")
                push!(τ_values, parse(Float64, τ_str))
            elseif name == "_ATT"
                push!(τ_values, Inf)
            else
                push!(τ_values, 0.0)
            end
        end
        
        sort_order = sortperm(τ_values)
        all_β = all_β[sort_order]
        all_names = all_names[sort_order]
    end
    
    # Return just the coefficient vector (not full model)
    return all_β
end








"""
    fit_bjs_wild_bootstrap(df; y, id, t, g,
                       controls = Symbol[],
                       fe::Tuple = (id, t),
                       weights::Union{Nothing,Symbol} = nothing,
                       cluster::Union{Nothing,Symbol} = nothing,
                       horizons::Union{Nothing,Bool,Vector{Int}} = nothing,
                       pretrends::Union{Nothing,Bool,Vector{Int}} = nothing,
                       control_type::Symbol = :notyet,
                       n_bootstrap::Int = 500,
                       rng::Union{Nothing,AbstractRNG} = nothing)

Wild bootstrap for BJS imputation estimator to obtain joint covariance matrix 
across pre-treatment and post-treatment estimates.

This function addresses the fundamental limitation of the BJS method: since pretrends
and treatment effects come from separate regressions, we cannot analytically compute
their joint covariance. Wild bootstrap solves this by:

1. **Bootstrap the entire BJS procedure** including both stages
2. **Respect cluster structure** when applying bootstrap weights
3. **Extract joint distribution** of all coefficients
4. **Compute empirical covariance** matrix

## Wild Bootstrap Procedure

For each bootstrap iteration:
1. **First stage residuals**: Get residuals from donor regression y ~ controls + FE
2. **Apply cluster-level wild weights**: Multiply residuals by εᵇ ~ Rademacher(±1)
3. **Bootstrap outcomes**: y*ᵇ = XβÌ, + ε*ᵇ (where ε*ᵇ respects cluster structure)
4. **Re-estimate BJS**: Run full BJS procedure on bootstrap data
5. **Extract all coefficients**: Both treatment effects and pretrends

# Arguments
Same as `fit_bjs` plus:
- `n_bootstrap::Int`: Number of bootstrap iterations (default: 500)
- `rng::Union{Nothing,AbstractRNG}`: Random number generator for reproducibility

# Returns
`SimpleModel` with bootstrap covariance matrix, fully compatible with StatsAPI.
This means you can use all existing functions like `cumulative_effects()`, 
`joint_pretrend_test()`, etc. directly on the bootstrap result.

Note: If you need the raw bootstrap draws for further analysis, you can access
them by running the bootstrap manually or modify the function to also return them.

# Examples
```julia
# Basic wild bootstrap
boot_model = fit_bjs_wild_bootstrap(df; y=:ret, id=:fund, t=:month, g=:g,
                                horizons=true, pretrends=true,
                                cluster=:fund, n_bootstrap=500)

# Now use ANY existing function with bootstrap covariance!
cum_effects = cumulative_effects(boot_model; ref_period=-1)
pretrend_test = joint_pretrend_test(boot_model; leads=[-3,-2,-1])

# Standard StatsAPI functions work too
coef(boot_model)        # Point estimates
vcov(boot_model)        # Bootstrap covariance matrix
coefnames(boot_model)   # Coefficient names
```
"""
function fit_bjs_wild_bootstrap(df::DataFrame; 
                            y::Symbol, 
                            id::Symbol, t::Symbol, g::Symbol,
                            controls::Vector{Symbol}=Symbol[],
                            fe::Tuple{Vararg{Symbol}}=(id, t),
                            weights::Union{Nothing,Symbol}=nothing,
                            cluster::Union{Nothing,Symbol}=id,
                            horizons::Union{Nothing,Bool,Vector{Int}}=nothing,
                            pretrends::Union{Nothing,Bool,Vector{Int}}=nothing,
                            control_type::Symbol=:notyet,
                            n_bootstrap::Int=500,
                            seed::Union{Nothing,Int}=nothing)

    rng = Random.default_rng()
    if isnothing(seed)
        nothing
    else
        Random.seed!(rng,seed)
    end
    
    # Get original estimates
    original_model = fit_bjs(df; y=y, id=id, t=t, g=g,
                                        controls=controls, fe=fe,
                                        weights=weights, cluster=cluster,
                                        horizons=horizons, pretrends=pretrends,
                                        control_type=control_type)
    
    original_coefs = StatsAPI.coef(original_model)
    original_names = StatsAPI.coefnames(original_model)
    n_coefs = length(original_coefs)
    
    # Prepare data (same preprocessing as original)
    d = copy(df)
    d[ismissing.(d[!, g]), g] .= 0
    
    # Missing data handling
    key_vars = [y, id, t, g]
    if !isnothing(weights); push!(key_vars, weights); end
    if !isnothing(cluster); push!(key_vars, cluster); end
    append!(key_vars, controls)
    append!(key_vars, collect(fe))
    key_vars = unique(key_vars)
    
    missing_mask = falses(nrow(d))
    for var in key_vars
        missing_mask .|= ismissing.(d[!, var])
    end
    if any(missing_mask)
        d = d[.!missing_mask, :]
    end
    
    # Create treatment variables
    treat = (.!ismissing.(d[!, g])) .& (d[!, g] .> 0) .& (d[!, t] .>= d[!, g])
    d[!, :__treat] = Int.(treat)
    make_eventtime!(d; t=t, g=g, new=:ttt, ref_p=-1_000_000)
    
    # Get donors and fit first stage (for bootstrap residuals)
    donors = donor_mask(d; t=t, g=g, control_type=control_type)
    
    # Build first stage formula
    if isempty(controls)
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ fe_terms_sum
    else
        control_terms = sum(term.(controls))
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ control_terms + fe_terms_sum
    end
    
    w_vec = isnothing(weights) ? nothing : Float64.(d[!, weights])
    
    # Fit original first stage to get residuals
    m1_original = reg(d[donors, :], f1, Vcov.robust(); 
                     weights = w_vec === nothing ? nothing : w_vec[donors],
                     save = :fe)
    
    # Get fitted values and residuals using fe_predict
    y_fitted_donors_full = fe_predict(m1_original, d)
    y_fitted_donors = y_fitted_donors_full[donors]
    residuals_donors = Float64.(d[donors, y]) .- y_fitted_donors
    
    # Set up cluster structure for wild bootstrap
    cl_var = isnothing(cluster) ? id : cluster
    cluster_ids = unique(d[donors, cl_var])
    n_clusters = length(cluster_ids)
    
    # Create mapping from observations to clusters (donor sample only)
    donor_cluster_map = Dict{eltype(cluster_ids), Vector{Int}}()
    for (i, obs_idx) in enumerate(findall(donors))
        cl_id = d[obs_idx, cl_var]
        if !haskey(donor_cluster_map, cl_id)
            donor_cluster_map[cl_id] = Int[]
        end
        push!(donor_cluster_map[cl_id], i)
    end
    
    # Storage for bootstrap results
    bootstrap_coefs = Matrix{Float64}(undef, n_bootstrap, n_coefs)
    
    for b in 1:n_bootstrap
        try
            # Generate cluster-level wild bootstrap weights (Rademacher)
            cluster_weights = Dict{eltype(cluster_ids), Float64}()
            for cl_id in cluster_ids
                cluster_weights[cl_id] = rand(rng, [-1.0, 1.0])
            end
            
            # Apply wild weights to residuals (cluster-level)
            residuals_bootstrap = copy(residuals_donors)
            for (cl_id, obs_indices) in donor_cluster_map
                weight = cluster_weights[cl_id]
                residuals_bootstrap[obs_indices] .*= weight
            end
            
            # Create bootstrap outcomes on donor sample
            y_bootstrap_donors = y_fitted_donors .+ residuals_bootstrap
            
            # Create bootstrap dataset
            d_bootstrap = copy(d)
            d_bootstrap[donors, y] = y_bootstrap_donors
            
            # Run full BJS estimation on bootstrap data
            boot_coefs = _fit_bjs_boot(d_bootstrap; 
                                                y=y, id=id, t=t, g=g,
                                                controls=controls, fe=fe,
                                                weights=weights,
                                                horizons=horizons, pretrends=pretrends,
                                                control_type=control_type)
            
            # Store bootstrap coefficients
            bootstrap_coefs[b, :] = boot_coefs
            
        catch e
            # Fill with NaN for failed iterations
            bootstrap_coefs[b, :] .= NaN
        end
    end
    
    # Remove failed iterations
    valid_iterations = .!any(isnan.(bootstrap_coefs), dims=2)[:, 1]
    bootstrap_coefs_clean = bootstrap_coefs[valid_iterations, :]
    n_valid = sum(valid_iterations)
    
    if n_valid < 50
        error("Too few successful bootstrap iterations ($n_valid). Check your model specification.")
    end
    
    # Compute bootstrap covariance matrix
    vcov_bootstrap = cov(bootstrap_coefs_clean)
    
    # Extract model information from original BJSModel
    n_obs = StatsAPI.nobs(original_model)
    dof_resid = StatsAPI.dof_residual(original_model)
    
    # Create BJSModel with bootstrap covariance
    bootstrap_model = BJSModel(
        original_coefs, vcov_bootstrap, original_names, n_obs, dof_resid;
        y_name = original_model.y_name,
        estimator_type = original_model.estimator_type,
        treatment_periods = original_model.treatment_periods,
        first_stage_r2 = original_model.first_stage_r2,
        n_treated = original_model.n_treated,
        n_donors = original_model.n_donors,
        control_type = original_model.control_type,
        boot=true,n_boot=n_bootstrap,cluster=cluster
    )
    
    return bootstrap_model
end




function fit_bjs_wild_bootstrap_mthreaded(df::DataFrame; 
                            y::Symbol, 
                            id::Symbol, t::Symbol, g::Symbol,
                            controls::Vector{Symbol}=Symbol[],
                            fe::Tuple{Vararg{Symbol}}=(id, t),
                            weights::Union{Nothing,Symbol}=nothing,
                            cluster::Union{Nothing,Symbol}=nothing,
                            horizons::Union{Nothing,Bool,Vector{Int}}=nothing,
                            pretrends::Union{Nothing,Bool,Vector{Int}}=nothing,
                            control_type::Symbol=:notyet,
                            n_bootstrap::Int=500,
                            seed::Union{Nothing,Int}=nothing)

    # Get original estimates
    original_model = fit_bjs(df; y=y, id=id, t=t, g=g,
                                        controls=controls, fe=fe,
                                        weights=weights, cluster=cluster,
                                        horizons=horizons, pretrends=pretrends,
                                        control_type=control_type)
    
    original_coefs = StatsAPI.coef(original_model)
    original_names = StatsAPI.coefnames(original_model)
    n_coefs = length(original_coefs)
    
    # Prepare data (same preprocessing as original)
    d = copy(df)
    d[ismissing.(d[!, g]), g] .= 0
    
    # Missing data handling
    key_vars = [y, id, t, g]
    if !isnothing(weights); push!(key_vars, weights); end
    if !isnothing(cluster); push!(key_vars, cluster); end
    append!(key_vars, controls)
    append!(key_vars, collect(fe))
    key_vars = unique(key_vars)
    
    missing_mask = falses(nrow(d))
    for var in key_vars
        missing_mask .|= ismissing.(d[!, var])
    end
    if any(missing_mask)
        d = d[.!missing_mask, :]
    end
    
    # Create treatment variables
    treat = (.!ismissing.(d[!, g])) .& (d[!, g] .> 0) .& (d[!, t] .>= d[!, g])
    d[!, :__treat] = Int.(treat)
    make_eventtime!(d; t=t, g=g, new=:ttt, ref_p=-1_000_000)
    
    # Get donors and fit first stage (for bootstrap residuals)
    donors = donor_mask(d; t=t, g=g, control_type=control_type)
    
    # Build first stage formula
    if isempty(controls)
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ fe_terms_sum
    else
        control_terms = sum(term.(controls))
        fe_terms_sum = reduce(+, [FixedEffectModels.fe(s) for s in fe])
        f1 = Term(y) ~ control_terms + fe_terms_sum
    end
    
    w_vec = isnothing(weights) ? nothing : Float64.(d[!, weights])
    
    # Fit original first stage to get residuals
    m1_original = reg(d[donors, :], f1, Vcov.robust(); 
                     weights = w_vec === nothing ? nothing : w_vec[donors],
                     save = :fe)
    
    # Get fitted values and residuals using fe_predict
    y_fitted_donors_full = fe_predict(m1_original, d)
    y_fitted_donors = y_fitted_donors_full[donors]
    residuals_donors = Float64.(d[donors, y]) .- y_fitted_donors
    
    # Set up cluster structure for wild bootstrap
    cl_var = isnothing(cluster) ? id : cluster
    cluster_ids = unique(d[donors, cl_var])
    n_clusters = length(cluster_ids)
    
    # Create mapping from observations to clusters (donor sample only)
    donor_cluster_map = Dict{eltype(cluster_ids), Vector{Int}}()
    for (i, obs_idx) in enumerate(findall(donors))
        cl_id = d[obs_idx, cl_var]
        if !haskey(donor_cluster_map, cl_id)
            donor_cluster_map[cl_id] = Int[]
        end
        push!(donor_cluster_map[cl_id], i)
    end
    
    # Storage for bootstrap results
    bootstrap_coefs = Matrix{Float64}(undef, n_bootstrap, n_coefs)
    
    # Create one RNG per thread (typically 4-16 RNGs instead of 500)
    base_seed = isnothing(seed) ? 12345 : seed
    thread_rngs = [Random.MersenneTwister() for _ in 1:Threads.nthreads()]
    
    Threads.@threads for b in 1:n_bootstrap
        thread_id = Threads.threadid()
        local_rng = thread_rngs[thread_id]
        
        # Seed deterministically based on iteration (small overhead, but much less than RNG creation)
        Random.seed!(local_rng, xor(base_seed, 0x9E3779B97F4A7C15 * UInt(b)))
        
        try
            # Generate cluster-level wild bootstrap weights (Rademacher)
            cluster_weights = Dict{eltype(cluster_ids), Float64}()
            for cl_id in cluster_ids
                cluster_weights[cl_id] = rand(local_rng, [-1.0, 1.0])
            end
            # Apply wild weights to residuals (cluster-level)
            residuals_bootstrap = copy(residuals_donors)
            for (cl_id, obs_indices) in donor_cluster_map
                weight = cluster_weights[cl_id]
                residuals_bootstrap[obs_indices] .*= weight
            end
            
            # Create bootstrap outcomes on donor sample
            y_bootstrap_donors = y_fitted_donors .+ residuals_bootstrap
            
            # Create bootstrap dataset
            d_bootstrap = copy(d)
            d_bootstrap[donors, y] = y_bootstrap_donors
            
            # Run streamlined BJS estimation on bootstrap data (MUCH FASTER)
            boot_coefs = _fit_bjs_boot(d_bootstrap; 
                                     y=y, id=id, t=t, g=g,
                                     controls=controls, fe=fe,
                                     weights=weights,
                                     horizons=horizons, pretrends=pretrends,
                                     control_type=control_type)
            
            # Store bootstrap coefficients
            bootstrap_coefs[b, :] = boot_coefs
            
        catch e
            # Fill with NaN for failed iterations
            bootstrap_coefs[b, :] .= NaN
        end
    end
    
    # Remove failed iterations
    valid_iterations = .!any(isnan.(bootstrap_coefs), dims=2)[:, 1]
    bootstrap_coefs_clean = bootstrap_coefs[valid_iterations, :]
    n_valid = sum(valid_iterations)
    
    if n_valid < 50
        error("Too few successful bootstrap iterations ($n_valid). Check your model specification.")
    end
    
    # Compute bootstrap covariance matrix
    vcov_bootstrap = cov(bootstrap_coefs_clean)
    
    # Extract model information from original BJSModel
    n_obs = StatsAPI.nobs(original_model)
    dof_resid = StatsAPI.dof_residual(original_model)
    
    # Create BJSModel with bootstrap covariance
    bootstrap_model = BJSModel(
        original_coefs, vcov_bootstrap, original_names, n_obs, dof_resid;
        y_name = original_model.y_name,
        estimator_type = original_model.estimator_type,
        treatment_periods = original_model.treatment_periods,
        first_stage_r2 = original_model.first_stage_r2,
        n_treated = original_model.n_treated,
        n_donors = original_model.n_donors,
        control_type = original_model.control_type,
        boot=true,n_boot=n_bootstrap,cluster=cluster
    )
    
    return bootstrap_model
end
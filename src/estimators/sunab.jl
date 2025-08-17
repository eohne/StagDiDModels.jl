"""
    fit_sunab(df; y, id, t, g;
              controls = Symbol[],
              cluster = nothing,
              weights::Union{Nothing,Symbol} = nothing,
              ref_p::Int = -1,
              ref_c::Union{Symbol,Vector{Int}} = :never,
              agg::Symbol = :dynamic)

Sun—Abraham event study that **matches `fixest::sunab`** exactly.

# What it does
- Build saturated `(cohort g) × (event time τ=t−g)` dummies **excluding**
  (i) the **pre-period base** `τ == ref_p` and
  (ii) the **reference cohort** `ref_c` (default `:never`).
- Regress `y ~ cohort×τ dummies + controls + FE(id) + FE(t)` with clustering/weights.
- **Aggregate** cohort×τ coefs to:
  - **Dynamic per-τ** effects (default): for each τ, average across cohorts using
    **design-matrix weights** (observation counts per (g,τ) cell).
  - **Static ATT** (`agg=:att`): average all **post** τ (τ ≥ 0, excluding ref_p)
    using the same **design weights**.

## Conventions (IMPORTANT)
- **Never-treated must be `g == 0` or `missing`.**
- `t` is an integer period index.

# Arguments
- `df::DataFrame`, `y::Symbol`, `id::Symbol`, `t::Symbol`, `g::Symbol`.

# Keyword Arguments
- `controls::Vector{Symbol}`: extra RHS covariates.
- `cluster`: clustering key or `nothing`.
- `weights`: row weights used in regression **and** aggregation weights.
- `ref_p`: pre-period base τ (default `-1`), excluded from estimation.
- `ref_c`: reference cohort (`:never` = never-treated; or `Vector{Int}` for specific cohorts).
- `agg`: `:dynamic` (per-τ effects), `:att` (static ATT), or `:both`.

# Returns
- If `agg=:dynamic`: `SunabModel` with coefficients `"τ::<τ>"`.
- If `agg=:att`:     `SunabModel` with single `"ATT"` coefficient.
- If `agg=:both`:    `NamedTuple` `(dynamic=SunabModel, att=SunabModel)`.

# Examples
```julia
# Dynamic ES with default references (ref_p = -1, ref_c = :never)
m_dyn = fit_sunab(df; y=:ret, id=:fund, t=:month, g=:g,
                  controls=[:size,:bm], cluster=:fund, weights=:w)

# Access the Sun-Abraham aggregated results:
coef(m_dyn)                    # Event-time coefficients τ::-2, τ::0, τ::1, etc.
sqrt.(diag(vcov(m_dyn)))       # Standard errors
coefnames(m_dyn)               # Coefficient names
m_dyn                          # Pretty-printed model output

# Single ATT aggregation over τ ≥ 0
m_att = fit_sunab(df; y=:ret, id=:fund, t=:month, g=:g,
                  controls=[:size,:bm], cluster=:fund, agg=:att)
coef(m_att)[1]                 # ATT estimate

# Both outputs at once
mods = fit_sunab(df; y=:ret, id=:fund, t=:month, g=:g, agg=:both)
coef(mods.dynamic)             # Event-time effects
coef(mods.att)                 # ATT
```
"""
function fit_sunab(df::DataFrame; y::Symbol, id::Symbol, t::Symbol, g::Symbol,
                   controls::Vector{Symbol}=Symbol[],
                   cluster=nothing,
                   weights::Union{Nothing,Symbol}=nothing,
                   ref_p::Int=-1,
                   ref_c::Union{Symbol,Vector{Int}}=:never,
                   agg::Symbol=:dynamic)

    d = copy(df)
    
    # Handle missing g values (mark as never-treated)
    d[ismissing.(d[!, g]), g] .= 0
    
    n = nrow(d)
    cohort = d[!, g]
    period = d[!, t]
    
    # Detect if we need to construct relative periods 
    period_unik = unique(period)
    cohort_unik = unique(cohort)
    
    # Case detection: if period contains negative values and 0, assume already relative
    is_relative = 0 ∈ period_unik && minimum(period_unik) < 0 && maximum(period_unik) > 0
    
    if is_relative
        # Case 1: period is already relative time
        rel_period = period
    else
        # Case 2: construct relative period from cohort and period
        # Never-treated cohorts (not in period) become references
        refs = setdiff(cohort_unik, period_unik)
        
        if length(refs) == length(cohort_unik)
            error("Problem creating relative periods. Expected cohort to be treatment period, but no cohort values found in period.")
        end
        
        # Keep only observations with valid cohorts (present in period data)
        keep_mask = .!(cohort .∈ Ref(refs))
        
        rel_period = fill(-1, n)  # Default for never-treated
        rel_period[keep_mask] = period[keep_mask] .- cohort[keep_mask]
    end
    
    d[!, :rel_period] = rel_period
    
    # Identify never-treated and always-treated
    never_treated_mask = (cohort .== 0) .| ismissing.(cohort)
    
    # Find always treated (no negative relative periods)
    always_treated_mask = falses(n)
    for gval in unique(cohort[cohort .> 0])
        g_mask = cohort .== gval
        if all(rel_period[g_mask] .>= 0)
            always_treated_mask[g_mask] .= true
        end
    end
    
    # Set up reference cohorts
    ref_cohorts = if ref_c === :never
        unique(cohort[never_treated_mask])
    else
        Vector{Int}(ref_c)
    end
    
    # Create interaction dummies: exclude reference period and reference cohorts
    include_mask = (rel_period .!= ref_p) .& .!(cohort .∈ Ref(ref_cohorts))
    
    # Create saturated cohort x relative_period interaction
    d[!, :sunab_key] = fill("REF", n)
    for i in 1:n
        if include_mask[i]
            d[i, :sunab_key] = "$(cohort[i])_$(rel_period[i])"
        end
    end
    
    # Make it categorical with REF as reference level
    d[!, :sunab_key] = CategoricalArray(d[!, :sunab_key])
    
    # Set "REF" as reference level (will be dropped from regression)
    levels!(d[!, :sunab_key], ["REF"; sort(setdiff(levels(d[!, :sunab_key]), ["REF"]))])
    
    # Build regression formula
    if isempty(controls)
        rhs = term(:sunab_key) + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    else
        control_terms = sum(term.(controls))
        rhs = term(:sunab_key) + control_terms + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    end
    
    f = Term(y) ~ rhs
    
    # Handle always-treated: set their outcomes to missing
    if any(always_treated_mask)
        d[always_treated_mask, y] .= missing
    end
    
    # Fit saturated regression
    vc = build_cluster_vcov(cluster)
    reg_weights = isnothing(weights) ? nothing : d[!, weights]
    
    sat = reg(d, f, vc; weights=weights)
    
    # Extract coefficients for cohort x period interactions
    coef_names = StatsAPI.coefnames(sat)
    β = StatsAPI.coef(sat)
    
    # Find sunab interaction coefficients (note the space after colon!)
    sunab_indices = Int[]
    gt_pairs = Tuple{Int,Int}[]
    
    for (i, name) in enumerate(coef_names)
        if startswith(name, "sunab_key: ") && name != "sunab_key: REF"
            # Extract "g_τ" from "sunab_key: g_τ"  
            interaction_part = replace(name, "sunab_key: " => "")
            parts = split(interaction_part, "_")
            if length(parts) == 2
                try
                    g_val = parse(Int, parts[1])
                    τ_val = parse(Int, parts[2])
                    push!(sunab_indices, i)
                    push!(gt_pairs, (g_val, τ_val))
                catch
                    # Skip if parsing fails
                end
            end
        end
    end
    
    if isempty(gt_pairs)
        error("No valid Sun-Abraham coefficients found. Found coefficient names: $(coef_names)")
    end
    
    # Compute design matrix weights (observation counts per (g,τ) cell)
    weights_vec = isnothing(weights) ? nothing : d[!, weights]
    cell_weights = Float64[]
    
    for (g_val, τ_val) in gt_pairs
        cell_mask = (cohort .== g_val) .& (rel_period .== τ_val)
        if isnothing(weights_vec)
            weight = Float64(sum(cell_mask))
        else
            weight = sum(weights_vec[cell_mask])
        end
        push!(cell_weights, weight)
    end
    
    # Get unique τ values and sort them
    τ_values = sort(unique(last.(gt_pairs)))
    n_τ = length(τ_values)
    n_coef_total = length(coef_names)
    
    # Build dynamic aggregation matrix:
    A_dynamic_full = zeros(n_τ, n_coef_total)
    dynamic_names = String[]
    
    for (τ_idx, τ) in enumerate(τ_values)
        # Find all (g,τ) pairs with this τ
        matching_pairs = [(i, gt_pairs[i]) for i in 1:length(gt_pairs) if gt_pairs[i][2] == τ]
        
        if !isempty(matching_pairs)
            # Get weights for this τ
            τ_weights = [cell_weights[i] for (i, _) in matching_pairs]
            total_weight = sum(τ_weights)
            
            # Normalize weights 
            if total_weight > 0
                τ_weights ./= total_weight
            else
                τ_weights = fill(1.0 / length(τ_weights), length(τ_weights))
            end
            
            # Set aggregation weights
            for (k, (pair_idx, _)) in enumerate(matching_pairs)
                coef_idx = sunab_indices[pair_idx]  # Index in full coefficient vector
                A_dynamic_full[τ_idx, coef_idx] = τ_weights[k]
            end
        end
        
        push!(dynamic_names, "τ::$(τ)")
    end
    
    # Create dynamic model and extract aggregated results

    β_dynamic = A_dynamic_full * StatsAPI.coef(sat)
    Σ_dynamic = A_dynamic_full * StatsAPI.vcov(sat) * transpose(A_dynamic_full) 
    
    # Calculate model statistics
    n_obs = StatsAPI.nobs(sat)
    dof_resid = StatsAPI.dof_residual(sat)
    r2_sat = StatsAPI.r2(sat)
    adjr2_sat = StatsAPI.adjr2(sat)
    
    # Calculate treatment counts
    n_treated = sum(.!never_treated_mask .& .!always_treated_mask .& (rel_period .>= 0))
    
    # Calculate "donor" counts: never-treated + pre-treatment observations
    n_never_treated = sum(never_treated_mask)
    n_pre_treatment = sum(.!never_treated_mask .& .!always_treated_mask .& (rel_period .< 0))
    n_donors = n_never_treated + n_pre_treatment
    
    # Determine estimator type
    est_type = length(β_dynamic) == 1 ? :static : :dynamic
    
    # Build final model
    final_model = SunabModel(
        β_dynamic, Σ_dynamic, dynamic_names, n_obs, dof_resid;
        y_name = y,
        estimator_type = est_type,
        treatment_periods = Int.(τ_values),
        r2 = r2_sat,
        adjr2 = adjr2_sat,
        n_post = n_treated,
        n_pre = n_donors,
        cluster = isnothing(cluster) ? :none : cluster
    )
    
    if agg === :dynamic
        return final_model
    end
    
    # ATT aggregation: average over post-treatment periods (τ ≥ 0, excluding ref_p)  
    post_τ_mask = (τ_values .>= 0) .& (τ_values .!= ref_p)
    post_τ_indices = findall(post_τ_mask)
    
    if isempty(post_τ_indices)
        @warn "No post-treatment periods found for ATT calculation"
        att_β = [0.0]
        att_Σ = reshape([1.0], 1, 1)
        att_names = ["ATT"]
    else
        # Weight each post-τ by its total design matrix weight
        post_weights = Float64[]
        for i in post_τ_indices
            τ = τ_values[i]
            # Sum weights for all (g,τ) pairs with this τ
            τ_total_weight = sum(cell_weights[j] for j in 1:length(gt_pairs) if gt_pairs[j][2] == τ)
            push!(post_weights, τ_total_weight)
        end
        
        total_post_weight = sum(post_weights)
        if total_post_weight > 0
            post_weights ./= total_post_weight
        else
            post_weights = fill(1.0 / length(post_weights), length(post_weights))
        end
        
        # Create ATT aggregation from dynamic results
        att_β = [sum(post_weights .* β_dynamic[post_τ_indices])]
        
        # ATT variance: w' Σ w where w is the weight vector
        w_vec = zeros(length(β_dynamic))
        w_vec[post_τ_indices] = post_weights
        att_Σ = reshape([w_vec' * Σ_dynamic * w_vec], 1, 1)
        att_names = ["ATT"]
    end
    
    att_model = SunabModel(
        att_β, att_Σ, att_names, n_obs, dof_resid;
        y_name = y,
        estimator_type = :static,
        treatment_periods = Int.(τ_values[post_τ_mask]),
        r2 = r2_sat,
        adjr2 = adjr2_sat,
        n_post = n_treated,
        n_pre = n_donors,
        cluster = isnothing(cluster) ? :none : cluster
    )
    
    return agg === :att ? att_model : (dynamic=final_model, att=att_model)
end
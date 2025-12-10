"""
    fit_gardner_static(df::DataFrame; y::Symbol, id::Symbol, t::Symbol, g::Symbol,
                     controls::Vector{Symbol}=Symbol[], cluster::Symbol,
                     weights::Union{Nothing,Symbol}=nothing, control_type::Symbol=:notyet,
                     autosample::Bool=true)

Gardner (2021) two-stage difference-in-differences estimator for static treatment effects.

Estimates Y(0) using donor observations, then regresses residualized outcome on treatment indicator.

# Data Conventions

**Never-treated units must be coded as `g == 0` or `g == missing`.**

# Arguments

- `df::DataFrame`: Panel dataset
- `y::Symbol`: Outcome variable
- `id::Symbol`: Unit identifier  
- `t::Symbol`: Time period (integer)
- `g::Symbol`: Treatment cohort (period when first treated)
- `controls::Vector{Symbol}`: Control variables for first stage
- `cluster::Symbol`: Clustering variable for standard errors - defaults to `id` if not provided
- `weights::Union{Nothing,Symbol}`: Observation weights
- `control_type::Symbol`: `:notyet` (not-yet + never treated) or `:never` (never treated only)
- `autosample::Bool`: If true (default), drop treated observations where FE cannot be imputed.
                      If false, error when FE cannot be imputed.

# Returns

`GardnerModel`

# Example

```julia
model = fit_gardner_static(df; y=:outcome, id=:unit, t=:year, g=:treatment_year, cluster=:state)
coef(model)[1]  # ATT estimate
```
"""
function fit_gardner_static(df::DataFrame; 
                          y::Symbol, 
                          id::Symbol, 
                          t::Symbol,
                          g::Symbol,
                          controls::Vector{Symbol}=Symbol[], 
                          cluster::Symbol=id,
                          weights::Union{Nothing,Symbol}=nothing,
                          control_type::Symbol=:notyet,
                          autosample::Bool=true)

    d = copy(df)
    
    # Handle missing g values (never-treated)
    d[ismissing.(d[!, g]), g] .= 0
    
    # Compute treatment indicator from timing variables (standard across all estimators)
    treat = (.!ismissing.(d[!, g])) .& (d[!, g] .> 0) .& (d[!, t] .>= d[!, g])
    d[!, :_ATT] = Int.(treat)
    
    # Get donor mask
    untreated_mask = donor_mask(d; t=t, g=g, control_type=control_type)
    n_donors = sum(untreated_mask)
    
    if n_donors == 0
        error("No untreated observations found for first stage estimation")
    end
    
    # Build first stage formula
    if isempty(controls)
        f1 = Term(y) ~ FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    else
        control_terms = sum(term.(controls))
        f1 = Term(y) ~ control_terms + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    end
    
    # Estimate first stage on untreated observations only
    untreated_data = d[untreated_mask, :]
    
    m1 = reg(untreated_data, f1, Vcov.robust(); 
             weights = weights,
             save = :fe)
    
    # Extract first stage R²
    first_stage_r2 = try
        StatsAPI.r2(m1)
    catch
        0.0
    end
    
    # === AUTOSAMPLE: Check which treated observations can be imputed ===
    treated_mask = BitVector(d[!, :_ATT] .== 1)
    keep_mask, n_dropped = apply_autosample(d, m1, treated_mask; 
                                             autosample=autosample, verbose=true)
    
    if n_dropped > 0
        # Filter the dataset
        d = d[keep_mask, :]
        treated_mask = BitVector(d[!, :_ATT] .== 1)
        untreated_mask = donor_mask(d; t=t, g=g, control_type=control_type)
    end
    
    # Update counts after autosample
    n_treated = sum(d[!, :_ATT])
    n_donors = sum(untreated_mask)
    
    # Update weights vector if needed
    w_vec = isnothing(weights) ? nothing : Float64.(d[!, weights])
    
    # Predict Y(0) on filtered sample
    y0_pred = fe_predict(m1, d)
    
    # Handle any missing predictions (set to 0) - should be minimal after autosample
    if any(ismissing, y0_pred)
        y0_pred = coalesce.(y0_pred, 0.0)
    end
    y0_pred = Float64.(y0_pred)
    
    # Create residualized outcome: ytilde = y - y0_pred
    d[!, :ytilde] = Float64.(d[!, y]) .- y0_pred
    
    # Stage 2: Regress residualized outcome on computed treatment indicator
    f2 = Term(:ytilde) ~ term(0) + term(:_ATT)
    
    m2 = reg(d, f2, Vcov.robust(); weights = weights)
    
    # --- Standard Error Correction ---
    
    # Extract design matrices 
    x1 = sparse_rhs_fe(d; controls=controls, fe=(id, t))
    x2 = Float64.(d[!, :_ATT])  # Use computed treatment indicator
    
    # Apply sqrt weights
    w = isnothing(w_vec) ? ones(Float64, nrow(d)) : sqrt.(w_vec)

    # Weight the matrices
    x1_weighted = spdiagm(0 => w) * x1  # n x p
    x2_weighted = x2 .* w              # n x 1
    
    # Create x10 from WEIGHTED x1 
    x10_weighted = copy(x1_weighted)  # Copy the weighted matrix
    treated_indices = findall(treated_mask)
    for i in treated_indices
        x10_weighted[i, :] .= 0.0  # Zero out treated rows
    end
    
    # Residuals
    second_u = residuals(m2, d) .* w # second stage residuals
    first_u = copy(d[!, :ytilde]) .* w  # first stage residuals 
    first_u[treated_mask] .= 0.0  # zero on treated 
    
    # Get x2tx2_inv 
    x2tx2_manual = sum(x2_weighted .^ 2)
    inv_x2tx2_manual = 1.0 / x2tx2_manual
    
    x2tx2_inv = inv_x2tx2_manual
    
    # Influence function components:
    IF_ss = x2tx2_inv * (x2_weighted .* second_u)  # n x 1
    
    # Compute cross-products
    x10tx10 = Matrix(x10_weighted' * x10_weighted)  # p x p
    x1tx2 = x1_weighted' * x2_weighted    # p x 1
    
    # Solve: Matrix::solve(Matrix::crossprod(x10), Matrix::crossprod(x1, x2))
    solve_term = try
        x10tx10 \ x1tx2  # p x 1
    catch
        pinv(x10tx10) * x1tx2
    end
    
    # Complete first stage IF: x2tx2_inv * solve_term' * (x10 * first_u)
    x10_weighted_resid = Matrix(x10_weighted) .* ( first_u)  # n x p element-wise
    IF_fs_temp = x10_weighted_resid * solve_term  # n x 1
    IF_fs = x2tx2_inv * IF_fs_temp  # n x 1
    
    # Total influence function: IF = IF_fs - IF_ss
    IF_total = IF_fs .- IF_ss  # n x 1
    
    # Cluster aggregation
    cl = d[!, cluster]
    cluster_sums = Dict{eltype(cl), Float64}()
    
    for (i, cluster_id) in enumerate(cl)
        if haskey(cluster_sums, cluster_id)
            cluster_sums[cluster_id] += IF_total[i]
        else
            cluster_sums[cluster_id] = IF_total[i]
        end
    end
    
    # Variance: sum of squared cluster contributions
    cluster_var = sum(v^2 for v in values(cluster_sums))
    
    # Extract coefficients, names, and create final model
    β = StatsAPI.coef(m2)
    names = StatsAPI.coefnames(m2)
    Σ = reshape([cluster_var], 1, 1)
    nobs_val = StatsAPI.nobs(m2)
    dof_resid = StatsAPI.dof_residual(m2)
    second_stage_adjr2 = StatsAPI.adjr2(m2,:McFadden)

    # Get weighted R2:
    residuals_vals = residuals(m2,d)  # second stage residuals
    y_vals = d[!, :ytilde]  # the outcome in second stage
    if !isnothing(w_vec)
        w_sum = sum(w_vec)
        y_weighted_mean = sum(w_vec .* y_vals) / w_sum
        TSS_A = sum(w_vec .* (y_vals .- y_weighted_mean).^2)
        RSS_A = sum(w_vec .* residuals_vals.^2)
    else
        y_mean = mean(y_vals)
        TSS_A = sum((y_vals .- y_mean).^2)
        RSS_A = sum(residuals_vals.^2)
    end
    r2 = 1 - (RSS_A/TSS_A) * (nobs_val / (nobs_val - 1))

    return GardnerModel(
        β, Σ, names, nobs_val, dof_resid;
        y_name = y,
        estimator_type = :static,
        treatment_periods = Int[],  # empty for static
        first_stage_r2 = first_stage_r2,
        second_stage_r2 = r2,
        second_stage_adjr2 = r2,
        n_treated = n_treated,
        n_donors = n_donors,
        control_type = control_type,
        cluster = isnothing(cluster) ? :none : cluster
    )
end


"""
    fit_gardner_dynamic(df::DataFrame; y::Symbol, id::Symbol, t::Symbol, g::Symbol,
                      controls::Vector{Symbol}=Symbol[], cluster::Symbol,
                      ref_p::Int=-1, weights::Union{Nothing,Symbol}=nothing,
                      control_type::Symbol=:notyet, autosample::Bool=true)

Gardner (2021) two-stage difference-in-differences estimator for event study analysis.

Estimates Y(0) using donor observations, then regresses residualized outcome on event-time indicators.
Event time τ = t - g, with coefficients for each τ ≠ ref_p.

# Data Conventions

**Never-treated units must be coded as `g == 0` or `g == missing`.**

# Arguments

- `df::DataFrame`: Panel dataset
- `y::Symbol`: Outcome variable
- `id::Symbol`: Unit identifier
- `t::Symbol`: Time period (integer)
- `g::Symbol`: Treatment cohort (period when first treated)
- `controls::Vector{Symbol}`: Control variables for first stage
- `cluster::Symbol`: Clustering variable for standard errors
- `ref_p::Int`: Reference period excluded from estimation (default -1)
- `weights::Union{Nothing,Symbol}`: Observation weights  
- `control_type::Symbol`: `:notyet` (not-yet + never treated) or `:never` (never treated only)
- `autosample::Bool`: If true (default), drop treated observations where FE cannot be imputed.

# Returns

`GardnerModel` with event-study coefficients and cluster-robust standard errors.

# Example

```julia
model = fit_gardner_dynamic(df; y=:outcome, id=:unit, t=:year, g=:treatment_year, 
                          cluster=:state, ref_p=-1)
coefnames(model)  # Event time coefficients: "τ::-2", "τ::0", etc.
```
"""
function fit_gardner_dynamic(df::DataFrame; y::Symbol, id::Symbol, t::Symbol, g::Symbol,
                           controls::Vector{Symbol}=Symbol[], cluster::Symbol=id,
                           ref_p::Int=-1,
                           weights::Union{Nothing,Symbol}=nothing,
                           control_type::Symbol=:notyet,
                           autosample::Bool=true)

    d = copy(df)
    
    # Handle missing g values (never-treated)
    d[ismissing.(d[!, g]), g] .= 0
    
    # Compute treatment indicator from timing variables
    treat = (.!ismissing.(d[!, g])) .& (d[!, g] .> 0) .& (d[!, t] .>= d[!, g])
    d[!, :_ATT] = Int.(treat)
    
    # Create event time 
    make_eventtime!(d; t=t, g=g, new=:ttt, ref_p=ref_p)
    
    # Get donor mask
    untreated_mask = donor_mask(d; t=t, g=g, control_type=control_type)
    n_donors = sum(untreated_mask)
    
    if n_donors == 0
        error("No untreated observations found")
    end
    
    # Build first stage formula
    if isempty(controls)
        f1 = Term(y) ~ FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    else
        control_terms = sum(term.(controls))
        f1 = Term(y) ~ control_terms + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    end
    
    wvec = isnothing(weights) ? nothing : Float64.(d[!, weights])
    untreated_data = d[untreated_mask, :]

    # Fit first stage
    m1 = reg(untreated_data, f1, Vcov.robust();
             weights = weights,
             save = :fe)

    # Extract first stage R²
    first_stage_r2 = try
        StatsAPI.r2(m1)
    catch
        0.0
    end

    # === AUTOSAMPLE: Check which treated observations can be imputed ===
    treated_mask = BitVector(d[!, :_ATT] .== 1)
    keep_mask, n_dropped = apply_autosample(d, m1, treated_mask; 
                                             autosample=autosample, verbose=true)
    
    if n_dropped > 0
        # Filter the dataset
        d = d[keep_mask, :]
        treated_mask = BitVector(d[!, :_ATT] .== 1)
        untreated_mask = donor_mask(d; t=t, g=g, control_type=control_type)
        # Update weights
        wvec = isnothing(weights) ? nothing : Float64.(d[!, weights])
    end
    
    # Update counts after autosample
    n_treated = sum(d[!, :_ATT])
    n_donors = sum(untreated_mask)
    
    # Extract treatment periods for diagnostics (excluding reference)
    unique_event_times = sort(unique(d[!, :ttt]))
    treatment_periods = filter(x -> x != ref_p, unique_event_times)

    # Residualize outcome 
    y_pred = fe_predict(m1, d)
    if any(ismissing, y_pred)
        y_pred = coalesce.(y_pred, 0.0)
    end
    d[!, :ytilde] = Float64.(d[!, y]) .- Float64.(y_pred)
    
    # Store first stage residuals for influence function (zero out treated)
    first_u = copy(d[!, :ytilde]) 
    first_u[d[!, :_ATT] .== 1] .= 0.0

    # Get unique event times and sort them numerically
    event_times_to_include = filter(x -> x != ref_p, unique_event_times)
    
    # Create dummy variables for each event time (excluding reference)
    dummy_names = Symbol[]
    for et in event_times_to_include
        dummy_name = Symbol("τ::$(et)")
        d[!, dummy_name] = Float64.(d[!, :ttt] .== et)
        push!(dummy_names, dummy_name)
    end
    
    # Build formula programmatically: ytilde ~ τ::-20 + τ::-19 + ... + τ::20
    dummy_terms = vcat(term(0), [term(name) for name in dummy_names])
    rhs = reduce(+, dummy_terms)
    f2 = Term(:ytilde) ~ rhs
    
    # Fit second stage regression
    m2 = reg(d, f2, Vcov.robust(); weights = weights)
    
    
    # Apply influence function correction (same logic as static case)
    X1 = sparse_rhs_fe(d; controls=controls, fe=(id, t))
    X2 = StatsModels.modelmatrix(f2, d)

    # Weights
    w = isnothing(wvec) ? ones(Float64, nrow(d)) : sqrt.(wvec)

    # Weight the matrices
    X1w = spdiagm(0 => w) * X1
    X2w = spdiagm(0 => w) * sparse(X2)

    # x10: Copy weighted X1 then zero out treated rows (like static)
    X10w = copy(X1w)
    treated_mask = d[!, :_ATT] .== 1
    treated_indices = findall(treated_mask)
    for i in treated_indices
        X10w[i, :] .= 0.0
    end

    # Weight residuals (like static that works)
    second_u = residuals(m2, d) .* w  
    first_u = copy(d[!, :ytilde]) .* w   
    first_u[treated_mask] .= 0.0

    # Cross products
    X2TX2 = Matrix(X2w' * X2w)
    X1TX2 = Matrix(X1w' * X2w)
    X10TX10 = Matrix(X10w' * X10w)

    # Matrix inverses
    inv_X2TX2 = try
        inv(cholesky(Symmetric(X2TX2)))
    catch
        pinv(X2TX2)
    end

    # Influence function
    IF_ss = inv_X2TX2 * transpose(X2w .* second_u)  

    solve_term = try
        X10TX10 \ X1TX2  
    catch
        pinv(X10TX10) * X1TX2
    end

    temp_matrix = transpose(X10w .* first_u)  
    IF_fs = inv_X2TX2 * transpose(solve_term) * temp_matrix  

    # Total IF
    IF = IF_fs .- IF_ss  

    # Cluster robust variance
    cl = d[!, cluster]
    cluster_ids = unique(cl)
    cluster_sums = zeros(Float64, size(IF, 1), length(cluster_ids))

    for (i, cid) in enumerate(cluster_ids)
        obs_idx = findall(cl .== cid)
        cluster_sums[:, i] = sum(IF[:, obs_idx], dims=2)
    end

    Σ = cluster_sums * cluster_sums'
    
    # Extract coefficients and names
    β = StatsAPI.coef(m2)
    names = StatsAPI.coefnames(m2)
    nobs_val = StatsAPI.nobs(m2)
    dof_resid = StatsAPI.dof_residual(m2)


    # Calculate R2 and adjusted R2 (like static but with multiple coefficients)
    residuals_vals = residuals(m2, d)
    y_vals = d[!, :ytilde]
    k = length(β)  # number of coefficients (multiple for dynamic)

    if !isnothing(wvec)
        w_sum = sum(wvec)
        y_weighted_mean = sum(wvec .* y_vals) / w_sum
        TSS_A = sum(wvec .* (y_vals .- y_weighted_mean).^2)
        RSS_A = sum(wvec .* residuals_vals.^2)
    else
        y_mean = mean(y_vals)
        TSS_A = sum((y_vals .- y_mean).^2)
        RSS_A = sum(residuals_vals.^2)
    end

    # Regular R² with DOF correction (like fixest)
    second_stage_r2 = 1 - (RSS_A/TSS_A) * (nobs_val / (nobs_val - 1))

    # Adjusted R² with proper DOF adjustment for multiple coefficients
    second_stage_adjr2 = 1 - (RSS_A/TSS_A) * ((nobs_val - 1) / (nobs_val - k-1))
    
    return GardnerModel(
        β, Σ, names, nobs_val, dof_resid;
        y_name = y,
        estimator_type = :dynamic,
        treatment_periods = treatment_periods,
        first_stage_r2 = first_stage_r2,
        second_stage_r2 = second_stage_r2,
        second_stage_adjr2 = second_stage_adjr2,
        n_treated = n_treated,
        n_donors = n_donors,
        control_type = control_type,
        cluster = isnothing(cluster) ? :none : cluster
    )
end
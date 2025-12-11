function quiet_reg(args...; kwargs...)
    with_logger(SimpleLogger(stderr, Logging.Warn)) do
        reg(args...; kwargs...)
    end
end
"""
    update_weights_control!(W, x; donors, touse, wei)

Orthogonalize each column of W with respect to regressor x.
Numerator sums over ALL observations, update applies only to donors.
"""
function update_weights_control!(
    W::AbstractMatrix{<:Real},
    x::AbstractVector{<:Real};
    donors::AbstractVector{Bool},
    touse::AbstractVector{Bool},
    wei::AbstractVector{<:Real},
    )
    n, k = size(W)
    
    # denom = sum(wei * x^2) over DONORS only
    denom = 0.0
    @inbounds for i in 1:n
        if donors[i] && touse[i]
            denom += wei[i] * (x[i]^2)
        end
    end
    
    if denom <= 1e-14
        return nothing
    end
    
    @inbounds for j in 1:k
        # Numerator: sum over ALL observations
        num = 0.0
        for i in 1:n
            if touse[i]
                num += wei[i] * W[i, j] * x[i]
            end
        end
        β = num / denom
        
        # Update: only donors
        for i in 1:n
            if donors[i] && touse[i]
                W[i, j] -= β * x[i]
            end
        end
    end
    
    return nothing
end

"""
    update_weights_fe!(W, fe; donors, touse, wei)

Orthogonalize each column of W with respect to FE dummies.
Numerator sums over ALL observations, update applies only to donors.
"""
function update_weights_fe!(
    W::AbstractMatrix{<:Real},
    fe::AbstractVector;
    donors::AbstractVector{Bool},
    touse::AbstractVector{Bool},
    wei::AbstractVector{<:Real},
    )
    n, k = size(W)
    
    # Build groups including ALL observations
    groups_all = Dict{eltype(fe), Vector{Int}}()
    @inbounds for i in 1:n
        if touse[i]
            g = fe[i]
            if !haskey(groups_all, g)
                groups_all[g] = Int[]
            end
            push!(groups_all[g], i)
        end
    end
    
    # Precompute donor-only denominators
    denom_g = Dict{eltype(fe), Float64}()
    @inbounds for i in 1:n
        if touse[i] && donors[i]
            g = fe[i]
            denom_g[g] = get(denom_g, g, 0.0) + wei[i]
        end
    end
    
    @inbounds for (g, all_idxs) in groups_all
        dg = get(denom_g, g, 0.0)
        if dg <= 1e-14
            continue
        end
        
        for j in 1:k
            # Numerator: sum over ALL observations in group
            num = 0.0
            for i in all_idxs
                num += wei[i] * W[i, j]
            end
            proj = num / dg
            
            # Update: only donors
            for i in all_idxs
                if donors[i]
                    W[i, j] -= proj
                end
            end
        end
    end
    
    return nothing
end

"""
    compute_influence_weights!(df, wtr_syms, controls, fe_syms, D_sym, wei_sym; ...)

Compute influence weights via iterative orthogonalization.
"""
function compute_influence_weights!(
    df::DataFrame,
    wtr_syms::Vector{Symbol},
    controls::Vector{Symbol},
    fe_syms::Vector{Symbol},
    D_sym::Symbol,
    wei_sym::Union{Nothing,Symbol};
    touse_sym::Union{Nothing,Symbol}=nothing,
    tol::Float64=1e-6,
    maxiter::Int=1000,
    )
    n = nrow(df)
    k = length(wtr_syms)
    
    D_vec = Int.(df[!, D_sym])
    donors = D_vec .== 0
    touse = touse_sym === nothing ? trues(n) : Bool.(df[!, touse_sym])
    wei = isnothing(wei_sym) ? ones(Float64, n) : Float64.(df[!, wei_sym])
    
    # Initialize W with raw wtr columns
    W = Array{Float64}(undef, n, k)
    for (j, sym) in pairs(wtr_syms)
        W[:, j] = Float64.(coalesce.(df[!, sym], 0.0))
    end
    
    # Precompute demeaned controls
    dm_controls = Dict{Symbol, Vector{Float64}}()
    control_denoms = Dict{Symbol, Float64}()
    
    for c in controls
        x = Float64.(df[!, c])
        
        # Weighted mean among donors
        num, den = 0.0, 0.0
        @inbounds for i in 1:n
            if donors[i] && touse[i]
                num += wei[i] * x[i]
                den += wei[i]
            end
        end
        μ = den > 0 ? num / den : 0.0
        x_dm = x .- μ
        
        # Denom = sum(wei * x_dm^2 | D==0 & touse)
        denom = 0.0
        @inbounds for i in 1:n
            if donors[i] && touse[i]
                denom += wei[i] * x_dm[i]^2
            end
        end
        
        dm_controls[c] = x_dm
        control_denoms[c] = denom
    end
    
    # Iterate until convergence
    iter = 0
    while iter < maxiter
        iter += 1
        W_old = copy(W)
        
        # 1) Simple controls
        for c in controls
            x_dm = dm_controls[c]
            denom = control_denoms[c]
            if denom <= 1e-14
                continue
            end
            update_weights_control!(W, x_dm; donors=donors, touse=touse, wei=wei)
        end
        
        # 2) Fixed effects
        for fe in fe_syms
            fe_vec = df[!, fe]
            update_weights_fe!(W, fe_vec; donors=donors, touse=touse, wei=wei)
        end
        
        # 3) Convergence check: SUM of absolute changes (matches Stata)
        total_change = 0.0
        @inbounds for j in 1:k
            for i in 1:n
                if donors[i] && touse[i]
                    total_change += abs(W[i, j] - W_old[i, j])
                end
            end
        end
        
        if total_change <= tol
            break
        end
    end
    
    if iter == maxiter
        @warn "compute_influence_weights!: did not converge within $maxiter iterations"
    end
    
    return W, iter
end

"""
    compute_cluster_scores_tau!(E_tau, d, wtr_syms, W, wei, cluster, g_sym, t_sym; ...)

Compute cluster-level influence scores for treatment effects (tau).
Returns matrix E where E[g,j] = cluster score for cluster g, coefficient j.
"""
function compute_cluster_scores_tau!(
    d::DataFrame,
    wtr_syms::Vector{Symbol},
    W::Matrix{Float64},
    wei::Vector{Float64},
    cluster_vec::AbstractVector,
    cluster_ids::Vector,
    g_sym::Symbol,
    t_sym::Symbol;
    treat_sym::Symbol = :D,
    avgeffectsby::Vector{Symbol} = Symbol[]
    )
    n, k = size(W)
    G = length(cluster_ids)
    
    # Default avgeffectsby to (cohort, time)
    if isempty(avgeffectsby)
        avgeffectsby = [g_sym, t_sym]
    end
    
    treated = d[!, treat_sym] .== 1
    effect = Float64.(d[!, :effect])
    
    # Create avgeffectsby group identifier
    avgby_vec = Vector{String}(undef, n)
    for i in 1:n
        key_parts = String[]
        for s in avgeffectsby
            push!(key_parts, string(d[i, s]))
        end
        avgby_vec[i] = join(key_parts, "_")
    end
    
    # Pre-allocate influence scores
    E = zeros(Float64, G, k)
    
    # Build cluster index mapping
    cluster_idx_map = Dict{eltype(cluster_vec), Int}()
    for (idx, g_val) in enumerate(cluster_ids)
        cluster_idx_map[g_val] = idx
    end
    
    # Compute resid0 for donors (Y - Y0)
    resid0 = zeros(Float64, n)
    for i in 1:n
        if !treated[i]
            resid0[i] = effect[i]  # effect = Y - Y0
        end
    end
    
    # Process each coefficient
    for j in 1:k
        wtrj = Float64.(coalesce.(d[!, wtr_syms[j]], 0.0))
        w_infl = W[:, j]
        
        # Compute cluster weights within (cluster, avgeffectsby) groups
        cw_key = Dict{Tuple{Any,String}, Float64}()
        for i in 1:n
            if treated[i] && wtrj[i] != 0
                key = (cluster_vec[i], avgby_vec[i])
                cw_key[key] = get(cw_key, key, 0.0) + wei[i] * wtrj[i]
            end
        end
        
        # Map to observation-level clusterweight
        clusterweight = zeros(Float64, n)
        for i in 1:n
            if treated[i]
                key = (cluster_vec[i], avgby_vec[i])
                clusterweight[i] = get(cw_key, key, 0.0)
            end
        end
        
        # Compute smartdenom within avgeffectsby groups
        smartdenom_by_avgby = Dict{String, Float64}()
        for i in 1:n
            if treated[i] && wtrj[i] != 0
                key = avgby_vec[i]
                smartdenom_by_avgby[key] = get(smartdenom_by_avgby, key, 0.0) + 
                                            clusterweight[i] * wei[i] * wtrj[i]
            end
        end
        
        # Compute smart weights
        smartweight = zeros(Float64, n)
        for i in 1:n
            if treated[i] && wtrj[i] != 0
                sd = get(smartdenom_by_avgby, avgby_vec[i], 0.0)
                if sd > 1e-14
                    smartweight[i] = clusterweight[i] * wei[i] * wtrj[i] / sd
                end
            end
        end
        
        # Compute avgtau within avgeffectsby groups
        avgtau_by_avgby = Dict{String, Float64}()
        for i in 1:n
            if treated[i]
                key = avgby_vec[i]
                avgtau_by_avgby[key] = get(avgtau_by_avgby, key, 0.0) + 
                                        effect[i] * smartweight[i]
            end
        end
        
        # Compute residuals
        resid = zeros(Float64, n)
        for i in 1:n
            if treated[i]
                avgtau_i = get(avgtau_by_avgby, avgby_vec[i], 0.0)
                resid[i] = effect[i] - avgtau_i
            else
                resid[i] = resid0[i]
            end
        end
        
        # Compute cluster-level influence scores
        for i in 1:n
            g_idx = cluster_idx_map[cluster_vec[i]]
            E[g_idx, j] += wei[i] * w_infl[i] * resid[i]
        end
    end
    
    return E
end

"""
    compute_cluster_scores_pretrends!(...)

Compute cluster-level influence scores for pre-trend coefficients.
1. Run regression Y ~ controls + pre_dummies + FE on D==0, save residual
2. For each pre_dummy: residualize on others + controls + FE
3. Normalize weights, compute cluster scores
"""
function compute_cluster_scores_pretrends!(
    d::DataFrame,
    pre_syms::Vector{Symbol},
    controls::Vector{Symbol},
    fe::Tuple,
    wei::Vector{Float64},
    cluster_vec::AbstractVector,
    cluster_ids::Vector,
    cluster::Union{Nothing,Symbol},
    weights::Union{Nothing,Symbol};
    y::Symbol,
    touse_sym::Symbol = :__touse
    )
    n = nrow(d)
    k_pre = length(pre_syms)
    G = length(cluster_ids)
    
    if k_pre == 0
        return zeros(Float64, G, 0), Float64[], Int[]
    end
    
    donors = d[!, :D] .== 0
    touse = d[!, touse_sym]
    donor_mask = donors .& touse
    
    # Build cluster index mapping
    cluster_idx_map = Dict{eltype(cluster_vec), Int}()
    for (idx, g_val) in enumerate(cluster_ids)
        cluster_idx_map[g_val] = idx
    end
    
    # Create donor subset with original row indices (copy to avoid modifying original)
    donor_global_idx = findall(donor_mask)
    d_donors = copy(d[donor_mask, :])
    d_donors[!, :__orig_row] = donor_global_idx
    n_donors = nrow(d_donors)
    
    # 1. Run pre-trend regression: Y ~ controls + pre_dummies + FE on D==0
    fe_terms = sum(FixedEffectModels.fe.(fe))
    pre_terms = sum(term.(pre_syms))
    
    if isempty(controls)
        f_pre = Term(y) ~ pre_terms + fe_terms
    else
        control_terms = sum(term.(controls))
        f_pre = Term(y) ~ control_terms + pre_terms + fe_terms
    end
    
    m_pre = quiet_reg(d_donors, f_pre; weights=weights, save=:residuals,progress_bar=false)
    
    # Compute DOF adjustment exactly as Stata does (line 564 of did_imputation.ado)
    N_pre = nobs(m_pre)
    df_m_pre = dof(m_pre)
    N_clust_pre = isnothing(cluster) ? N_pre : length(unique(d_donors[m_pre.esample, cluster]))
    
    # Compute df_a properly: only count FE that are NOT nested within the cluster
    df_a_pre = 0
    for fe_var in fe
        fe_col = d_donors[m_pre.esample, fe_var]
        n_levels = length(unique(fe_col))
        
        if isnothing(cluster)
            df_a_pre += n_levels - 1
        elseif fe_var == cluster
            # FE same as cluster - contributes 0
        else
            cluster_col = d_donors[m_pre.esample, cluster]
            is_nested = true
            fe_to_cluster = Dict{eltype(fe_col), Set{eltype(cluster_col)}}()
            for (f, c) in zip(fe_col, cluster_col)
                if !haskey(fe_to_cluster, f)
                    fe_to_cluster[f] = Set{eltype(cluster_col)}()
                end
                push!(fe_to_cluster[f], c)
                if length(fe_to_cluster[f]) > 1
                    is_nested = false
                    break
                end
            end
            
            if !is_nested
                df_a_pre += n_levels - 1
            end
        end
    end
    
    denom = N_pre - df_m_pre - df_a_pre
    if denom <= 0
        denom = 1
    end
    dof_adj_pre = ((N_pre - 1) / denom) * (N_clust_pre / max(N_clust_pre - 1, 1))
    
    # Get coefficients and identify valid (non-omitted) pre-trends
    pre_coefnames = coefnames(m_pre)
    pre_coefs_all = coef(m_pre)
    pre_se_all = stderror(m_pre)
    
    pre_β = Float64[]
    valid_pre_syms = Symbol[]
    valid_pre_idx = Int[]
    
    for (j, sym) in enumerate(pre_syms)
        idx = findfirst(==(String(sym)), pre_coefnames)
        if idx !== nothing
            b = pre_coefs_all[idx]
            se = pre_se_all[idx]
            is_omitted = abs(b) < 1e-10 && (abs(se) < 1e-10 || !isfinite(se))
            if !is_omitted
                push!(pre_β, b)
                push!(valid_pre_syms, sym)
                push!(valid_pre_idx, j)
            end
        end
    end
    
    omitted_pre = setdiff(pre_syms, valid_pre_syms)
    if !isempty(omitted_pre)
        @warn "Collinear pre-trends omitted: $(join(omitted_pre, ", "))"
    end
    k_valid = length(valid_pre_syms)
    if k_valid == 0
        return zeros(Float64, G, 0), Float64[], Int[]
    end
    
    # 2. Get preresid and map back to full dataset using esample
    preresid_raw = residuals(m_pre)
    esample = m_pre.esample
    
    # Map residuals back to full dataset
    preresid = zeros(Float64, n)
    for i in 1:n_donors
        if i <= length(esample) && esample[i] && i <= length(preresid_raw)
            val = preresid_raw[i]
            if !ismissing(val)
                orig_row = d_donors[i, :__orig_row]
                preresid[orig_row] = Float64(val)
            end
        end
    end
    
    # 4. For each valid pre-trend, compute influence weights and cluster scores
    E_pre = zeros(Float64, G, k_valid)
    
    for (j, sym) in enumerate(valid_pre_syms)
        pre_dummy = Float64.(d[!, sym])
        
        other_pre = filter(s -> s != sym, valid_pre_syms)
        
        rhs_terms = []
        if !isempty(controls)
            push!(rhs_terms, sum(term.(controls)))
        end
        if !isempty(other_pre)
            push!(rhs_terms, sum(term.(other_pre)))
        end
        
        if isempty(rhs_terms)
            f_resid = Term(sym) ~ fe_terms
        else
            f_resid = Term(sym) ~ reduce(+, rhs_terms) + fe_terms
        end
        
        m_resid = quiet_reg(d_donors, f_resid, Vcov.simple(); weights=weights, save=:residuals,progress_bar=false)
        preweight_raw = residuals(m_resid)
        esample_resid = m_resid.esample
        
        # Map back to full dataset and multiply by wei
        preweight = zeros(Float64, n)
        for i in 1:n_donors
            if i <= length(esample_resid) && esample_resid[i] && i <= length(preweight_raw)
                val = preweight_raw[i]
                if !ismissing(val)
                    orig_row = d_donors[i, :__orig_row]
                    preweight[orig_row] = Float64(val) * wei[orig_row]
                end
            end
        end
        
        # Normalize: sum(preweight) where D==0 & touse & pretrendvar==1
        norm_sum = 0.0
        n_pre_obs = 0
        for i in 1:n
            if donor_mask[i] && pre_dummy[i] == 1.0
                norm_sum += preweight[i]
                n_pre_obs += 1
            end
        end
        
        if abs(norm_sum) > 1e-14
            for i in 1:n
                preweight[i] /= norm_sum
            end
        end
        
        # Compute cluster scores
        for i in 1:n
            if touse[i]
                g_idx = cluster_idx_map[cluster_vec[i]]
                E_pre[g_idx, j] += preweight[i] * preresid[i]
            end
        end
        
        E_pre[:, j] .*= sqrt(dof_adj_pre)
    end
    
    return E_pre, pre_β, valid_pre_idx
end

"""
    compute_cluster_scores_controls!(...)

Compute cluster scores for control variable coefficients.
"""
function compute_cluster_scores_controls!(
    d::DataFrame,
    controls::Vector{Symbol},
    fe::Tuple,
    wei::Vector{Float64},
    cluster_vec::AbstractVector,
    cluster_ids::Vector,
    cluster::Union{Nothing,Symbol},
    weights::Union{Nothing,Symbol},
    imput_resid::Vector{Float64};
    y::Symbol,
    touse_sym::Symbol = :__touse
    )
    n = nrow(d)
    k_ctrl = length(controls)
    G = length(cluster_ids)
    
    if k_ctrl == 0
        return zeros(Float64, G, 0), Float64[], Symbol[]
    end
    
    donors = d[!, :D] .== 0
    touse = d[!, touse_sym]
    donor_mask = donors .& touse
    
    cluster_idx_map = Dict{eltype(cluster_vec), Int}()
    for (idx, g_val) in enumerate(cluster_ids)
        cluster_idx_map[g_val] = idx
    end
    
    donor_global_idx = findall(donor_mask)
    d_donors = copy(d[donor_mask, :])
    d_donors[!, :__orig_row] = donor_global_idx
    n_donors = nrow(d_donors)
    
    fe_terms = sum(FixedEffectModels.fe.(fe))
    if isempty(controls)
        return zeros(Float64, G, 0), Float64[], Symbol[]
    end
    
    control_terms = sum(term.(controls))
    f_ctrl = Term(y) ~ control_terms + fe_terms
    
    m_ctrl = quiet_reg(d_donors, f_ctrl; weights=weights, save=:residuals,progress_bar=false)
    
    ctrl_coefnames = coefnames(m_ctrl)
    ctrl_coefs_all = coef(m_ctrl)
    ctrl_se_all = stderror(m_ctrl)
    
    N_ctrl = nobs(m_ctrl)
    df_m_ctrl = dof(m_ctrl)
    N_clust_ctrl = isnothing(cluster) ? N_ctrl : length(unique(d_donors[m_ctrl.esample, cluster]))
    
    df_a_ctrl = 0
    for fe_var in fe
        fe_col = d_donors[m_ctrl.esample, fe_var]
        n_levels = length(unique(fe_col))
        
        if isnothing(cluster)
            df_a_ctrl += n_levels - 1
        elseif fe_var == cluster
            # FE same as cluster - contributes 0
        else
            cluster_col = d_donors[m_ctrl.esample, cluster]
            is_nested = true
            fe_to_cluster = Dict{eltype(fe_col), Set{eltype(cluster_col)}}()
            for (f, c) in zip(fe_col, cluster_col)
                if !haskey(fe_to_cluster, f)
                    fe_to_cluster[f] = Set{eltype(cluster_col)}()
                end
                push!(fe_to_cluster[f], c)
                if length(fe_to_cluster[f]) > 1
                    is_nested = false
                    break
                end
            end
            
            if !is_nested
                df_a_ctrl += n_levels - 1
            end
        end
    end
    
    denom = N_ctrl - df_m_ctrl - df_a_ctrl
    if denom <= 0
        denom = 1
    end
    dof_adj_ctrl = ((N_ctrl - 1) / denom) * (N_clust_ctrl / max(N_clust_ctrl - 1, 1))
    
    β_ctrl = Float64[]
    valid_ctrl_syms = Symbol[]
    valid_ctrl_idx = Int[]
    
    for (j, ctrl) in enumerate(controls)
        idx = findfirst(==(String(ctrl)), ctrl_coefnames)
        if idx !== nothing
            b = ctrl_coefs_all[idx]
            se = ctrl_se_all[idx]
            is_omitted = abs(b) < 1e-10 && (abs(se) < 1e-10 || !isfinite(se))
            if !is_omitted
                push!(β_ctrl, b)
                push!(valid_ctrl_syms, ctrl)
                push!(valid_ctrl_idx, j)
            end
        end
    end
    
    omitted_ctrls = setdiff(controls, valid_ctrl_syms)
    if !isempty(omitted_ctrls)
        @warn "Collinear controls omitted: $(join(omitted_ctrls, ", "))"
    end
    k_valid = length(valid_ctrl_syms)
    if k_valid == 0
        return zeros(Float64, G, 0), Float64[], Symbol[]
    end
    
    E_ctrl = zeros(Float64, G, k_valid)
    
    for (j, ctrl) in enumerate(valid_ctrl_syms)
        ctrl_vec = Float64.(d[!, ctrl])
        
        other_ctrl = filter(s -> s != ctrl, valid_ctrl_syms)
        
        if isempty(other_ctrl)
            f_resid = Term(ctrl) ~ fe_terms
        else
            f_resid = Term(ctrl) ~ sum(term.(other_ctrl)) + fe_terms
        end
        
        m_resid = quiet_reg(d_donors, f_resid, Vcov.simple(); weights=weights, save=:residuals,progress_bar=false)
        ctrlweight_raw = residuals(m_resid)
        esample_resid = m_resid.esample
        
        # Map back to full dataset and multiply by wei
        ctrlweight = zeros(Float64, n)
        for i in 1:n_donors
            if i <= length(esample_resid) && esample_resid[i] && i <= length(ctrlweight_raw)
                val = ctrlweight_raw[i]
                if !ismissing(val)
                    orig_row = d_donors[i, :__orig_row]
                    ctrlweight[orig_row] = Float64(val) * wei[orig_row]
                end
            end
        end
        
        # Normalize
        norm_sum = 0.0
        for i in 1:n
            if donor_mask[i]
                norm_sum += ctrlweight[i] * ctrl_vec[i]
            end
        end
        
        if abs(norm_sum) > 1e-14
            for i in 1:n
                ctrlweight[i] /= norm_sum
            end
        end
        
        # Compute cluster scores
        for i in 1:n
            if touse[i]
                g_idx = cluster_idx_map[cluster_vec[i]]
                E_ctrl[g_idx, j] += ctrlweight[i] * imput_resid[i]
            end
        end
        
        E_ctrl[:, j] .*= sqrt(dof_adj_ctrl)
    end
    
    return E_ctrl, β_ctrl, valid_ctrl_syms
end



"""
    fit_bjs(df::DataFrame;
        y::Symbol,
        id::Symbol,
        t::Symbol,
        g::Symbol,
        controls::Vector{Symbol} = Symbol[],
        fe::Tuple = (id, t),
        weights::Union{Nothing,Symbol} = nothing,
        cluster::Union{Nothing,Symbol} = nothing,
        horizons::Union{Nothing,Bool,Vector{Int}} = nothing,
        pretrends::Union{Nothing,Bool,Int,Vector{Int}} = nothing,
        avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
        control_type::Symbol = :notyet,
        tol::Float64 = 1e-6,
        maxiter::Int = 1000,
        autosample::Bool = true)

Estimate treatment effects using the Borusyak, Jaravel, and Spiess (2023) 
imputation estimator for staggered adoption designs. This implementation 
follows the logic of the official Stata `did_imputation` command, including 
its analytic influence-function variance estimation.

# Arguments
- `df::DataFrame`: Long panel dataset in (i,t) format.
- `y::Symbol`: Outcome variable.
- `id::Symbol`: Unit (e.g. firm, fund) identifier.
- `t::Symbol`: Time period identifier (integer-coded).
- `g::Symbol`: First treatment period for each unit. Missing/0 values indicate never-treated units.

# Keyword Arguments
- `controls::Vector{Symbol}`: Time-varying covariates included in the imputation regression.
- `fe::Tuple = (id, t)`: Absorbed fixed effects. Default is unit + time FE.
- `weights::Union{Nothing,Symbol}`: Optional observation weights.
- `cluster::Union{Nothing,Symbol}`: Clustering variable for standard errors.
- `horizons::Union{Nothing,Bool,Vector{Int}}`: Controls dynamic vs. static estimation:
    - `nothing` → static ATT (single coefficient)
    - `true` → estimate all post-treatment horizons (τ ≥ 0)  
    - `Vector{Int}` → estimate only selected horizons (e.g., `[0,1,2,3,4,5]`)
- `pretrends::Union{Nothing,Bool,Int,Vector{Int}}`: Pre-trend coefficients:
    - `nothing` or `false` → no pre-trend terms  
    - `true` → all negative horizons (τ < 0)
    - `Int` → number of pre-periods (e.g., `5` means K=-1 to K=-5)
    - `Vector{Int}` → specific pre-horizons
- `avgeffectsby`: Grouping variables for averaging treatment effects. Default is `[g, t]` (cohort x time), matching Stata's default behavior.
- `control_type::Symbol = :notyet`: Donor pool definition.
- `tol::Float64 = 1e-6`: Convergence tolerance for influence weight iteration.
- `maxiter::Int = 1000`: Maximum iterations for influence weight computation.
- `autosample::Bool = true`: If true (default), drop treated observations where FE cannot 
                             be imputed from donor sample. If false, error.

# Returns
`BJSModel` object.

# Notes
- Never-treated units: code as `missing` or `0` in `g`
- Time variable must be integer-coded
- When `autosample=true`, observations where FE cannot be imputed are dropped with a warning.
"""
function fit_bjs_st(df::DataFrame;
    y::Symbol,
    id::Symbol,
    t::Symbol,
    g::Symbol,
    controls::Vector{Symbol} = Symbol[],
    fe::Tuple = (id, t),
    weights::Union{Nothing,Symbol} = nothing,
    cluster::Union{Nothing,Symbol} = nothing,
    horizons::Union{Nothing,Bool,Vector{Int}} = nothing,
    pretrends::Union{Nothing,Bool,Int,Vector{Int}} = nothing,
    avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true
    )
    if avgeffectsby === nothing
        avgeffectsby = [g, t]
    end
    
    #=== 0. Copy and clean data ===#
    d = copy(df)
    d[!, g] = replace(d[!, g], 0 => missing)
    key_vars = Symbol[y, id, t]
    append!(key_vars, controls)
    if !isnothing(weights); push!(key_vars, weights); end
    if !isnothing(cluster); push!(key_vars, cluster); end
    key_vars = unique(key_vars)
    
    miss = falses(nrow(d))
    for v in key_vars
        miss .|= ismissing.(d[!, v])
    end
    if any(miss)
        d = d[.!miss, :]
    end
    
    n = nrow(d)
    n == 0 && error("No observations left after dropping missings.")
    
    d[!, :__touse] = trues(n)
    
    #=== 1. Event time K and treatment indicator D ===#
    K = Vector{Union{Int,Missing}}(undef, n)
    for i in 1:n
        gi = d[i, g]
        K[i] = (ismissing(gi) || gi == 0) ? missing : d[i, t] - gi
    end
    d[!, :K] = K
    
    D = Vector{Int}(undef, n)
    for i in 1:n
        D[i] = ismissing(K[i]) ? 0 : (K[i] >= 0 ? 1 : 0)
    end
    d[!, :D] = D
    
    donors = D .== 0
    n_donors = count(donors)
    n_treated = n - n_donors
    n_donors == 0 && error("No donor observations.")
    
    #=== 2. Observation weights ===#
    wei = isnothing(weights) ? ones(Float64, n) : Float64.(d[!, weights])
    
    #=== 3. First-stage regression ===#
    fe_terms = sum(FixedEffectModels.fe.(fe))
    f1 = isempty(controls) ? 
         Term(y) ~ fe_terms : 
         Term(y) ~ sum(term.(controls)) + fe_terms
    
    m1 = quiet_reg(d[donors, :], f1; weights=weights, save=:all, progress_bar=false)
    r2_1 = FixedEffectModels.r2(m1)
    
    #=== AUTOSAMPLE: Check which treated observations can be imputed ===#
    treated_mask = BitVector(D .== 1)
    keep_mask, n_dropped = apply_autosample(d, m1, treated_mask; 
                                             autosample=autosample, verbose=true)
    
    if n_dropped > 0
        d = d[keep_mask, :]
        n = nrow(d)
        
        wei = isnothing(weights) ? ones(Float64, n) : Float64.(d[!, weights])
        D = d[!, :D]
        K = d[!, :K]
        donors = D .== 0
        n_donors = count(donors)
        n_treated = n - n_donors
        treated_mask = BitVector(D .== 1)
    end
    
    # Save first-stage imputation residual (needed for control SEs)
    
    imput_resid_raw = residuals(m1)
    esample_m1 = m1.esample
    imput_resid = zeros(Float64, n)
    
    # Map residuals: donors in current d should match donors in original fit
    donor_indices = findall(donors)
    for (i, donor_i) in enumerate(donor_indices)
        if i <= length(esample_m1) && i <= length(imput_resid_raw) && esample_m1[i]
            val = imput_resid_raw[i]  
            if !ismissing(val)
                imput_resid[donor_i] = Float64(val)
            end
            # If missing, imput_resid[donor_i] stays 0.0 (already initialized)
        end
    end
    
    #=== 4. Predict Y0 and compute effect ===#
    y0hat = fe_predict(m1, d)
    d[!, :Y0_hat] = y0hat
    
    eff = Float64.(d[!, y]) .- Float64.(coalesce.(y0hat, 0.0))
    d[!, :effect] = eff
    
    keep = .!ismissing.(y0hat)
    if any(.!keep)
        d = d[keep, :]
        imput_resid = imput_resid[keep]
        n = nrow(d)
        wei = isnothing(weights) ? ones(Float64, n) : Float64.(d[!, weights])
        D = d[!, :D]
        K = d[!, :K]
        donors = D .== 0
        n_donors = count(donors)
        n_treated = n - n_donors
        eff = d[!, :effect]
    end
    
    d[!, :__imput_resid] = imput_resid
    
    #=== 5. Construct wtr weights for tau ===#
    wtr_syms = Symbol[]
    if horizons === nothing || horizons === false
        push!(wtr_syms, :__wtr_static)
        d[!, :__wtr_static] = Float64.(D .== 1)
        s = sum(wei[i] * d[i, :__wtr_static] for i in 1:n if D[i] == 1)
        s > 0 && (d[!, :__wtr_static] ./= s)
    else
        τvals = horizons === true ? 
                unique(filter(!ismissing, K[D .== 1])) : 
                collect(horizons)
        τvals = sort(unique(filter(≥(0), τvals)))
        
        for τ in τvals
            nm = Symbol("__wtr$(τ)")
            push!(wtr_syms, nm)
            col = zeros(Float64, n)
            for i in 1:n
                if D[i] == 1 && !ismissing(K[i]) && K[i] == τ
                    col[i] = 1.0
                end
            end
            s = sum(wei[i] * col[i] for i in 1:n)
            s > 0 && (col ./= s)
            d[!, nm] = col
        end
    end
    k_tau = length(wtr_syms)
    
    #=== 6. Point estimates for tau ===#
    β_tau = zeros(Float64, k_tau)
    for j in 1:k_tau
        wname = wtr_syms[j]
        s = 0.0
        for i in 1:n
            if D[i] == 1
                s += eff[i] * d[i, wname] * wei[i]
            end
        end
        β_tau[j] = s
    end
    
    #=== 7. Construct pre-trend dummies ===#
    pre_syms = Symbol[]
    pre_horizons = Int[]
    
    if pretrends !== nothing && pretrends !== false
        if pretrends === true
            all_neg = sort(unique(filter(<(0), filter(!ismissing, K))), rev=true)
            pre_horizons = all_neg
        elseif pretrends isa Int
            pre_horizons = collect(-1:-1:-pretrends)
        else
            pre_horizons = sort(collect(pretrends), rev=true)
        end
        pre_horizons = filter(<(0), pre_horizons)
        
        for τ in pre_horizons
            nm = Symbol("pre$(abs(τ))")
            push!(pre_syms, nm)
            col = zeros(Float64, n)
            for i in 1:n
                if !ismissing(K[i]) && K[i] == τ
                    col[i] = 1.0
                end
            end
            d[!, nm] = col
        end
    end
    
    #=== 8. Compute influence weights for tau ===#
    fe_syms = Symbol[fe...]
    W_tau, iters = compute_influence_weights!(
        d, wtr_syms, controls, fe_syms, :D, weights;
        touse_sym = :__touse,
        tol = tol,
        maxiter = maxiter,
    )
    
    #=== 9. Setup cluster structure ===#
    cluster_vec = isnothing(cluster) ? collect(1:n) : d[!, cluster]
    cluster_ids = unique(cluster_vec)
    G = length(cluster_ids)
    
    #=== 10. Compute cluster scores for tau ===#
    E_tau = compute_cluster_scores_tau!(
        d, wtr_syms, W_tau, wei, cluster_vec, cluster_ids, g, t;
        treat_sym = :D,
        avgeffectsby = avgeffectsby
    )
    
    #=== 11. Compute cluster scores for pre-trends ===#
    E_pre, β_pre, valid_pre_idx = compute_cluster_scores_pretrends!(
        d, pre_syms, controls, fe, wei, cluster_vec, cluster_ids, cluster, weights;
        y = y,
        touse_sym = :__touse
    )
    k_pre = length(β_pre)
    
    #=== 11b. Compute cluster scores for controls ===#
    E_ctrl, β_ctrl, valid_ctrl_syms = compute_cluster_scores_controls!(
        d, controls, fe, wei, cluster_vec, cluster_ids, cluster, weights, 
        d[!, :__imput_resid];
        y = y,
        touse_sym = :__touse
    )
    k_ctrl = length(β_ctrl)
    
    #=== 12. Combine cluster scores and compute joint VCOV ===#
    if k_pre > 0
        E_pre_chron = E_pre[:, end:-1:1]
        β_pre_chron = β_pre[end:-1:1]
        valid_pre_idx_chron = valid_pre_idx[end:-1:1]
    else
        E_pre_chron = zeros(Float64, G, 0)
        β_pre_chron = Float64[]
        valid_pre_idx_chron = Int[]
    end
    
    if k_ctrl > 0
        E_combined = hcat(E_pre_chron, E_tau, E_ctrl)
        all_β = vcat(β_pre_chron, β_tau, β_ctrl)
    else
        E_combined = hcat(E_pre_chron, E_tau)
        all_β = vcat(β_pre_chron, β_tau)
    end
    
    Σ_full = transpose(E_combined) * E_combined
    
    #=== 13. Assemble names ===#
    tau_names = if horizons === nothing || horizons === false
        ["_ATT"]
    else
        τvals = Int[]
        for wname in wtr_syms
            s = String(wname)
            τ = parse(Int, replace(s, "__wtr" => ""))
            push!(τvals, τ)
        end
        ["τ::$(τ)" for τ in sort(τvals)]
    end
    
    pre_names_final = k_pre > 0 ? ["τ::-$(abs(pre_horizons[i]))" for i in valid_pre_idx_chron] : String[]
    ctrl_names_final = k_ctrl > 0 ? String.(valid_ctrl_syms) : String[]
    
    all_names = vcat(pre_names_final, tau_names, ctrl_names_final)
    
    #=== 14. DOF and return ===#
    cluster_sym = isnothing(cluster) ? :_no_cluster : cluster
    n_clusters = isnothing(cluster) ? n : G
    dof_resid = max(n_clusters - 1, 1)
    
    t_periods = if horizons === nothing || horizons === false
        Int[]
    else
        sort(unique(filter(≥(0), filter(!ismissing, K[D .== 1]))))
    end
    
    return BJSModel(
        all_β, Σ_full, all_names, n, dof_resid;
        y_name = y,
        estimator_type = (horizons === nothing || horizons === false) ? :static : :dynamic,
        treatment_periods = unique(skipmissing(t_periods)),
        first_stage_r2 = r2_1,
        n_treated = n_treated,
        n_donors = n_donors,
        control_type = control_type,
        cluster = cluster_sym
    )
end


"""
    fit_bjs(df::DataFrame;
        y::Symbol,
        id::Symbol,
        t::Symbol,
        g::Symbol,
        controls::Vector{Symbol} = Symbol[],
        fe::Tuple = (id, t),
        weights::Union{Nothing,Symbol} = nothing,
        cluster::Union{Nothing,Symbol} = nothing,
        horizons::Union{Nothing,Bool,Vector{Int}} = nothing,
        pretrends::Union{Nothing,Bool,Int,Vector{Int}} = nothing,
        avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
        control_type::Symbol = :notyet,
        tol::Float64 = 1e-6,
        maxiter::Int = 1000,
        autosample::Bool = true,
        multithreaded::Bool = true)

Estimate treatment effects using the Borusyak, Jaravel, and Spiess (2023) 
imputation estimator for staggered adoption designs. This implementation 
follows the logic of the official Stata `did_imputation` command, including 
its analytic influence-function variance estimation.

# Arguments
- `df::DataFrame`: Long panel dataset in (i,t) format.
- `y::Symbol`: Outcome variable.
- `id::Symbol`: Unit (e.g. firm, fund) identifier.
- `t::Symbol`: Time period identifier (integer-coded).
- `g::Symbol`: First treatment period for each unit. Missing/0 values indicate never-treated units.

# Keyword Arguments
- `controls::Vector{Symbol}`: Time-varying covariates included in the imputation regression.
- `fe::Tuple = (id, t)`: Absorbed fixed effects. Default is unit + time FE.
- `weights::Union{Nothing,Symbol}`: Optional observation weights.
- `cluster::Union{Nothing,Symbol}`: Clustering variable for standard errors.
- `horizons::Union{Nothing,Bool,Vector{Int}}`: Controls dynamic vs. static estimation:
    - `nothing` → static ATT (single coefficient)
    - `true` → estimate all post-treatment horizons (τ ≥ 0)  
    - `Vector{Int}` → estimate only selected horizons (e.g., `[0,1,2,3,4,5]`)
- `pretrends::Union{Nothing,Bool,Int,Vector{Int}}`: Pre-trend coefficients:
    - `nothing` or `false` → no pre-trend terms  
    - `true` → all negative horizons (τ < 0)
    - `Int` → number of pre-periods (e.g., `5` means K=-1 to K=-5)
    - `Vector{Int}` → specific pre-horizons
- `avgeffectsby`: Grouping variables for averaging treatment effects. Default is `[g, t]` (cohort x time), matching Stata's default behavior.
- `control_type::Symbol = :notyet`: Donor pool definition.
- `tol::Float64 = 1e-6`: Convergence tolerance for influence weight iteration.
- `maxiter::Int = 1000`: Maximum iterations for influence weight computation.
- `autosample::Bool = true`: If true (default), drop treated observations where FE cannot 
                             be imputed from donor sample. If false, error.
- `multithreaded::Bool = true`: whether to multithread or not.

# Returns
`BJSModel` object.

# Notes
- Never-treated units: code as `missing` or `0` in `g`
- Time variable must be integer-coded
- When `autosample=true`, observations where FE cannot be imputed are dropped with a warning.
"""
function fit_bjs(df::DataFrame;
    y::Symbol,
    id::Symbol,
    t::Symbol,
    g::Symbol,
    controls::Vector{Symbol} = Symbol[],
    fe::Tuple = (id, t),
    weights::Union{Nothing,Symbol} = nothing,
    cluster::Union{Nothing,Symbol} = nothing,
    horizons::Union{Nothing,Bool,Vector{Int}} = nothing,
    pretrends::Union{Nothing,Bool,Int,Vector{Int}} = nothing,
    avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true,
    multithreaded::Bool = true
    )
    
    if multithreaded
        return fit_bjs_mt(df; 
            y=y, id=id, t=t, g=g,
            controls=controls, fe=fe, weights=weights,
            cluster=cluster, horizons=horizons, pretrends=pretrends,
            avgeffectsby=avgeffectsby, control_type=control_type,
            tol=tol, maxiter=maxiter, autosample=autosample
        )
    else
        return fit_bjs_st(df;
            y=y, id=id, t=t, g=g,
            controls=controls, fe=fe, weights=weights,
            cluster=cluster, horizons=horizons, pretrends=pretrends,
            avgeffectsby=avgeffectsby, control_type=control_type,
            tol=tol, maxiter=maxiter, autosample=autosample
        )
    end
end

"""
    fit_bjs_static(df::DataFrame; y, id, t, g, controls=Symbol[], fe=(id,t),
                   weights=nothing, cluster=nothing, control_type=:notyet,
                   tol=1e-6, maxiter=1000, autosample=true, multithreaded=true)

Static ATT estimator using Borusyak, Jaravel, and Spiess (2023) imputation.
Wrapper for `fit_bjs(...; horizons=nothing, pretrends=nothing)`.
"""
function fit_bjs_static(df::DataFrame;
    y::Symbol,
    id::Symbol,
    t::Symbol,
    g::Symbol,
    controls::Vector{Symbol} = Symbol[],
    fe::Tuple = (id, t),
    weights::Union{Nothing,Symbol} = nothing,
    cluster::Union{Nothing,Symbol} = nothing,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true,
    multithreaded::Bool = true)
    
    return fit_bjs(df; y=y, id=id, t=t, g=g,
                   controls=controls, fe=fe, weights=weights,
                   cluster=cluster, horizons=nothing, pretrends=nothing,
                   control_type=control_type, tol=tol, maxiter=maxiter,
                   autosample=autosample,
                   multithreaded = multithreaded)
end


"""
    fit_bjs_dynamic(df::DataFrame; y, id, t, g, controls=Symbol[], fe=(id,t),
                    weights=nothing, cluster=nothing, horizons=true,
                    pretrends=true, avgeffectsby=nothing,
                    control_type=:notyet, tol=1e-6, maxiter=1000, autosample=true, multithreaded=true)

Dynamic event-study estimator using Borusyak, Jaravel, and Spiess (2023) imputation.
Wrapper for `fit_bjs` with `horizons=true` and `pretrends=true` by default.
"""
function fit_bjs_dynamic(df::DataFrame;
    y::Symbol,
    id::Symbol,
    t::Symbol,
    g::Symbol,
    controls::Vector{Symbol} = Symbol[],
    fe::Tuple = (id, t),
    weights::Union{Nothing,Symbol} = nothing,
    cluster::Union{Nothing,Symbol} = nothing,
    horizons::Union{Bool,Vector{Int}} = true,
    pretrends::Union{Nothing,Bool,Int,Vector{Int}} = true,
    avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true,
    multithreaded::Bool = true)
    
    return fit_bjs(df; y=y, id=id, t=t, g=g,
                   controls=controls, fe=fe, weights=weights,
                   cluster=cluster, horizons=horizons, pretrends=pretrends,
                   avgeffectsby=avgeffectsby, control_type=control_type,
                   tol=tol, maxiter=maxiter, autosample=autosample,
                   multithreaded = multithreaded)
end
"""
    compute_cluster_scores_pretrends!(...)

Compute cluster-level influence scores for pre-trend coefficients.
1. Run regression Y ~ controls + pre_dummies + FE on D==0, save residual
2. For each pre_dummy: residualize on others + controls + FE
3. Normalize weights, compute cluster scores
"""
function compute_cluster_scores_pretrends_mt!(
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
    touse_sym::Symbol = :__touse,
    verbose::Bool=false
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
    
    # Create donor subset with original row indices
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
    
    m_pre = quiet_reg(d_donors, f_pre; weights=weights, save=:residuals, progress_bar=false)
    
    # Compute DOF adjustment exactly as Stata does
    N_pre = nobs(m_pre)
    df_m_pre = dof(m_pre)
    N_clust_pre = isnothing(cluster) ? N_pre : length(unique(d_donors[m_pre.esample, cluster]))
    
    # Compute df_a properly
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
    if !isempty(omitted_pre) && verbose
        @warn "Collinear pre-trends omitted: $(join(omitted_pre, ", "))"
    end
    k_valid = length(valid_pre_syms)
    if k_valid == 0
        return zeros(Float64, G, 0), Float64[], Int[]
    end
    
    # 2. Get preresid and map back to full dataset
    preresid_raw = residuals(m_pre)
    esample = m_pre.esample
    
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
    
    # Pre-allocate result matrix and thread-local buffers
    E_pre = zeros(Float64, G, k_valid)
    
    nt = Threads.nthreads()
    preweight_buffers = [zeros(Float64, n) for _ in 1:nt]
    
    # Pre-extract pre_dummy columns
    pre_dummy_cols = [Float64.(d[!, sym]) for sym in valid_pre_syms]
    
    # 3. For each valid pre-trend, compute influence weights and cluster scores
    # NOTE: Cannot easily thread this due to reg() calls - keep sequential
    for (j, sym) in enumerate(valid_pre_syms)
        pre_dummy = pre_dummy_cols[j]
        
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
        
        m_resid = quiet_reg(d_donors, f_resid, Vcov.simple(); weights=weights, save=:residuals, progress_bar=false)
        preweight_raw = residuals(m_resid)
        esample_resid = m_resid.esample
        
        # Map back to full dataset and multiply by wei
        preweight = preweight_buffers[1]  # Reuse buffer (sequential here)
        fill!(preweight, 0.0)
        
        for i in 1:n_donors
            if i <= length(esample_resid) && esample_resid[i] && i <= length(preweight_raw)
                val = preweight_raw[i]
                if !ismissing(val)
                    orig_row = d_donors[i, :__orig_row]
                    preweight[orig_row] = Float64(val) * wei[orig_row]
                end
            end
        end
        
        # Normalize
        norm_sum = 0.0
        for i in 1:n
            if donor_mask[i] && pre_dummy[i] == 1.0
                norm_sum += preweight[i]
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
function compute_cluster_scores_controls_mt!(
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
    touse_sym::Symbol = :__touse,
    verbose::Bool=false
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
    
    m_ctrl = quiet_reg(d_donors, f_ctrl; weights=weights, save=:residuals, progress_bar=false)
    
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
    if !isempty(omitted_ctrls)  && verbose
        @warn "Collinear controls omitted: $(join(omitted_ctrls, ", "))"
    end
    k_valid = length(valid_ctrl_syms)
    if k_valid == 0
        return zeros(Float64, G, 0), Float64[], Symbol[]
    end
    
    E_ctrl = zeros(Float64, G, k_valid)
    
    # Pre-extract control columns
    ctrl_cols = [Float64.(d[!, ctrl]) for ctrl in valid_ctrl_syms]
    
    # Pre-allocate buffer (sequential loop, so just need one)
    ctrlweight = zeros(Float64, n)
    
    # NOTE: Cannot easily thread due to reg() calls
    for (j, ctrl) in enumerate(valid_ctrl_syms)
        ctrl_vec = ctrl_cols[j]
        
        other_ctrl = filter(s -> s != ctrl, valid_ctrl_syms)
        
        if isempty(other_ctrl)
            f_resid = Term(ctrl) ~ fe_terms
        else
            f_resid = Term(ctrl) ~ sum(term.(other_ctrl)) + fe_terms
        end
        
        m_resid = quiet_reg(d_donors, f_resid, Vcov.simple(); weights=weights, save=:residuals, progress_bar=false)
        ctrlweight_raw = residuals(m_resid)
        esample_resid = m_resid.esample
        
        # Reset and fill buffer
        fill!(ctrlweight, 0.0)
        
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



function fit_bjs_mt(df::DataFrame;
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
    verbose::Bool = false
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
                                             autosample=autosample, verbose=verbose)
    
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
    W_tau, iters = compute_influence_weights_mt!(
        d, wtr_syms, controls, fe_syms, :D, weights;
        touse_sym = :__touse,
        tol = tol,
        maxiter = maxiter,
        verbose=verbose
    )
    
    #=== 9. Setup cluster structure ===#
    cluster_vec = isnothing(cluster) ? collect(1:n) : d[!, cluster]
    cluster_ids = unique(cluster_vec)
    G = length(cluster_ids)
    
    #=== 10. Compute cluster scores for tau ===#
    E_tau = compute_cluster_scores_tau_mt!(
        d, wtr_syms, W_tau, wei, cluster_vec, cluster_ids, g, t;
        treat_sym = :D,
        avgeffectsby = avgeffectsby
    )
    
    #=== 11. Compute cluster scores for pre-trends ===#
    E_pre, β_pre, valid_pre_idx = compute_cluster_scores_pretrends_mt!(
        d, pre_syms, controls, fe, wei, cluster_vec, cluster_ids, cluster, weights;
        y = y,
        touse_sym = :__touse,
        verbose=verbose
    )
    k_pre = length(β_pre)
    
    #=== 11b. Compute cluster scores for controls ===#
    E_ctrl, β_ctrl, valid_ctrl_syms = compute_cluster_scores_controls_mt!(
        d, controls, fe, wei, cluster_vec, cluster_ids, cluster, weights, 
        d[!, :__imput_resid];
        y = y,
        touse_sym = :__touse,
        verbose=verbose
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
Pre-computed FE structure for fast updates without allocation.
"""
struct FEStructure
    group_indices::Vector{Vector{Int}}  # indices for each group
    group_denoms::Vector{Float64}       # sum(wei) for donors in each group
    obs_to_group::Vector{Int}           # which group each obs belongs to
end

"""
Build FE structure once, reuse across iterations.
"""
function build_fe_structure(
    fe_vec::AbstractVector,
    donors::AbstractVector{Bool},
    touse::AbstractVector{Bool},
    wei::AbstractVector{<:Real}
    )
    n = length(fe_vec)
    
    # Map FE values to integer indices
    unique_fe = unique(fe_vec[touse])
    fe_to_int = Dict{eltype(fe_vec), Int}(v => i for (i, v) in enumerate(unique_fe))
    n_groups = length(unique_fe)
    
    # Build obs_to_group
    obs_to_group = zeros(Int, n)
    for i in 1:n
        if touse[i]
            obs_to_group[i] = fe_to_int[fe_vec[i]]
        end
    end
    
    # Build group_indices
    group_indices = [Int[] for _ in 1:n_groups]
    for i in 1:n
        if touse[i]
            push!(group_indices[obs_to_group[i]], i)
        end
    end
    
    # Build group_denoms (donor weights)
    group_denoms = zeros(Float64, n_groups)
    for i in 1:n
        if touse[i] && donors[i]
            group_denoms[obs_to_group[i]] += wei[i]
        end
    end
    
    return FEStructure(group_indices, group_denoms, obs_to_group)
end


"""
Update weights using pre-computed FE structure. Zero allocations.
"""
function update_weights_fe_fast!(
    W::AbstractMatrix{<:Real},
    fe_struct::FEStructure,
    donors::AbstractVector{Bool},
    wei::AbstractVector{<:Real}
    )
    n, k = size(W)
    n_groups = length(fe_struct.group_indices)
    
    # No threading - the overhead isn't worth it for typical k
    @inbounds for gi in 1:n_groups
        dg = fe_struct.group_denoms[gi]
        if dg <= 1e-14
            continue
        end
        
        all_idxs = fe_struct.group_indices[gi]
        
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
Update weights for a control variable. Zero allocations.
"""
function update_weights_control_fast!(
    W::AbstractMatrix{<:Real},
    x_dm::AbstractVector{<:Real},
    denom::Float64,
    donors::AbstractVector{Bool},
    touse::AbstractVector{Bool},
    wei::AbstractVector{<:Real}
    )
    if denom <= 1e-14
        return nothing
    end
    
    n, k = size(W)
    
    @inbounds for j in 1:k
        num = 0.0
        for i in 1:n
            if touse[i]
                num += wei[i] * W[i, j] * x_dm[i]
            end
        end
        β = num / denom
        
        for i in 1:n
            if donors[i] && touse[i]
                W[i, j] -= β * x_dm[i]
            end
        end
    end
    
    return nothing
end


"""
Compute influence weights via iterative orthogonalization.
All structures pre-computed, zero allocations in the iteration loop.
"""
function compute_influence_weights_mt!(
    df::DataFrame,
    wtr_syms::Vector{Symbol},
    controls::Vector{Symbol},
    fe_syms::Vector{Symbol},
    D_sym::Symbol,
    wei_sym::Union{Nothing,Symbol};
    touse_sym::Union{Nothing,Symbol}=nothing,
    tol::Float64=1e-6,
    maxiter::Int=1000,
    verbose::Bool=false
    )
    n = nrow(df)
    k = length(wtr_syms)
    
    D_vec = Int.(df[!, D_sym])
    donors = BitVector(D_vec .== 0)
    touse = touse_sym === nothing ? trues(n) : BitVector(df[!, touse_sym])
    wei = isnothing(wei_sym) ? ones(Float64, n) : Float64.(df[!, wei_sym])
    
    # Initialize W with raw wtr columns
    W = Matrix{Float64}(undef, n, k)
    for (j, sym) in enumerate(wtr_syms)
        col = df[!, sym]
        @inbounds for i in 1:n
            W[i, j] = coalesce(col[i], 0.0)
        end
    end
    
    # === PRE-COMPUTE EVERYTHING BEFORE THE LOOP ===
    
    # Pre-compute demeaned controls and denominators
    dm_controls = Vector{Vector{Float64}}(undef, length(controls))
    control_denoms = Vector{Float64}(undef, length(controls))
    
    for (ci, c) in enumerate(controls)
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
        
        x_dm = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            x_dm[i] = x[i] - μ
        end
        
        # Denom = sum(wei * x_dm^2 | D==0 & touse)
        denom = 0.0
        @inbounds for i in 1:n
            if donors[i] && touse[i]
                denom += wei[i] * x_dm[i]^2
            end
        end
        
        dm_controls[ci] = x_dm
        control_denoms[ci] = denom
    end
    
    # Pre-compute FE structures
    fe_structures = Vector{FEStructure}(undef, length(fe_syms))
    for (fi, fe_sym) in enumerate(fe_syms)
        fe_vec = df[!, fe_sym]
        fe_structures[fi] = build_fe_structure(fe_vec, donors, touse, wei)
    end
    
    # Pre-allocate W_old
    W_old = similar(W)
    
    # === ITERATION LOOP - NO ALLOCATIONS ===
    iter = 0
    while iter < maxiter
        iter += 1
        copyto!(W_old, W)
        
        # 1) Simple controls
        for ci in 1:length(controls)
            update_weights_control_fast!(W, dm_controls[ci], control_denoms[ci], 
                                         donors, touse, wei)
        end
        
        # 2) Fixed effects
        for fi in 1:length(fe_syms)
            update_weights_fe_fast!(W, fe_structures[fi], donors, wei)
        end
        
        # 3) Convergence check
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
Compute cluster scores for tau coefficients.
All indexing pre-computed, zero allocations in main loop.
"""
function compute_cluster_scores_tau_mt!(
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
    
    if isempty(avgeffectsby)
        avgeffectsby = [g_sym, t_sym]
    end
    
    treated = BitVector(d[!, treat_sym] .== 1)
    effect = Vector{Float64}(d[!, :effect])
    
    # === PRE-COMPUTE ALL INDICES ===
    
    # Cluster index for each observation
    cluster_idx = Vector{Int}(undef, n)
    cluster_idx_map = Dict{eltype(cluster_vec), Int}()
    for (idx, g_val) in enumerate(cluster_ids)
        cluster_idx_map[g_val] = idx
    end
    @inbounds for i in 1:n
        cluster_idx[i] = cluster_idx_map[cluster_vec[i]]
    end
    
    # Avgby index for each observation
    avgby_strings = Vector{String}(undef, n)
    for i in 1:n
        parts = String[]
        for s in avgeffectsby
            push!(parts, string(d[i, s]))
        end
        avgby_strings[i] = join(parts, "_")
    end
    
    unique_avgby = unique(avgby_strings)
    avgby_to_int = Dict{String, Int}(s => i for (i, s) in enumerate(unique_avgby))
    n_avgby = length(unique_avgby)
    
    avgby_idx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        avgby_idx[i] = avgby_to_int[avgby_strings[i]]
    end
    
    # (cluster, avgby) pair index
    pair_to_int = Dict{Tuple{Int,Int}, Int}()
    next_pair = 1
    for i in 1:n
        pair = (cluster_idx[i], avgby_idx[i])
        if !haskey(pair_to_int, pair)
            pair_to_int[pair] = next_pair
            next_pair += 1
        end
    end
    n_pairs = next_pair - 1
    
    pair_idx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        pair_idx[i] = pair_to_int[(cluster_idx[i], avgby_idx[i])]
    end
    
    # Resid0 for donors
    resid0 = zeros(Float64, n)
    @inbounds for i in 1:n
        if !treated[i]
            resid0[i] = effect[i]
        end
    end
    
    # Pre-extract wtr columns
    wtr_cols = Vector{Vector{Float64}}(undef, k)
    for j in 1:k
        wtr_cols[j] = Float64.(coalesce.(d[!, wtr_syms[j]], 0.0))
    end
    
    # Result matrix
    E = zeros(Float64, G, k)
    
    # === WORK BUFFERS (allocated once) ===
    cw_by_pair = zeros(Float64, n_pairs)
    clusterweight = zeros(Float64, n)
    smartdenom_by_avgby = zeros(Float64, n_avgby)
    smartweight = zeros(Float64, n)
    avgtau_by_avgby = zeros(Float64, n_avgby)
    resid = zeros(Float64, n)
    
    # === MAIN LOOP - NO ALLOCATIONS ===
    for j in 1:k
        # Reset buffers
        fill!(cw_by_pair, 0.0)
        fill!(clusterweight, 0.0)
        fill!(smartdenom_by_avgby, 0.0)
        fill!(smartweight, 0.0)
        fill!(avgtau_by_avgby, 0.0)
        fill!(resid, 0.0)
        
        wtrj = wtr_cols[j]
        
        # Compute cluster weights by pair
        @inbounds for i in 1:n
            if treated[i] && wtrj[i] != 0.0
                cw_by_pair[pair_idx[i]] += wei[i] * wtrj[i]
            end
        end
        
        # Map to observation-level
        @inbounds for i in 1:n
            if treated[i]
                clusterweight[i] = cw_by_pair[pair_idx[i]]
            end
        end
        
        # Compute smartdenom
        @inbounds for i in 1:n
            if treated[i] && wtrj[i] != 0.0
                smartdenom_by_avgby[avgby_idx[i]] += clusterweight[i] * wei[i] * wtrj[i]
            end
        end
        
        # Compute smart weights
        @inbounds for i in 1:n
            if treated[i] && wtrj[i] != 0.0
                sd = smartdenom_by_avgby[avgby_idx[i]]
                if sd > 1e-14
                    smartweight[i] = clusterweight[i] * wei[i] * wtrj[i] / sd
                end
            end
        end
        
        # Compute avgtau
        @inbounds for i in 1:n
            if treated[i]
                avgtau_by_avgby[avgby_idx[i]] += effect[i] * smartweight[i]
            end
        end
        
        # Compute residuals
        @inbounds for i in 1:n
            if treated[i]
                resid[i] = effect[i] - avgtau_by_avgby[avgby_idx[i]]
            else
                resid[i] = resid0[i]
            end
        end
        
        # Accumulate into E
        @inbounds for i in 1:n
            E[cluster_idx[i], j] += wei[i] * W[i, j] * resid[i]
        end
    end
    
    return E
end


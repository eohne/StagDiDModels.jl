
"""
    check_fe_imputable(model::FixedEffectModel, newdata::DataFrame, treated_mask::BitVector)

Check which treated observations have fixed effect levels that can be imputed 
from the first-stage model.

Returns a BitVector where `true` means the observation CAN be imputed.
"""
function check_fe_imputable(model::FixedEffectModel, newdata::DataFrame, treated_mask::BitVector)
    n = nrow(newdata)
    can_impute = trues(n)
    
    if !FixedEffectModels.has_fe(model)
        return can_impute
    end
    
    fe_df = model.fe
    
    if nrow(fe_df) == 0
        @warn "Model has no saved FE estimates"
        return can_impute
    end
    
    for k in model.fekeys
        fe_col_name = "fe_" * string(k)
        if !(fe_col_name in names(fe_df))
            continue
        end
        
        # FIX: Get FE levels that have NON-MISSING coefficients
        # This is what fe_predict actually uses (it does dropmissing)
        fe_levels_col = fe_df[!, k]
        fe_coefs_col = fe_df[!, fe_col_name]
        
        # Only include levels where coefficient is not missing
        available_fe = Set{eltype(fe_levels_col)}()
        for i in 1:length(fe_levels_col)
            level = fe_levels_col[i]
            coef = fe_coefs_col[i]
            if !ismissing(level) && !ismissing(coef)
                push!(available_fe, level)
            end
        end
        
        # Check each treated observation
        for i in 1:n
            if treated_mask[i]
                obs_fe_val = newdata[i, k]
                if ismissing(obs_fe_val) || !(obs_fe_val in available_fe)
                    can_impute[i] = false
                end
            end
        end
    end
    
    return can_impute
end


"""
    apply_autosample(d::DataFrame, model::FixedEffectModel, treated_mask::BitVector;
                     autosample::Bool=true, verbose::Bool=true)

Apply Stata-style autosample: drop treated observations where FE cannot be imputed.
"""
function apply_autosample(d::DataFrame, model::FixedEffectModel, treated_mask::BitVector;
                          autosample::Bool=true, verbose::Bool=true)
    
    can_impute = check_fe_imputable(model, d, treated_mask)
    cannot_impute_mask = .!can_impute .& treated_mask
    n_cannot_impute = sum(cannot_impute_mask)
    
    if n_cannot_impute == 0
        return trues(nrow(d)), 0
    end
    
    if !autosample
        error("Could not impute FE for $n_cannot_impute treated observations. " *
              "Use autosample=true to drop these observations automatically.")
    end
    
    if verbose
        n_total = sum(treated_mask)
        @warn "Autosample: dropping $n_cannot_impute / $n_total treated obs where FE coefficient is missing (singleton or not in donor sample)"
    end
    
    keep_mask = .!cannot_impute_mask
    return keep_mask, n_cannot_impute
end
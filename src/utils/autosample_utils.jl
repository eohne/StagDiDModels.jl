
"""
    check_fe_imputable(model::FixedEffectModel, newdata::DataFrame, treated_mask::BitVector)

Check which treated observations have fixed effect levels that can be imputed 
from the first-stage model (i.e., FE levels that appear in the donor sample).

Returns a BitVector where `true` means the observation CAN be imputed.

This mimics Stata's `did_imputation` autosample behavior.
"""
function check_fe_imputable(model::FixedEffectModel, newdata::DataFrame, treated_mask::BitVector)
    n = nrow(newdata)
    can_impute = trues(n)
    
    if !FixedEffectModels.has_fe(model)
        return can_impute
    end
    
    # Get the FE levels that were estimated in the first stage (donor sample)
    fe_df = model.fe
    
    for k in model.fekeys
        # Get the FE values that exist in the first-stage model
        fe_col_name = "fe_" * string(k)
        if !(fe_col_name in names(fe_df))
            continue
        end
        
        # Get unique FE levels from the model (dropping missings)
        available_fe = Set(skipmissing(fe_df[!, k]))
        
        # Check each observation
        for i in 1:n
            if treated_mask[i]  # Only check treated observations
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

Apply Stata-style autosample: identify and optionally drop treated observations 
where fixed effects cannot be imputed from the donor sample.

# Arguments
- `d::DataFrame`: The full dataset
- `model::FixedEffectModel`: First-stage model fitted on donors
- `treated_mask::BitVector`: Mask indicating which observations are treated
- `autosample::Bool`: If true, drop problematic observations. If false, error.
- `verbose::Bool`: If true, print warnings about dropped observations.

# Returns
- `keep_mask::BitVector`: Mask of observations to keep
- `n_dropped::Int`: Number of dropped observations
"""
function apply_autosample(d::DataFrame, model::FixedEffectModel, treated_mask::BitVector;
                          autosample::Bool=true, verbose::Bool=true)
    
    can_impute = check_fe_imputable(model, d, treated_mask)
    cannot_impute_mask = .!can_impute .& treated_mask
    n_cannot_impute = sum(cannot_impute_mask)
    
    if n_cannot_impute == 0
        # All observations can be imputed
        return trues(nrow(d)), 0
    end
    
    if !autosample
        error("Could not impute FE for $n_cannot_impute treated observations. " *
              "Use autosample=true to drop these observations automatically.")
    end
    
    if verbose
        @warn "Dropping $n_cannot_impute treated observations where FE could not be imputed from donor sample."
    end
    
    keep_mask = .!cannot_impute_mask
    return keep_mask, n_cannot_impute
end
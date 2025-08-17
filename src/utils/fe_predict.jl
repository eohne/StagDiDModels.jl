"""
    fe_predict(model::FixedEffectModel, newdata)

Fixed version of FixedEffectModels.predict() that handles:
1. FE-only models (no covariates besides fixed effects)
2. Missing fixed effect levels in newdata (sets their contribution to 0)

This is almost identical to the original predict function, with minimal changes.
"""
function fe_predict(model::FixedEffectModel, newdata)
    
    # Input validation
    Tables.istable(newdata) ||
        throw(ArgumentError("expected second argument to be a Table, got $(typeof(newdata))"))
    
    FixedEffectModels.has_cont_fe_interaction(model.formula) && 
        throw(ArgumentError("Interaction of fixed effect and continuous variable detected in formula; this is currently not supported in predict"))
    
    # Convert to columntable
    cdata = StatsModels.columntable(newdata)
    
    # === Handle FE-only models in missing_omit ===
    if length(model.coef) > 0
        # Standard case: has covariates
        cols, nonmissings = StatsModels.missing_omit(cdata, model.formula_schema.rhs)
        Xnew = StatsModels.modelmatrix(model.formula_schema, cols)
        if all(nonmissings)
            out = Xnew * model.coef
        else
            out = Vector{Union{Float64, Missing}}(missing, length(Tables.rows(cdata)))
            out[nonmissings] = Xnew * model.coef 
        end
    else
        # FE-only case: no covariates to check for missing, linear prediction = 0
        out = zeros(Float64, length(Tables.rows(cdata)))
        nonmissings = trues(length(Tables.rows(cdata)))
    end
    
    # Join FE estimates onto data and sum row-wise
    if FixedEffectModels.has_fe(model)
        nrow(model.fe) > 0 || throw(ArgumentError("Model has no estimated fixed effects. To store estimates of fixed effects, run reg with the option save = :fe"))
        df = DataFrame(newdata; copycols = false)
        
        # Join logic -> join FEs one by one! The combination of all fixed effects may not be available in the new data but indivial levels might.
        fes_joined = select(df, model.fekeys)
        for k in model.fekeys
            leftjoin!(fes_joined,dropmissing(unique(select(model.fe,string(k),"fe_"*string(k)))), on = k, makeunique = true, matchmissing = :equal)
        end
        
        # === Handle missing FE levels ===
        # Original code would have missing FE coefficients for unseen levels
        # Replace missing FE coefficients with 0 before summing
        fe_cols = names(select(fes_joined, Not(model.fekeys)))
        for col in fe_cols
            fes_joined[!, col] = coalesce.(fes_joined[!, col], 0.0)
        end
        
        # Sum FE contributions row-wise
        fes = combine(fes_joined, AsTable(Not(model.fekeys)) => sum)
        
        # Handle potential missing in FE keys
        if any(ismissing, Matrix(select(df, model.fekeys))) || any(ismissing, Matrix(fes))
            out = allowmissing(out)
        end
        
        # Add FE contributions to linear prediction
        out[nonmissings] .+= fes[nonmissings, 1]
        
        # Handle observations with missing covariates
        if any(.!nonmissings)
            out[.!nonmissings] .= missing
        end
    end
    
    return out
end
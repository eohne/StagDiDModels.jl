"""
    cumulative_effects(m::StatsAPI.StatisticalModel; 
                       pattern::Regex = r"τ::(-?\\d+)",
                       ref_period::Union{Nothing,Int} = nothing)

Compute cumulative treatment effects and their standard errors for plotting.

This function creates cumulative effects that sum "away from" the reference period:
- **Pre-treatment**: Cumulates backwards from reference (if ref=-1, cum(-3) = β_{-3} + β_{-2})  
- **Post-treatment**: Cumulates forward from period 0 (cum(2) = β_0 + β_1 + β_2)

Works with all DiD model types: `BJSModel` (Bootstrap version only), `GardnerModel`, `SunabModel`, `TWFEModel`.

# Arguments
- `m::StatisticalModel`: Event study model with τ coefficients (BJS, Gardner, Sun-Abraham, TWFE)
- `pattern::Regex`: Pattern to extract τ values (default matches "τ::-2", "τ::3", etc.)
- `ref_period::Union{Nothing,Int}`: Reference period that was excluded (default: auto-detect)

# Returns
NamedTuple with:
- `τ`: Vector of time periods  
- `cumulative`: Vector of cumulative effects
- `std_errors`: Vector of standard errors
- `conf_low`: Lower 95% confidence bounds
- `conf_high`: Upper 95% confidence bounds

# Examples
```julia
# BJS event study
m_bjs = fit_bjs_dynamic(df; y=:ret, id=:fund, t=:month, g=:g)
cum_bjs = cumulative_effects(m_bjs)

# Gardner event study  
m_gardner = fit_did2s_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
cum_gardner = cumulative_effects(m_gardner)

# Sun-Abraham event study
m_sunab = fit_sunab(df; y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
cum_sunab = cumulative_effects(m_sunab)

# TWFE event study
m_twfe = fit_twfe_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
cum_twfe = cumulative_effects(m_twfe)

# Plot any of them
using Plots
plot(cum_bjs.τ, cum_bjs.cumulative, 
     ribbon=1.96.*cum_bjs.std_errors,
     xlabel="Event Time", ylabel="Cumulative Effect",
     title="BJS Cumulative Effects")
```
"""
function cumulative_effects(m::StatsAPI.StatisticalModel; 
                            pattern::Regex = r"τ::(-?\d+)",
                            ref_period::Union{Nothing,Int} = nothing)
    
    nms = StatsAPI.coefnames(m)
    coefs = StatsAPI.coef(m)
    vcov_mat = StatsAPI.vcov(m)
    
    # Extract τ values and their positions
    τ_info = Vector{Tuple{Int, Int}}()  # (τ, coef_index)
    
    for (i, nm) in enumerate(nms)
        mm = match(pattern, nm)
        if mm !== nothing
            τ = parse(Int, mm.captures[end])
            push!(τ_info, (τ, i))
        end
    end
    
    isempty(τ_info) && error("No event-time coefficients matched by pattern.")
    
    # Sort by τ
    sort!(τ_info, by = x -> x[1])
    τ_values = [x[1] for x in τ_info]
    coef_indices = [x[2] for x in τ_info]
    
    # Auto-detect reference period if not provided
    if isnothing(ref_period)
        if length(τ_values) > 1
            full_range = minimum(τ_values):maximum(τ_values)
            missing_periods = setdiff(full_range, τ_values)
            if length(missing_periods) == 1
                ref_period = missing_periods[1]
            elseif -1 ∉ τ_values && -1 ∈ full_range
                ref_period = -1
            end
        end
    end
    
    # Build cumulative transformation matrix
    n_coefs = length(nms)
    n_periods = length(τ_values)
    A = zeros(n_periods, n_coefs)
    
    # Split periods around reference
    if !isnothing(ref_period)
        pre_mask = τ_values .< ref_period
        post_mask = τ_values .> ref_period
    else
        pre_mask = τ_values .< 0
        post_mask = τ_values .>= 0
    end
    
    # Pre-treatment cumulation (backwards towards reference)
    for (i, τ) in enumerate(τ_values)
        if pre_mask[i]
            # Sum from τ towards reference (exclusive)
            upper_bound = isnothing(ref_period) ? -1 : ref_period - 1
            for (j, τ_other) in enumerate(τ_values)
                if τ_other >= τ && τ_other <= upper_bound
                    A[i, coef_indices[j]] = 1.0
                end
            end
        end
    end
    
    # Post-treatment cumulation (forward from 0 or first post period)
    start_period = 0
    if !isnothing(ref_period) && ref_period >= 0
        start_period = ref_period + 1
    end
    
    for (i, τ) in enumerate(τ_values)
        if post_mask[i]
            # Sum from start_period to τ (inclusive)
            for (j, τ_other) in enumerate(τ_values)
                if τ_other >= start_period && τ_other <= τ
                    A[i, coef_indices[j]] = 1.0
                end
            end
        end
    end
    
    # Compute cumulative effects and standard errors
    cumulative = A * coefs
    cumulative_vcov = A * vcov_mat * A'
    std_errors = sqrt.(diag(cumulative_vcov))
    
    # 95% confidence intervals
    conf_low = cumulative .- 1.96 .* std_errors
    conf_high = cumulative .+ 1.96 .* std_errors
    
    return (
        τ = τ_values,
        cumulative = cumulative,
        std_errors = std_errors,
        conf_low = conf_low,
        conf_high = conf_high
    )
end
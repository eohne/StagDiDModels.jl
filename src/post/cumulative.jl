"""
    cumulative_effects(m::StatsAPI.StatisticalModel;
                       pattern::Regex = r"τ::(-?\\d+)",
                       ref_period::Union{Nothing,Int} = nothing,
                       level::Real = 0.95)

Compute cumulative treatment effects (e.g. cumulative abnormal returns) and their
confidence bounds for plotting. The reference period is normalized to zero, so the
cumulative path passes through 0 at the reference.

- **Post-treatment**: cumulates forward from period 0 — `cum(h) = β₀ + β₁ + … + β_h`.
- **Pre-treatment**: cumulates *back* toward the reference, so pre-period
  coefficients enter with a negative sign — e.g. with `ref = -1`,
  `cum(-3) = -(β₋₃ + β₋₂)`. This makes the curve a continuous cumulative-return
  path anchored at the reference; under parallel trends the pre-period values are ≈ 0.

Standard errors come from `A·Σ·Aᵀ` for each linear combination (full coefficient
covariance, including cross-covariances), and the confidence bounds use a
t-quantile with the model's residual degrees of freedom (matching the estimators'
own `confint`).

Works with all DiD model types: `BJSModel`, `GardnerModel`, `SunabModel`, `TWFEModel`.

# Arguments
- `m::StatisticalModel`: Event study model with τ coefficients (BJS, Gardner, Sun-Abraham, TWFE)
- `pattern::Regex`: Pattern to extract τ values (default matches "τ::-2", "τ::3", etc.)
- `ref_period::Union{Nothing,Int}`: Reference period that was excluded (default: auto-detect)
- `level::Real`: Confidence level for the bounds (default `0.95`)

# Returns
NamedTuple with:
- `τ`: Vector of event times
- `cumulative`: Vector of cumulative effects
- `std_errors`: Vector of standard errors
- `conf_low`: Lower confidence bounds at `level`
- `conf_high`: Upper confidence bounds at `level`

# Examples
```julia
# Cumulative abnormal returns from a BJS event study
m_bjs = fit_bjs_dynamic(df; y=:ret, id=:fund, t=:month, g=:g)
cum_bjs = cumulative_effects(m_bjs)               # 95% bounds
cum_90  = cumulative_effects(m_bjs; level = 0.90) # 90% bounds

# Works with any estimator
m_gardner = fit_gardner_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
m_sunab   = fit_sunab(df;          y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
m_twfe    = fit_twfe_dynamic(df;   y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
cum = cumulative_effects(m_gardner)
```
"""
function cumulative_effects(m::StatsAPI.StatisticalModel;
                            pattern::Regex = r"τ::(-?\d+)",
                            ref_period::Union{Nothing,Int} = nothing,
                            level::Real = 0.95)
    
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
                    A[i, coef_indices[j]] = -1.0
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

    # Confidence bounds at `level`, using a t-quantile with the model's residual
    # degrees of freedom (consistent with the estimators' own `confint`).
    α = 1 - level
    crit = quantile(TDist(StatsAPI.dof_residual(m)), 1 - α / 2)
    conf_low = cumulative .- crit .* std_errors
    conf_high = cumulative .+ crit .* std_errors
    
    return (
        τ = τ_values,
        cumulative = cumulative,
        std_errors = std_errors,
        conf_low = conf_low,
        conf_high = conf_high
    )
end
module StagDiDModelsMakieExt

using StagDiDModels
using Distributions: Normal, quantile
using Makie
using StatsAPI

# Import types we need to dispatch on
import StagDiDModels: BJSModel, GardnerModel, SunabModel, TWFEModel, 
                      cumulative_effects, fit_bjs_dynamic, fit_gardner_dynamic, 
                      fit_sunab, fit_twfe_dynamic

# Export plotting functions
export plot_event_study, plot_cumulative, plot_comparison

#=============================================================================
    Color Palette and Styling
=============================================================================#

const ESTIMATOR_COLORS = Dict(
    :bjs => :steelblue,
    :gardner => :coral,
    :sunab => :seagreen,
    :twfe => :purple
)

const ESTIMATOR_LABELS = Dict(
    :bjs => "BJS (Imputation)",
    :gardner => "Gardner (2-Stage)",
    :sunab => "Sun-Abraham",
    :twfe => "TWFE"
)

#=============================================================================
    Helper Functions
=============================================================================#

"""
Extract τ values and coefficients from a model for plotting.
Returns (τ_values, coefs, std_errors) for event-time coefficients only.
"""
function extract_event_study_data(m::StatsAPI.StatisticalModel; 
                                   pattern::Regex = r"τ::(-?\d+)")
    nms = StatsAPI.coefnames(m)
    coefs = StatsAPI.coef(m)
    ses = StatsAPI.stderror(m)
    
    τ_values = Int[]
    β_values = Float64[]
    se_values = Float64[]
    
    for (i, nm) in enumerate(nms)
        mm = match(pattern, nm)
        if mm !== nothing
            τ = parse(Int, mm.captures[1])
            push!(τ_values, τ)
            push!(β_values, coefs[i])
            push!(se_values, ses[i])
        end
    end
    
    # Sort by τ
    perm = sortperm(τ_values)
    return τ_values[perm], β_values[perm], se_values[perm]
end

#=============================================================================
    plot_event_study: Plot dynamic treatment effects for a single model
=============================================================================#

"""
    plot_event_study(m::StatisticalModel; kwargs...)

Plot event study coefficients with confidence intervals.

# Arguments
- `m::StatisticalModel`: A fitted DiD model (BJSModel, GardnerModel, SunabModel, or TWFEModel)

# Keyword Arguments
- `level::Float64 = 0.95`: Confidence level for intervals
- `title::String = "Event Study"`: Plot title
- `xlabel::String = "Event Time (τ)"`: X-axis label
- `ylabel::String = "Treatment Effect"`: Y-axis label
- `color = :steelblue`: Color for points and bands
- `ref_line::Bool = true`: Whether to show horizontal line at y=0
- `vline::Union{Nothing,Int} = -1`: Vertical line position (e.g., at τ=-1)
- `markersize::Int = 8`: Size of coefficient markers
- `linewidth::Float64 = 1.5`: Width of CI lines
- `figsize::Tuple = (800, 500)`: Figure size

# Returns
`Figure` object

# Example
```julia
using CairoMakie
m = fit_bjs_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)
fig = plot_event_study(m; title="BJS Event Study", color=:steelblue)
save("event_study.png", fig)
```
"""
function plot_event_study(m::StatsAPI.StatisticalModel;
                          level::Float64 = 0.95,
                          title::String = "Event Study",
                          xlabel::String = "Event Time (τ)",
                          ylabel::String = "Treatment Effect",
                          color = :steelblue,
                          ref_line::Bool = true,
                          vline::Union{Nothing,Int} = -1,
                          markersize::Int = 8,
                          linewidth::Float64 = 1.5,
                          figsize::Tuple = (800, 500))
    
    τ, β, se = extract_event_study_data(m)
    
    # Critical value for confidence intervals
    α = (1 - level) / 2
    z = quantile_normal(1 - α)
    
    ci_low = β .- z .* se
    ci_high = β .+ z .* se
    
    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1],
              title = title,
              xlabel = xlabel,
              ylabel = ylabel)
    
    # Reference line at y = 0
    if ref_line
        hlines!(ax, [0], color = :gray, linestyle = :dash, linewidth = 1)
    end
    
    # Vertical line (e.g., at reference period)
    if !isnothing(vline)
        vlines!(ax, [vline + 0.5], color = :gray, linestyle = :dot, linewidth = 1)
    end
    
    # Confidence interval band
    band!(ax, τ, ci_low, ci_high, color = (color, 0.2))
    
    # CI error bars
    rangebars!(ax, τ, ci_low, ci_high, color = color, linewidth = linewidth)
    
    # Point estimates
    scatter!(ax, τ, β, color = color, markersize = markersize)
    lines!(ax, τ, β, color = color, linewidth = linewidth)
    
    return fig
end

# Convenience: standard normal quantile
quantile_normal(p) = quantile(Normal(), p)

#=============================================================================
    plot_cumulative: Plot cumulative treatment effects
=============================================================================#

"""
    plot_cumulative(m::StatisticalModel; kwargs...)
    plot_cumulative(cum_result::NamedTuple; kwargs...)

Plot cumulative treatment effects.

# Arguments
- `m::StatisticalModel`: A fitted DiD model, OR
- `cum_result::NamedTuple`: Output from `cumulative_effects(m)`

# Keyword Arguments
- `level::Float64 = 0.95`: Confidence level (only used if passing a model)
- `title::String = "Cumulative Effects"`: Plot title
- `xlabel::String = "Event Time (τ)"`: X-axis label
- `ylabel::String = "Cumulative Treatment Effect"`: Y-axis label
- `color = :darkblue`: Color for points and bands
- `ref_line::Bool = true`: Whether to show horizontal line at y=0
- `figsize::Tuple = (800, 500)`: Figure size

# Returns
`Figure` object

# Example
```julia
m = fit_bjs_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, cluster=:fund)

# Method 1: Pass the model directly
fig = plot_cumulative(m)

# Method 2: Pass pre-computed cumulative effects
cum = cumulative_effects(m)
fig = plot_cumulative(cum)
```
"""
function plot_cumulative(m::StatsAPI.StatisticalModel; kwargs...)
    cum = cumulative_effects(m)
    return plot_cumulative(cum; kwargs...)
end

function plot_cumulative(cum::NamedTuple;
                         title::String = "Cumulative Effects",
                         xlabel::String = "Event Time (τ)",
                         ylabel::String = "Cumulative Treatment Effect",
                         color = :darkblue,
                         ref_line::Bool = true,
                         figsize::Tuple = (800, 500),
                         kwargs...)  # absorb unused kwargs like `level`
    
    τ = cum.τ
    cumulative = cum.cumulative
    ci_low = cum.conf_low
    ci_high = cum.conf_high
    
    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1],
              title = title,
              xlabel = xlabel,
              ylabel = ylabel)
    
    # Reference line at y = 0
    if ref_line
        hlines!(ax, [0], color = :gray, linestyle = :dash, linewidth = 1)
    end
    
    # Confidence interval band
    band!(ax, τ, ci_low, ci_high, color = (color, 0.2))
    
    # CI error bars
    rangebars!(ax, τ, ci_low, ci_high, color = color, linewidth = 1.5)
    
    # Point estimates
    scatter!(ax, τ, cumulative, color = color, markersize = 8)
    lines!(ax, τ, cumulative, color = color, linewidth = 1.5)
    
    return fig
end

#=============================================================================
    plot_comparison: Estimate multiple models and plot together
=============================================================================#

"""
    plot_comparison(df::DataFrame; y, id, t, g, 
                    models = [:bjs, :gardner, :sunab, :twfe], kwargs...)

Estimate multiple DiD models and plot their event study coefficients together.

# Arguments
- `df::DataFrame`: Panel dataset

# Required Keyword Arguments
- `y::Symbol`: Outcome variable
- `id::Symbol`: Unit identifier
- `t::Symbol`: Time period
- `g::Symbol`: Treatment cohort

# Optional Keyword Arguments
- `models::Vector{Symbol} = [:bjs, :gardner, :sunab, :twfe]`: Which estimators to include
- `controls::Vector{Symbol} = Symbol[]`: Control variables
- `cluster::Union{Nothing,Symbol} = nothing`: Clustering variable
- `weights::Union{Nothing,Symbol} = nothing`: Observation weights
- `ref_p::Int = -1`: Reference period for Gardner/TWFE
- `level::Float64 = 0.95`: Confidence level
- `title::String = "Event Study Comparison"`: Plot title
- `xlabel::String = "Event Time (τ)"`: X-axis label
- `ylabel::String = "Treatment Effect"`: Y-axis label
- `figsize::Tuple = (900, 600)`: Figure size
- `legend_position = :lt`: Legend position
- `offset::Float64 = 0.15`: Horizontal offset between estimators for visibility

# Returns
`NamedTuple` with:
- `figure`: The `Figure` object
- `models`: Dict of fitted model objects

# Example
```julia
result = plot_comparison(df; 
    y=:ret, id=:fund, t=:month, g=:g,
    models=[:bjs, :gardner, :sunab],
    cluster=:fund,
    title="Comparing DiD Estimators"
)
save("comparison.png", result.figure)

# Access individual models
coef(result.models[:bjs])
```
"""
function plot_comparison(df;
                         y::Symbol,
                         id::Symbol,
                         t::Symbol,
                         g::Symbol,
                         models::Vector{Symbol} = [:bjs, :gardner, :sunab, :twfe],
                         controls::Vector{Symbol} = Symbol[],
                         cluster::Union{Nothing,Symbol} = nothing,
                         weights::Union{Nothing,Symbol} = nothing,
                         ref_p::Int = -1,
                         level::Float64 = 0.95,
                         title::String = "Event Study Comparison",
                         xlabel::String = "Event Time (τ)",
                         ylabel::String = "Treatment Effect",
                         figsize::Tuple = (900, 600),
                         legend_position = :lt,
                         offset::Float64 = 0.15)
    
    # Fit all requested models
    fitted_models = Dict{Symbol, Any}()
    
    for m in models
        try
            if m == :bjs
                fitted_models[:bjs] = fit_bjs_dynamic(df; 
                    y=y, id=id, t=t, g=g, 
                    controls=controls, 
                    weights=weights,
                    cluster=cluster,
                    horizons=true, 
                    pretrends=false)
            elseif m == :gardner
                fitted_models[:gardner] = fit_gardner_dynamic(df;
                    y=y, id=id, t=t, g=g,
                    controls=controls,
                    weights=weights,
                    cluster=isnothing(cluster) ? id : cluster,
                    ref_p=ref_p)
            elseif m == :sunab
                fitted_models[:sunab] = fit_sunab(df;
                    y=y, id=id, t=t, g=g,
                    controls=controls,
                    weights=weights,
                    cluster=cluster,
                    ref_p=ref_p,
                    agg=:dynamic)
            elseif m == :twfe
                fitted_models[:twfe] = fit_twfe_dynamic(df;
                    y=y, id=id, t=t, g=g,
                    controls=controls,
                    weights=weights,
                    cluster=cluster,
                    ref_p=ref_p)
            else
                @warn "Unknown model type: $m"
            end
        catch e
            @warn "Failed to fit $m model: $e"
        end
    end
    
    if isempty(fitted_models)
        error("No models were successfully fitted")
    end
    
    # Create figure
    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1],
              title = title,
              xlabel = xlabel,
              ylabel = ylabel)
    
    # Reference line at y = 0
    hlines!(ax, [0], color = :gray, linestyle = :dash, linewidth = 1)
    
    # Vertical line at reference period
    vlines!(ax, [ref_p + 0.5], color = :gray, linestyle = :dot, linewidth = 1)
    
    # Critical value for confidence intervals
    α = (1 - level) / 2
    z = quantile_normal(1 - α)
    
    # Plot each model with offset
    n_models = length(fitted_models)
    model_keys = collect(keys(fitted_models))
    
    plot_elements = []
    
    for (idx, (model_name, model)) in enumerate(fitted_models)
        τ, β, se = extract_event_study_data(model)
        
        # Horizontal offset for visibility
        τ_offset = τ .+ (idx - (n_models + 1) / 2) * offset
        
        ci_low = β .- z .* se
        ci_high = β .+ z .* se
        
        color = get(ESTIMATOR_COLORS, model_name, :black)
        label = get(ESTIMATOR_LABELS, model_name, String(model_name))
        
        # CI error bars
        rangebars!(ax, τ_offset, ci_low, ci_high, color = color, linewidth = 1.5)
        
        # Point estimates
        sc = scatter!(ax, τ_offset, β, color = color, markersize = 8, label = label)
        lines!(ax, τ_offset, β, color = color, linewidth = 1.5)
        
        push!(plot_elements, (sc, label))
    end
    
    # Add legend
    Legend(fig[1, 2], 
           [el[1] for el in plot_elements],
           [el[2] for el in plot_elements],
           framevisible = false)
    
    return (figure = fig, models = fitted_models)
end

end # module
"""
    fit_twfe_dynamic(df; y, id, t, g;
                     controls = Symbol[],
                     cluster = nothing,
                     ref_p::Int = -1,
                     bin_leads::Union{Int,Nothing} = nothing,
                     bin_lags::Union{Int,Nothing} = nothing,
                     weights::Union{Nothing,Symbol} = nothing)

Two-way fixed effects **event study**:
```text
y_it ~ Σ_{τ≠ ref_p} 1{t-g=τ} + controls + FE(id) + FE(t)
````

## Conventions (IMPORTANT)

* **Never-treated must be encoded as `g == 0` or `missing`.** You do *not* need `Inf` or out-of-sample dates.
* `t` should be an **integer index** increasing by 1 (e.g., months 1,2,3,…).

# Arguments

* `df::DataFrame`: input panel.
* `y::Symbol`: outcome column.
* `id::Symbol`: unit identifier (fixed effects).
* `t::Symbol`: time index (fixed effects).
* `g::Symbol`: first-treatment time.

# Keyword Arguments

* `controls::Vector{Symbol}`: additional covariates (no FE here).
* `cluster`: one-way cluster key (e.g., `:id`, `:firm`), or `nothing`.
* `ref_p`: omitted pre-period τ (default `-1`).
* `bin_leads`: bin all τ ≤ `-bin_leads` into that single lead (optional).
* `bin_lags`: bin all τ ≥ `bin_lags` into that single lag (optional).
* `weights`: optional row-weights column.

# Returns

* `StatsAPI.StatisticalModel` with **per-τ coefficients** (excluding `ref_p`) and clustered vcov.

# Examples

```julia
m = fit_twfe_dynamic(df; y=:ret, id=:fund, t=:month, g=:g,
                     controls=[:size,:bm], cluster=:fund, ref_p=-1)
```

"""
function fit_twfe_dynamic(df::DataFrame; y::Symbol, id::Symbol, t::Symbol, g::Symbol,
                          controls::Vector{Symbol}=Symbol[],
                          cluster=nothing,
                          ref_p::Int=-1,
                          bin_leads::Union{Int,Nothing}=nothing,
                          bin_lags::Union{Int,Nothing}=nothing,
                          weights::Union{Nothing,Symbol}=nothing)

    d = copy(df)

    # Build event time and optional bins
    make_eventtime!(d; t=t, g=g, new=:ttt,ref_p=ref_p)
    nm = :ttt
    if !isnothing(bin_leads) || !isnothing(bin_lags)
        bin_eventtime!(d; ttt=:ttt, bin_leads=bin_leads, bin_lags=bin_lags, new=:ttt_b)
        nm = :ttt_b
    end

    # Get unique event times and sort them numerically
    unique_event_times = sort(unique(d[!, nm]))
    event_times_to_include = filter(x -> x != ref_p, unique_event_times)
    
    # Create dummy variables for each event time (excluding reference)
    dummy_names = Symbol[]
    for et in event_times_to_include
        dummy_name = Symbol("τ::$(et)")
        d[!, dummy_name] = Float64.(d[!, nm] .== et)
        push!(dummy_names, dummy_name)
    end
    
    # Build formula programmatically: ytilde ~ τ::-20 + τ::-19 + ... + τ::20
    dummy_terms = vcat(term(0), [term(name) for name in dummy_names])
    if isempty(controls)
        all_terms = dummy_terms
    else
        all_terms = vcat(dummy_terms, term.(controls))
    end
    rhs = reduce(+, all_terms)
    f = Term(y) ~ rhs + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)

    vc = build_cluster_vcov(cluster)
    m = reg(d, f, vc; weights=weights)
    TWFEModel(coef(m), vcov(m), coefnames(m), nobs(m), dof_residual(m), y, :dynamic,
                        sort(event_times_to_include),r2(m), adjr2(m), 
                        sum(d[:,:ttt].>=0), sum(d[:,:ttt].<0), isnothing(cluster) ? :none : cluster)
end



"""
    fit_twfe_static(df; y, id, t, g;
                    controls = Symbol[],
                    cluster = nothing,
                    weights::Union{Nothing,Symbol} = nothing,
                    control_type::Symbol = :notyet)

Two-way fixed effects **static ATT**:
```text
y_it ~ post_it + controls + FE(id) + FE(t),  post_it = 1{t ≥ g, g>0}
````

## Conventions

* **Never-treated must be `g == 0` or `missing`.**
* `t` is an integer index.

# Arguments / Keyword Arguments

* `df, y, id, t, g, controls, cluster` as usual.
* `weights`: a column of row-weights (forwarded to the fit).
* `control_type`: donor-rule for constructing `post` (kept for parity - not used by the static ATT formula itself).

# Returns

* `StatisticalModel` with a single `"_ATT"` coefficient and clustered vcov.

# Example

```julia
m = fit_twfe_static(df; y=:ret, id=:fund, t=:month, g=:g, controls=[:size], cluster=:fund)
```

"""
function fit_twfe_static(df::DataFrame; y::Symbol, id::Symbol, t::Symbol, g::Symbol,
    controls::Vector{Symbol}=Symbol[], cluster=nothing,
    weights::Union{Nothing,Symbol}=nothing,
    control_type::Symbol=:notyet)

    d = copy(df)
    make_post!(d; t=t, g=g, new=:_ATT)  # 1{t >= g, g>0}, with g==0/missing as never-treated
    
    # Build formula
    if isempty(controls)
        rhs = term(:_ATT) + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    else
        control_terms = sum(term.(controls))
        rhs = control_terms + term(:_ATT) + FixedEffectModels.fe(id) + FixedEffectModels.fe(t)
    end
    
    f = Term(y) ~ rhs
    vc = build_cluster_vcov(cluster)
    wv = isnothing(weights) ? nothing : Float64.(d[!, weights])
    
    m=reg(d, f, vc; weights=wv)
        TWFEModel(coef(m), vcov(m), coefnames(m), nobs(m), dof_residual(m), y, :static,
                        Int[],r2(m), adjr2(m), 
                        sum(d[:,:_ATT].>0), sum(d[:,:_ATT].==0),isnothing(cluster) ? :none : cluster)
end
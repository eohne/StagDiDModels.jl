"""
    build_project_wtr!(d, base_specs, project, wei; verbose=false)

Construct projection weight columns implementing Stata `did_imputation`'s
`project()` option (continuous treatment-effect heterogeneity).

Instead of reporting a weighted *average* of the imputed individual effects
`τ̂ = Y - Ŷ(0)`, `project()` reports the coefficients of a (weighted) OLS
regression of those individual effects on the `project` covariates:
`τ̂ ≈ β₀ + β₁ x₁ + …`. Each coefficient is obtained via Frisch–Waugh–Lovell:
residualize the relevant regressor against the others (among the treated cells
for that base column), normalize by `Σ wᵢ·residᵢ²`, and use the result as a new
`wtr` column. The existing point-estimate, influence-weight, and cluster-score
machinery then runs unchanged on these columns.

`base_specs` is a vector of `(basename, mask)` pairs — one entry for the static
case (`mask = D==1`) or one per horizon (`mask = D==1 & K==h`). For each base it
emits an intercept component (`<basename>_cons`) and one slope per covariate
(`<basename>_<var>`). Collinear / degenerate components are dropped with a
warning, mirroring Stata. Returns `(wtr_syms, wtr_names)`.
"""
function build_project_wtr!(d::DataFrame,
                            base_specs::Vector{<:Tuple{String,<:AbstractVector{Bool}}},
                            project::Vector{Symbol},
                            wei::Vector{Float64};
                            verbose::Bool=false)
    n = nrow(d)
    P = [Float64.(d[!, p]) for p in project]
    nP = length(project)
    wtr_syms = Symbol[]
    wtr_names = String[]
    counter = Ref(0)

    # Weighted-LS residual of y on X (over the local subsample), normalized by
    # Σ w·resid², written into a full-length column on the rows in S.
    function add_col!(name::String, S::Vector{Int}, X::AbstractMatrix{Float64},
                      y::Vector{Float64}, wS::Vector{Float64})
        if size(X, 2) == 0
            resid = copy(y)
        else
            sw = sqrt.(wS)
            β = (X .* sw) \ (y .* sw)
            resid = y .- X * β
        end
        denom = sum(wS .* resid .^ 2)
        if denom < 1e-6
            verbose && @warn "project: dropping $name (collinearity / insufficient variation)"
            return
        end
        counter[] += 1
        sym = Symbol("__proj$(counter[])")
        col = zeros(Float64, n)
        @inbounds for (li, gi) in enumerate(S)
            col[gi] = resid[li] / denom
        end
        d[!, sym] = col
        push!(wtr_syms, sym)
        push!(wtr_names, name)
    end

    for (basename, mask) in base_specs
        S = findall(mask)
        isempty(S) && continue
        wS = wei[S]
        # Intercept component: residual of 1 on the covariates (no intercept).
        Xc = nP == 0 ? Matrix{Float64}(undef, length(S), 0) :
                       hcat([P[j][S] for j in 1:nP]...)
        add_col!(basename * "_cons", S, Xc, ones(length(S)), wS)
        # Slope components: residual of each covariate on intercept + the others.
        for k in 1:nP
            yk = P[k][S]
            others = [P[j][S] for j in 1:nP if j != k]
            Xk = hcat(ones(length(S)), others...)
            add_col!(basename * "_" * String(project[k]), S, Xk, yk, wS)
        end
    end

    return wtr_syms, wtr_names
end


"""
    build_hetby_wtr!(d, base_specs, hetby, wei, D)

Construct subgroup weight columns implementing Stata `did_imputation`'s
`hetby()` option (discrete treatment-effect heterogeneity).

Each base treatment-weight column (the static ATT, or one per horizon) is split
into one column per distinct value of the `hetby` variable observed among the
treated, and each split column is normalized so its weighted sum over the
treated cells equals 1. The estimator then reports a separate (average) effect
for every (base × subgroup) cell. Returns `(wtr_syms, wtr_names, vals)`.
"""
function build_hetby_wtr!(d::DataFrame,
                          base_specs::Vector{<:Tuple{String,<:AbstractVector{Bool}}},
                          hetby::Symbol,
                          wei::Vector{Float64},
                          D::AbstractVector{<:Integer})
    n = nrow(d)
    hv = d[!, hetby]
    vals = sort(unique(hv[i] for i in 1:n if D[i] == 1 && !ismissing(hv[i])))
    length(vals) > 30 && error("hetby: variable $hetby takes more than 30 values among the treated.")
    isempty(vals) && error("hetby: variable $hetby is always missing among the treated.")

    wtr_syms = Symbol[]
    wtr_names = String[]
    counter = Ref(0)
    for (basename, mask) in base_specs
        for val in vals
            col = zeros(Float64, n)
            s = 0.0
            @inbounds for i in 1:n
                if mask[i] && !ismissing(hv[i]) && hv[i] == val
                    col[i] = 1.0
                    s += wei[i]
                end
            end
            s <= 0 && continue
            col ./= s
            counter[] += 1
            sym = Symbol("__het$(counter[])")
            d[!, sym] = col
            push!(wtr_syms, sym)
            push!(wtr_names, "$(basename)_$(val)")
        end
    end
    return wtr_syms, wtr_names, vals
end


"""
    minn_keep_mask(d, wtr_syms, wei, D, minn) -> BitVector

For each treatment-weight column, compute the effective sample size behind the
coefficient as `1 / HHI`, where `HHI = Σ (wᵢ·vᵢ / Σ wⱼ·|vⱼ|)²` over the treated
cells (matching Stata `did_imputation`'s `minn` rule). Returns a mask that is
`false` for columns whose effective sample size is below `minn` (or that carry
no weight at all), so they can be suppressed.
"""
function minn_keep_mask(d::DataFrame, wtr_syms::Vector{Symbol},
                        wei::Vector{Float64}, D::AbstractVector{<:Integer}, minn::Real)
    n = nrow(d)
    k = length(wtr_syms)
    keep = trues(k)
    for j in 1:k
        v = Float64.(coalesce.(d[!, wtr_syms[j]], 0.0))
        s = 0.0
        @inbounds for i in 1:n
            D[i] == 1 && (s += wei[i] * abs(v[i]))
        end
        if s == 0
            keep[j] = false
            continue
        end
        hhi = 0.0
        @inbounds for i in 1:n
            D[i] == 1 && (hhi += (v[i] * wei[i] / s)^2)
        end
        hhi > 1 / minn && (keep[j] = false)
    end
    return keep
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
- `project::Vector{Symbol} = Symbol[]`: **Continuous** treatment-effect
    heterogeneity (Stata `project()`). Use this when you care how the effect
    *varies with* one or more (typically continuous) covariates rather than its
    average. Instead of a weighted mean of the imputed individual effects
    `τ̂ = Y - Ŷ(0)`, it reports the coefficients of a (weighted) OLS projection
    `τ̂ ≈ β₀ + β₁x₁ + …`: an intercept (`_cons`) plus one slope per covariate. A
    positive, significant slope means the effect grows with that covariate. Works
    with `horizons` (applied per horizon — see *Returns*). Cannot be combined with
    `hetby`.
- `hetby::Union{Nothing,Symbol} = nothing`: **Discrete** treatment-effect
    heterogeneity (Stata `hetby()`). Use this to get a separate average effect for
    each subgroup defined by a categorical variable (≤30 distinct values among the
    treated). Works with `horizons` (applied per horizon — see *Returns*). Cannot
    be combined with `project`.
- `minn::Real = 0`: Minimum effective sample size per coefficient (Stata `minn`, but
    defaulting to `0` = off for backward compatibility). When `> 0`, coefficients whose
    effective sample size `1/HHI` falls below `minn` are suppressed with a warning.
    `minn = 30` reproduces Stata's default behavior.
- `avgeffectsby`: Grouping variables for averaging treatment effects. Default is `[g, t]` (cohort x time), matching Stata's default behavior.
- `control_type::Symbol = :notyet`: Donor pool definition.
- `tol::Float64 = 1e-6`: Convergence tolerance for influence weight iteration.
- `maxiter::Int = 1000`: Maximum iterations for influence weight computation.
- `autosample::Bool = true`: If true (default), drop treated observations where FE cannot
                             be imputed from donor sample. If false, error.
- `verbose::Bool = false`: If warnings should be printed. (e.g. dropping of singletons, collinear pretrends or controls)
- `multithreaded::Bool = true`: whether to multithread or not.
- `compute_ses::Bool = true`: when `false`, skip the (expensive) variance
    estimation — the iterative influence-weight orthogonalization and per-cluster
    score accumulation — and return only the point estimates. The covariance matrix
    is filled with `NaN`, so `coef`/`coefnames` are valid while `vcov`/`stderror`/
    `pvalue`/`confint` return `NaN`. Intended for tight loops (e.g. bootstrap /
    placebo simulations) that need coefficients only; point estimates are identical
    to a full fit.

# Returns
A `BJSModel`. Its coefficient vector (`coef(m)` / `coefnames(m)`) is laid out in
three blocks, in this order:

1. **Pre-trends** — one coefficient per requested negative horizon, named
   `τ::-1`, `τ::-2`, … (controlled by `pretrends`). These are a falsification
   check on the imputation model and are **always the standard per-horizon
   estimates**; they are *not* split or reshaped by `project` or `hetby`.
2. **Treatment effects** — the post-period block. Its shape depends on the
   options (`h` denotes a post-treatment horizon):

   | mode             | static (`horizons=nothing`) | dynamic (`horizons` set)              |
   |------------------|-----------------------------|---------------------------------------|
   | plain            | `_ATT`                      | `τ::0`, `τ::1`, …                      |
   | `project=[:x]`   | `τ_cons`, `τ_x`             | `τ0_cons`, `τ0_x`, `τ1_cons`, `τ1_x`, …|
   | `hetby=:grp`     | one `τ_<val>` per subgroup  | one `τ<h>_<val>` per horizon × subgroup|

3. **Controls** — one coefficient per surviving control (only if `controls` given).

So `project`/`hetby` reshape **only** the post-period block; the pre-trend
coefficients are identical to a plain fit. Everything is accessible through the
standard StatsAPI interface: `coef`, `coefnames`, `vcov`, `stderror`, `confint`,
`coeftable`.

# Notes
- Never-treated units: code as `missing` or `0` in `g`
- Time variable must be integer-coded
- When `autosample=true`, observations where FE cannot be imputed are dropped with a warning.
- `project` and `hetby` are mutually exclusive (passing both errors).
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
    project::Vector{Symbol} = Symbol[],
    hetby::Union{Nothing,Symbol} = nothing,
    minn::Real = 0,
    avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true,
    verbose::Bool = false,
    multithreaded::Bool = true,   # selects threaded orthogonalization (see note)
    compute_ses::Bool = true
    )

    # `multithreaded` selects whether the influence-weight orthogonalization runs
    # multithreaded. It parallelizes over disjoint work (FE groups / W columns),
    # so results are identical to the serial path — only the wall-clock changes,
    # and only when Julia is started with multiple threads. The cluster-score and
    # `reg`-driven steps stay serial. (The previous `fit_bjs_st` path was a
    # redundant, slower reimplementation and was removed; both paths now share
    # `fit_bjs_mt`.)
    return fit_bjs_mt(df;
        y=y, id=id, t=t, g=g,
        controls=controls, fe=fe, weights=weights,
        cluster=cluster, horizons=horizons, pretrends=pretrends,
        project=project, hetby=hetby, minn=minn,
        avgeffectsby=avgeffectsby, control_type=control_type,
        tol=tol, maxiter=maxiter, autosample=autosample, verbose=verbose,
        threaded=multithreaded, compute_ses=compute_ses
    )
end

"""
    fit_bjs_static(df::DataFrame; y, id, t, g, controls=Symbol[], fe=(id,t),
                   weights=nothing, cluster=nothing, control_type=:notyet,
                   tol=1e-6, maxiter=1000, autosample=true, verbose=false, multithreaded=true)

Static ATT estimator using Borusyak, Jaravel, and Spiess (2023) imputation.
Wrapper for `fit_bjs(...; horizons=nothing, pretrends=nothing)`.

Returns a single `_ATT` coefficient by default. Heterogeneity options are
supported and reshape that single effect:
- `project::Vector{Symbol}`: continuous heterogeneity → an intercept `τ_cons`
  plus a slope `τ_<var>` per covariate (OLS projection of the individual effects).
- `hetby::Symbol`: discrete heterogeneity → one `τ_<value>` per subgroup (≤30).

`project` and `hetby` cannot be combined. See [`fit_bjs`](@ref) for the full
description of these options, `minn`, and the returned coefficient layout.
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
    project::Vector{Symbol} = Symbol[],
    hetby::Union{Nothing,Symbol} = nothing,
    minn::Real = 0,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true,
    verbose::Bool = false,
    multithreaded::Bool = true,
    compute_ses::Bool = true
    )

    return fit_bjs(df; y=y, id=id, t=t, g=g,
                   controls=controls, fe=fe, weights=weights,
                   cluster=cluster, horizons=nothing, pretrends=nothing,
                   project=project, hetby=hetby, minn=minn,
                   control_type=control_type, tol=tol, maxiter=maxiter,
                   autosample=autosample,verbose=verbose,
                   multithreaded = multithreaded, compute_ses=compute_ses)
end


"""
    fit_bjs_dynamic(df::DataFrame; y, id, t, g, controls=Symbol[], fe=(id,t),
                    weights=nothing, cluster=nothing, horizons=true,
                    pretrends=true, avgeffectsby=nothing,
                    control_type=:notyet, tol=1e-6, maxiter=1000, autosample=true,verbose=false, multithreaded=true)

Dynamic event-study estimator using Borusyak, Jaravel, and Spiess (2023) imputation.
Wrapper for `fit_bjs` with `horizons=true` and `pretrends=true` by default.

Returns per-horizon post-treatment effects (`τ::0`, `τ::1`, …) plus pre-trend
coefficients (`τ::-1`, `τ::-2`, …). `project` and `hetby` work here too and are
applied **per horizon**:
- `project::Vector{Symbol}` → `τ0_cons`, `τ0_<var>`, `τ1_cons`, `τ1_<var>`, …
- `hetby::Symbol` → one `τ<h>_<value>` per horizon × subgroup.

In both cases the **pre-trend coefficients are returned unchanged** (one per
negative horizon) — heterogeneity only reshapes the post-period effects. The two
options cannot be combined. See [`fit_bjs`](@ref) for full details.
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
    project::Vector{Symbol} = Symbol[],
    hetby::Union{Nothing,Symbol} = nothing,
    minn::Real = 0,
    avgeffectsby::Union{Nothing,Vector{Symbol}} = nothing,
    control_type::Symbol = :notyet,
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    autosample::Bool = true,
    verbose::Bool = false,
    multithreaded::Bool = true,
    compute_ses::Bool = true)

    return fit_bjs(df; y=y, id=id, t=t, g=g,
                   controls=controls, fe=fe, weights=weights,
                   cluster=cluster, horizons=horizons, pretrends=pretrends,
                   project=project, hetby=hetby, minn=minn,
                   avgeffectsby=avgeffectsby, control_type=control_type,
                   tol=tol, maxiter=maxiter, autosample=autosample,verbose=verbose,
                   multithreaded = multithreaded, compute_ses=compute_ses)
end
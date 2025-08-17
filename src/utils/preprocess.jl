# Return a sum of control Terms, or `nothing` if none
_controls_term(controls::Vector{Symbol}) =
    isempty(controls) ? nothing : sum(term.(controls))

# Sum of FE terms from a tuple of Symbols (handles 0/1/2+ symbols)
fe_terms(fe_syms::Tuple{Vararg{Symbol}}) =
    isempty(fe_syms) ? nothing : reduce(+, (fe(s) for s in fe_syms))


function add_terms(xs...)
    terms = filter(!isnothing, collect(xs))
    if isempty(terms)
        return ConstantTerm(0)  # Return a proper term instead of 0
    elseif length(terms) == 1
        return terms[1]
    else
        return reduce(+, terms)
    end
end

# Create event-time ttt = t - g
function make_eventtime!(df::DataFrame; t::Symbol, g::Symbol, new::Symbol=:ttt,ref_p=-1)
    df[!, new] = df[!, t] .- df[!, g]
    df[(isequal.(df[:,g],0) .| ismissing.(df[:,g])),new] .=ref_p
    sort!(df, :ttt)
    return df
end

# Create post indicator: 1 if treated and t >= g
function make_post!(df::DataFrame; t::Symbol, g::Symbol, treat::Union{Symbol,Nothing}=nothing, new::Symbol=:post)
    treated = isnothing(treat) ? .!ismissing.(df[!, g]) : df[!, treat] .== 1
    df[!, new] = ifelse.(treated .& (df[!, t] .>= df[!, g]), 1, 0)
    replace!(df[!, new], missing=>0)
    return df
end

# Clamp event-time into lead/lag bins
function bin_eventtime!(df::DataFrame; ttt::Symbol=:ttt, bin_leads::Union{Int,Nothing}=nothing, bin_lags::Union{Int,Nothing}=nothing, new::Symbol=:ttt_b)
    v = copy(df[!, ttt])
    if !isnothing(bin_leads); v = map(x -> x < -bin_leads ? -bin_leads : x, v) end
    if !isnothing(bin_lags);  v = map(x -> x >  bin_lags ?  bin_lags : x, v) end
    df[!, new] = v
    return df
end

# Make ordered categorical with clean levels
function factorize!(df::DataFrame, s::Symbol; stringy::Bool=true)
    a = stringy ? string.(df[!, s]) : df[!, s]
    df[!, s] = CategoricalArray(a; ordered=true)
    levels!(df[!, s], sort!(unique(levels(df[!, s]))))
    return df
end

# Simple support diagnostics per event-time
function event_support(df::DataFrame; ttt::Symbol=:ttt, treated::Union{Symbol,Nothing}=nothing)
    if isnothing(treated)
        combine(groupby(df, ttt), nrow => :N)
    else
        combine(groupby(df, [ttt, treated]), nrow => :N)
    end
end

# build sparse one-hot dummies for a vector (Int/String/Categorical ok)
function _sparse_dummies(vec)
    m = length(vec)
    lv = collect(unique(vec))
    p = length(lv)
    row = Vector{Int}(undef, m)
    col = Vector{Int}(undef, m)
    val = ones(Float64, m)
    map = Dict{eltype(lv),Int}(lv[i] => i for i in 1:p)
    @inbounds for i in 1:m
        row[i] = i
        col[i] = map[vec[i]]
    end
    return sparse(row, col, val, m, p), lv
end

"""
    sparse_rhs_fe(df; controls=[], fe::Tuple)

Return a sparse model matrix `X1` that concatenates:
- continuous RHS controls (as sparse),
- one-hot FE dummies for each symbol in `fe` (e.g., `(:id, :t)`).

No intercept is included (matches `~ 0 + ...` in fixest).
"""
function sparse_rhs_fe(df::DataFrame; controls::Vector{Symbol}=Symbol[], fe::Tuple)
    m = nrow(df)
    blocks = Vector{SparseMatrixCSC{Float64,Int}}()

    # controls (dense -> sparse)
    if !isempty(controls)
        Xc = reduce(hcat, (Float64.(df[!, s]) for s in controls))
        push!(blocks, sparse(Xc))
    end

    # FE dummies
    for s in fe
        Xf, _ = _sparse_dummies(df[!, s])
        push!(blocks, Xf)
    end

    return hcat(blocks...)
end

# Which observations are eligible donors at time t?
function donor_mask(df::DataFrame; t::Symbol, g::Symbol, control_type::Symbol=:notyet)
    if control_type === :never
        # Never-treated: g <= 0 (includes g == 0 and originally missing g)
        return df[!, g] .<= 0
    elseif control_type === :notyet
        # Not-yet-treated OR never-treated
        return (df[!, g] .<= 0) .| (df[!, t] .< df[!, g])
    else
        error("control_type must be :notyet or :never")
    end
end
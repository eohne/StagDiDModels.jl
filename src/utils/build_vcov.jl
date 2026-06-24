# Normalize any cluster specification into the uniform stored representation:
# a `Vector{Symbol}` of clustering variables. An empty vector means "no
# clustering" (i.e. heteroskedasticity-robust SEs). `nothing`, `:none`, and the
# internal `:_no_cluster` sentinel all map to the empty (robust) case. This is
# what every model struct stores in its `cluster` field, so BJS/Gardner simply
# never hold more than one entry while TWFE/Sunab may hold several (multiway).
_clustervec(::Nothing) = Symbol[]
_clustervec(c::Symbol) = (c === :none || c === :_no_cluster) ? Symbol[] : [c]
_clustervec(c::Tuple) = collect(Symbol, c)
_clustervec(c::AbstractVector) = Vector{Symbol}(map(Symbol, c))
_clustervec(c) = [Symbol(c)]

# Compact label for the cluster variables, used in the `show` summary block.
_cluster_show(v::AbstractVector{Symbol}) = isempty(v) ? "none" : join(string.(v), ", ")

# Build CovarianceMatrices vcov object from cluster specification
function build_cluster_vcov(cluster)
    if cluster === nothing
        return Vcov.robust()
    elseif cluster isa Tuple
        # Multi-way clustering: pass multiple symbols as separate arguments
        syms = map(Symbol, cluster)
        return Vcov.cluster(syms...)  # Just pass symbols directly
    elseif cluster isa Vector
        # Vector of clustering variables: pass as separate arguments
        syms = map(Symbol, cluster)
        return Vcov.cluster(syms...)  # Just pass symbols directly
    else
        # Single clustering variable
        return Vcov.cluster(Symbol(cluster))  # Just pass the symbol directly
    end
end
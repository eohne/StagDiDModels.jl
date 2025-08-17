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
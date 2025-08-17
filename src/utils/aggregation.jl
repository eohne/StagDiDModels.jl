# Equal-weights A over cohorts for each event-time τ
function A_equal_weights(gt_pairs::Vector{Tuple{Int,Int}})
    τs = sort!(unique(last.(gt_pairs)))
    A = zeros(length(τs), length(gt_pairs))
    for (j, (_g, τ)) in enumerate(gt_pairs)
        i = findfirst(==(τ), τs)
        denom = count(x -> last(x) == τ, gt_pairs)
        A[i, j] = 1/denom
    end
    names = "ttt::" .* string.(τs)
    return A, names
end

# Observation-weighted A using per-(g,τ) counts w
function A_obs_weights(gt_pairs::Vector{Tuple{Int,Int}}, w::AbstractVector)
    τs = sort!(unique(last.(gt_pairs)))
    A = zeros(length(τs), length(gt_pairs))
    for (j, (_g, τ)) in enumerate(gt_pairs)
        i = findfirst(==(τ), τs)
        denom = sum(w[k] for k in eachindex(gt_pairs) if last(gt_pairs[k]) == τ)
        A[i, j] = w[j] / denom
    end
    names = "ttt::" .* string.(τs)
    return A, names
end

module ExperimentUtils

using GBTQuantum
using Statistics

export enumerate_states, weighted_mean, weighted_r2, ordinary_r2,
       predict_all, shift_tree_leaves, scale_tree, tree_leaf_count,
       exact_probabilities, draw_categorical_indices, compress_indices,
       write_namedtuple_csv, finite_mean

"""Enumerate the full N-spin z-basis using ±1 Int8 spins."""
function enumerate_states(N::Int)
    d = 1 << N
    states = Matrix{Int8}(undef, d, N)
    @inbounds for s in 0:(d-1), i in 1:N
        states[s+1, i] = ((s >> (i-1)) & 1) == 1 ? Int8(1) : Int8(-1)
    end
    return states
end

function weighted_mean(y, w)
    W = sum(w)
    W <= 0 && return NaN
    return sum(Float64(w[i]) * Float64(y[i]) for i in eachindex(y, w)) / W
end

function weighted_r2(ytrue, ypred, w)
    mu = weighted_mean(ytrue, w)
    ss_res = sum(Float64(w[i]) * (Float64(ytrue[i]) - Float64(ypred[i]))^2
                 for i in eachindex(ytrue, ypred, w))
    ss_tot = sum(Float64(w[i]) * (Float64(ytrue[i]) - mu)^2
                 for i in eachindex(ytrue, w))
    ss_tot <= eps(Float64) && return NaN
    return 1.0 - ss_res / ss_tot
end

function ordinary_r2(ytrue, ypred)
    mu = mean(ytrue)
    ss_res = sum((ytrue .- ypred).^2)
    ss_tot = sum((ytrue .- mu).^2)
    ss_tot <= eps(Float64) && return NaN
    return 1.0 - ss_res / ss_tot
end

function predict_all(tree::RegressionTree, X::AbstractMatrix{<:Real})
    y = Vector{Float64}(undef, size(X, 1))
    @inbounds for i in axes(X, 1)
        y[i] = predict(tree, @view(X[i, :]))
    end
    return y
end

function shift_tree_leaves(tree::RegressionTree, shift::Real)
    nodes = copy(tree.nodes)
    @inbounds for i in eachindex(nodes)
        n = nodes[i]
        if n.isleaf
            nodes[i] = Node(n.feature, n.value - Float64(shift), n.left, n.right, true)
        end
    end
    return RegressionTree(nodes)
end

function scale_tree(tree::RegressionTree, scale::Real)
    a = Float64(scale)
    nodes = [n.isleaf ? Node(n.feature, a*n.value, n.left, n.right, true) : n
             for n in tree.nodes]
    return RegressionTree(nodes)
end

tree_leaf_count(tree::RegressionTree) = count(n -> n.isleaf, tree.nodes)

function exact_probabilities(model::LogGBState, states::Matrix{Int8})
    logweights = [2.0 * logamplitude(model, @view(states[s, :])) for s in axes(states, 1)]
    m = maximum(logweights)
    w = exp.(logweights .- m)
    return w / sum(w)
end

function draw_categorical_indices(rng::AbstractRNG, p::AbstractVector{<:Real}, M::Int)
    cdf = cumsum(Float64.(p))
    cdf[end] = 1.0
    return [searchsortedfirst(cdf, rand(rng)) for _ in 1:M]
end

function compress_indices(indices::Vector{Int})
    d = Dict{Int,Int}()
    for idx in indices
        d[idx] = get(d, idx, 0) + 1
    end
    unique_idx = sort!(collect(keys(d)))
    counts = Float64[d[idx] for idx in unique_idx]
    return unique_idx, counts
end

function write_namedtuple_csv(path, rows)
    isempty(rows) && error("No rows generated; refusing to write an empty CSV.")
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(names), ","))
        for row in rows
            println(io, join([getproperty(row, n) for n in names], ","))
        end
    end
end

function finite_mean(values)
    x = Float64[v for v in values if isfinite(v)]
    return isempty(x) ? NaN : mean(x)
end

end # module ExperimentUtils

struct Node
    feature::Int32
    value::Float64
    left::Int32
    right::Int32
    isleaf::Bool
end

struct RegressionTree
    nodes::Vector{Node}
end

mutable struct GBMachine
    bias::Float64
    trees::Vector{RegressionTree}
end

GBMachine(bias::Real=0.0) = GBMachine(Float64(bias), RegressionTree[])

@inline function predict(t::RegressionTree, x::AbstractVector{<:Real})
    i = Int32(1)
    @inbounds while true
        n = t.nodes[i]
        n.isleaf && return n.value
        i = x[n.feature] < 0 ? n.left : n.right
    end
end

@inline function predict(m::GBMachine, x::AbstractVector{<:Real})
    y = m.bias
    @inbounds for t in m.trees
        y += predict(t, x)
    end
    return y
end

# Weighted least-squares tree for raw ±1 spins.
# Because the only meaningful split is x[f] < 0, threshold search is eliminated.
function grow_tree(X::AbstractMatrix{<:Real},
                   y::AbstractVector{<:Real},
                   w::AbstractVector{<:Real};
                   max_depth::Int=4,
                   min_weight::Float64=1.0,
                   min_gain::Float64=0.0)
    nobs, nfeatures = size(X)
    length(y) == nobs || throw(DimensionMismatch("X and y have incompatible sizes"))
    length(w) == nobs || throw(DimensionMismatch("X and w have incompatible sizes"))
    nobs == 0 && throw(ArgumentError("cannot grow a tree from an empty data set"))

    nodes = Node[]
    root_idx = collect(1:nobs)
    rootW = 0.0
    rootS = 0.0
    @inbounds @simd for i in 1:nobs
        wi = Float64(w[i])
        rootW += wi
        rootS += wi * Float64(y[i])
    end
    rootW > 0 || throw(ArgumentError("sum of tree weights must be positive"))

    function build(idx::Vector{Int}, depth::Int, W::Float64, S::Float64)
        leaf_value = S / W
        pos = Int32(length(nodes) + 1)
        push!(nodes, Node(0, leaf_value, 0, 0, true))

        (depth >= max_depth || length(idx) <= 1 || W < 2min_weight) && return pos

        parent_score = S*S/W
        best_gain = min_gain
        best_feature = 0
        bestWL = bestSL = 0.0

        @inbounds for f in 1:nfeatures
            WL = 0.0
            SL = 0.0
            @simd for k in eachindex(idx)
                i = idx[k]
                if X[i,f] < 0
                    wi = Float64(w[i])
                    WL += wi
                    SL += wi * Float64(y[i])
                end
            end

            WR = W - WL
            (WL < min_weight || WR < min_weight) && continue
            SR = S - SL
            gain = SL*SL/WL + SR*SR/WR - parent_score

            if gain > best_gain
                best_gain = gain
                best_feature = f
                bestWL = WL
                bestSL = SL
            end
        end

        best_feature == 0 && return pos

        # Partition only once, after selecting the best feature.
        nl = 0
        @inbounds for i in idx
            nl += X[i,best_feature] < 0
        end
        leftidx = Vector{Int}(undef, nl)
        rightidx = Vector{Int}(undef, length(idx)-nl)
        il = ir = 0
        @inbounds for i in idx
            if X[i,best_feature] < 0
                il += 1
                leftidx[il] = i
            else
                ir += 1
                rightidx[ir] = i
            end
        end

        WR = W - bestWL
        SR = S - bestSL
        left = build(leftidx, depth+1, bestWL, bestSL)
        right = build(rightidx, depth+1, WR, SR)
        nodes[pos] = Node(Int32(best_feature), 0.0, left, right, false)
        return pos
    end

    build(root_idx, 0, rootW, rootS)
    return RegressionTree(nodes)
end

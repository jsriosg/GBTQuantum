using Test
using GBTQuantum

@testset "Tree target translation invariance" begin
    # Nontrivial weighted data: enough observations/features to exercise
    # recursive split selection and nonuniform importance weights.
    X = Int8[
        -1 -1 -1;
        -1 -1  1;
        -1  1 -1;
        -1  1  1;
         1 -1 -1;
         1 -1  1;
         1  1 -1;
         1  1  1
    ]

    y = [-2.1, -0.4, 1.7, 0.2, 3.4, 1.1, -1.3, 2.6]
    w = [1.0, 2.0, 1.5, 0.75, 3.0, 1.25, 2.5, 0.5]
    c = 7.314159265358979

    t = GBTQuantum.grow_tree(X, y, w; max_depth=3)
    tc = GBTQuantum.grow_tree(X, y .+ c, w; max_depth=3)

    @test length(t.nodes) == length(tc.nodes)

    # 1. Translation must leave the complete greedy topology unchanged.
    for (n, nc) in zip(t.nodes, tc.nodes)
        @test n.isleaf == nc.isleaf
        @test n.feature == nc.feature
        @test n.left == nc.left
        @test n.right == nc.right
    end

    # 2. Before gauge fixing, every prediction must differ by exactly c.
    pred = [GBTQuantum.predict(t, @view X[i, :]) for i in axes(X, 1)]
    predc = [GBTQuantum.predict(tc, @view X[i, :]) for i in axes(X, 1)]
    @test predc ≈ pred .+ c atol=1e-12 rtol=1e-12

    # Reproduce the weighted gauge fixing used by the optimizer:
    # subtract the weighted mean tree prediction from every leaf.
    function gauge_fixed_predictions(tree)
        p = [GBTQuantum.predict(tree, @view X[i, :]) for i in axes(X, 1)]
        μ = sum(w .* p) / sum(w)
        return p .- μ
    end

    # 3. The gauge-fixed trees must therefore make identical predictions.
    @test gauge_fixed_predictions(tc) ≈ gauge_fixed_predictions(t) atol=1e-12 rtol=1e-12
end

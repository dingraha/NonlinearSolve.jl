"""
    SimpleNewtonRaphson(autodiff)
    SimpleNewtonRaphson(; autodiff = nothing)

A low-overhead implementation of Newton-Raphson. This method is non-allocating on scalar
and static array problems.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to  `nothing` (i.e.
    automatic backend selection). Valid choices include jacobian backends from
    `DifferentiationInterface.jl`.
"""
@concrete struct SimpleNewtonRaphson <: AbstractSimpleNonlinearSolveAlgorithm
    autodiff
    linesearch <: Union{Val{false}, Val{true}}
end

function SimpleNewtonRaphson(; autodiff = nothing, linesearch::Union{Bool, Val{true}, Val{false}} = Val(false),
    )
    linesearch = linesearch isa Bool ? Val(linesearch) : linesearch
    return SimpleNewtonRaphson(autodiff, linesearch)
end

const SimpleGaussNewton = SimpleNewtonRaphson

function configure_autodiff(prob, alg::SimpleNewtonRaphson)
    autodiff = something(alg.autodiff, AutoForwardDiff())
    autodiff = SciMLBase.has_jac(prob.f) ? autodiff :
        NonlinearSolveBase.select_jacobian_autodiff(prob, autodiff)
    @set! alg.autodiff = autodiff
    return alg
end

function SciMLBase.__solve(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        alg::SimpleNewtonRaphson, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false), termination_condition = nothing, kwargs...
    )
    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    autodiff = alg.autodiff
    x = NLBUtils.maybe_unaliased(prob.u0, alias_u0)
    fx = NLBUtils.evaluate_f(prob, x)

    iszero(fx) &&
        return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

    abstol, reltol,
        tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    if alg.linesearch isa Val{true}
        ls_alg = BackTracking(; autodiff=autodiff)
        ls_cache = init(prob, ls_alg, fx, x)
    else
        ls_cache = nothing
    end

    @bb xo = similar(x)
    fx_cache = (SciMLBase.isinplace(prob) && !SciMLBase.has_jac(prob.f)) ?
        NLBUtils.safe_similar(fx) : fx
    jac_cache = Utils.prepare_jacobian(prob, autodiff, fx_cache, x)
    J = Utils.compute_jacobian!!(nothing, prob, autodiff, fx_cache, x, jac_cache)

    for _ in 1:maxiters
        @bb copyto!(xo, x)
        δx = NLBUtils.restructure(x, J \ NLBUtils.safe_vec(fx))
        @bb δx .*= -1

        if ls_cache === nothing
            α = true
        else
            ls_sol = solve!(ls_cache, xo, δx)
            α = ls_sol.step_size # Ignores the return code for now
        end

        @bb x .+= α * δx

        solved, retcode, fx_sol, x_sol = Utils.check_termination(tc_cache, fx, x, xo, prob)
        solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

        fx = NLBUtils.evaluate_f!!(prob, fx, x)
        J = Utils.compute_jacobian!!(J, prob, autodiff, fx_cache, x, jac_cache)
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

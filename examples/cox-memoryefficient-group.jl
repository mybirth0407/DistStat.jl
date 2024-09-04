using DistStat, Random, LinearAlgebra, LoopVectorization
using NPZ, Pickle
using CUDA, Adapt


mutable struct COXUpdate
    maxiter::Int
    step::Int
    verbose::Bool
    tol::Real
    function COXUpdate(; maxiter::Int=100, step::Int=10, verbose::Bool=false, tol::Real=1e-10)
        maxiter > 0 || throw(ArgumentError("maxiter must be greater than 0."))
        tol > 0 || throw(ArgumentError("tol must be positive."))
        new(maxiter, step, verbose, tol)
    end
end


"""
    breslow_ind(x)
    
returns indexes of result of cumulutive sum corresponding to "W". `x` is assumed to be nonincreasing.
"""
function breslow_ind(x::AbstractVector)
    uniq = unique(x)
    lastinds = findlast.(isequal.(uniq), [x])
    invinds = findfirst.(isequal.(x), [uniq])
    lastinds[invinds]
end


mutable struct COXVariables{T, A}
    m::Int # rows, number of subjects
    n::Int # cols, number of predictors
    β::MPIVector{T,A} # [n]
    β_prev::MPIVector{T,A} # r x [n]
    δ::A # indicator for right censoring (0 if censored)
    λ::T # regularization parameter
    t::AbstractVector{T} # a vector containing timestamps, should be in descending order. 
    breslow::A
    σ::T # step size. 1/(2 * opnorm(X)^2) for guaranteed convergence
    grad::MPIVector{T,A}
    w::A
    W::A
    W_dist::MPIMatrix{T,A} # Updated field for group sizes
    q::A # (1-π)δ, distributed
    eval_obj::Bool
    obj_prev::Real

    function COXVariables(X::MPIMatrix{T,A}, δ::A, λ::AbstractFloat,
                        t::A, group_info::Vector{Int};
                        # σ::T=convert(T, 1/(2*opnorm(X; verbose=true)^2)), 
                        σ::T=0.000001,
                        eval_obj=false) where {T,A}
        m, n = size(X)
        β = MPIVector{T,A}(group_info, undef, n)
        β_prev = MPIVector{T,A}(group_info, undef, n)
        # β = MPIVector{T,A}(group_info, undef, n)
        # β_prev = MPIVector{T,A}(group_info, undef, n)
        fill!(β, zero(T))
        fill!(β_prev, zero(T))
        δ = convert(A{T}, δ)
        breslow = convert(A{Int}, breslow_ind(convert(Array, t)))
        grad = MPIVector{T,A}(group_info, undef, n)
        w = A{T}(undef, m)
        W = A{T}(undef, m)
        q = A{T}(undef, m)
        
        # println(size(β)) # 1301
        # println(size(β_prev)) # 1301
        # println(size(w)) # 723
        # println(size(W)) # 723
        # println(size(q)) # 723
        
        # Ensure group_sizes match with m and n
        # W_dist = MPIArray(MPI.COMM_WORLD, (partition_sizes,), m)
        # MPIMatrix{T,A} = MPIArray{T,2,A}
        W_dist = MPIMatrix{T,A}(group_info, undef, 1, m)
        # println(size(W_dist)) # 723

        new{T,A}(m, n, β, β_prev, δ, λ, t, breslow, σ, grad, w, W, W_dist, q, eval_obj, -Inf)
    end
end


function reset!(v::COXVariables{T,A}; seed=nothing) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
end


function soft_threshold(x::T, λ::T) ::T where T <: AbstractFloat
    @assert λ >= 0 "Argument λ must be greater than or equal to zero."
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end


function π_δ!(out, w, W_dist, δ, breslow, W_range)
    # fill `out` with zeros beforehand. 
    m = length(δ)
    W_base = minimum(W_range) - 1
    W_local = W_dist.localarray
    @avx for i in 1:m
        outi = zero(eltype(w))
        for j in 1:length(W_range)
            outi += ifelse(breslow[i] <= breslow[j + W_base], δ[j + W_base] * w[i] / W_local[j], zero(eltype(w)))
        end
        out[i] = outi
    end
    DistStat.Barrier()
    DistStat.Allreduce!(out)
    return out
end


function get_breslow!(out, cumsum_w, bind)
    out .= cumsum_w[bind]
    out
end


function cox_grad!(group_info, out, w, W, W_dist, t, q, X, β, δ, bind)
    T = eltype(β)
    m, n = size(X)
    mul!(w, X, β)
    w .= exp.(w) 
    cumsum!(q, w) # q is used as a dummy variable
    get_breslow!(W, q, bind)
    W_dist .= distribute(group_info, reshape(W, 1, :))
    fill!(q, zero(eltype(q)))
    π_δ!(q, w, W_dist, δ, bind, W_dist.partitioning[DistStat.Rank()+1][2])
    q .= δ .- q
    mul!(out, transpose(X), q) # ([n]) = (n x [m]) x (m)
    out
end


cox_grad!(group_info::Vector{Int}, v::COXVariables{T,A}, X) where {T,A} = cox_grad!(group_info, v.grad, v.w, v.W, v.W_dist, v.t, v.q, X, v.β, v.δ, v.breslow)


function update!(group_info::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    #mul!(v.tmp_m_local1, X, v.β) # {m} = {m x [n]} * {[n]}.
    #v.tmp_m_local1 .= exp.(v.tmp_m_local1) # w
    #cumsum!(v.tmp_m_local2, v.tmp_m_local1) # W. TODO: deal with ties.
    #v.tmp_1m .= distribute(reshape(v.tmp_m_local2, 1, :)) # W_dist: distribute W.
    #v.tmp_mm .= v.π_ind .* v.tmp_m_local1 ./ v.tmp_1m # (π_ind .* w) ./ W_dist. computation order is determined for CuArray safety. 
    #pd = mul!(v.tmp_m_local1, v.tmp_mm, v.δ) # {m} = {m x [m]} * {m}.
    #v.tmp_m_local2 .= v.δ .- pd # {m}. 
    #grad = mul!(v.tmp_n, transpose(X), v.tmp_m_local2) # {[n]} = {[n] x m} * {m}.
    cox_grad!(group_info, v, X)
    v.β .= soft_threshold.(v.β .+ v.σ .* v.grad , v.λ) # {[n]}.
end


function get_objective!(group_info::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad is dummy
    nnz = sum(v.grad)
    
    if v.eval_obj
        v.w .= exp.(mul!(v.w, X, v.β))
        cumsum!(v.q, v.w) # q is dummy
        get_breslow!(v.W, v.q, v.breslow)
        o1 = dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W))
        # println(size(v.λ)) # ()
        # println(size(v.β)) # (1301, )
        obj = dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W)) .- v.λ .* sum(abs.(v.β))
        converged = abs(v.obj_prev - obj) < 1e-12
        # print(v.W)
        return converged, (obj, nnz)
    else
        v.grad .= abs.(v.β_prev .- v.β)
        relchange = norm(v.grad) / (norm(v.β) + 1e-20)
        converged = relchange < 1e-12
        return converged, (relchange, maximum(v.grad), nnz)
    end
end


function cox_one_iter!(group_info::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables)
    copyto!(v.β_prev, v.β)
    update!(group_info, X, u, v)
end


function loop!(group_info::Vector{Int}, X::MPIArray, u, iterfun, evalfun, args...)
    converged = false
    t = 0
    while !converged && t < u.maxiter
        t += 1
        iterfun(group_info, X, u, args...)
        if t % u.step == 0
            converged, monitor = evalfun(group_info, X, u, args...)
            if DistStat.Rank() == 0
                println(t, ' ', monitor)
            end
        end
    end
end


function cox!(group_info::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables)
    loop!(group_info, X, u, cox_one_iter!, get_objective!, v)
end


include("cmdline.jl")
opts = parse_commandline_cox()
if DistStat.Rank() == 0
    println("world size: ", DistStat.Size())
    println(opts)
end


iter = opts["iter"]
interval = opts["step"]
T = Float64
A = Array

if opts["Float32"]
    T = Float32
end
init_opt = opts["init_from_master"]
seed = opts["seed"]
eval_obj = opts["eval_obj"]
lambda = opts["lambda"]

# Vector of Vector{Any}
variable_to_groups = Pickle.load("examples/variable_to_groups_index.pkl")
# Vector{Int}
group_info = Int64[vcat(variable_to_groups...)...]

X = npyread("examples/array_predictors.npy", A=A, group_info=group_info);
# println(X)
println("main/localarray :", size(X.localarray))
# println(1/(2*opnorm(X; verbose=true)^2))

# println(size(X)) # (723, 1301)
survival = npzread("examples/survival.npz");
survival_event = survival["status"]
δ = convert(A{T}, survival_event)
# print(size(X))

DistStat.Bcast!(δ) # synchronize the choice for δ.

uquick = COXUpdate(;maxiter=2, step=1, verbose=true)
u = COXUpdate(;maxiter=iter, step=interval, verbose=true)
# for simulation run, we just assume that the data are in reversed order of survivial time
t = convert(A{T}, collect(reverse(1:size(X,1))))
v = COXVariables(X, δ, lambda, t, group_info; eval_obj=eval_obj)

# variable_to_groups = [convert(Vector{Int64}, x) for x in Pickle.load("examples/variable_to_groups_index.pkl")]

cox!(group_info, X, uquick, v)
reset!(v; seed=seed)

if DistStat.Rank() == 0
    println("---------------------------------")
    @time cox!(group_info, X, u, v)
else
    cox!(group_info, X, u, v)
end

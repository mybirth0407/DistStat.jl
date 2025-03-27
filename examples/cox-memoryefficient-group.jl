using DistStat, Random, LinearAlgebra, LoopVectorization
using NPZ, Pickle
using CUDA, Adapt
using MPI
using StatsBase

import MPI: COMM_WORLD


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
    penalty::Penalty
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

    function COXVariables(
        X::MPIMatrix{T,A},
        δ::A,
        λ::AbstractFloat,
        t::A,
        group_info::Vector{Int64},
        penalty::Penalty;
        # σ::T=convert(T, 1/(2*opnorm(X; verbose=true)^2)), 
        # σ::T=0.000001,
        σ::T,
        eval_obj=false
    ) where {T,A}

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
        fill!(grad, zero(T))
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

        new{T,A}(m, n, penalty, β, β_prev, δ, λ, t, breslow, σ, grad, w, W, W_dist, q, eval_obj, -Inf)
    end
end


function reset!(v::COXVariables{T,A}; seed=nothing) where {T,A}
    fill!(v.β, zero(T))
    fill!(v.β_prev, zero(T))
    fill!(v.grad, zero(T))
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
    # println("cox-group/cox_grad! size(w): ", size(w))
    # println("cox-group/cox_grad! size(X): ", size(X)) # (500, 2000)
    # println("cox-group/cox_grad! size(β): ", size(β)) (2000)
    # println("cox-group/cox_grad! local size(X): ", size(X.localarray))
    # println("cox-group/cox_grad! local size(β): ", size(β.localarray))
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


function get_objective!(group_info::Vector{Int}, splits::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables{T,A}) where {T,A}
    v.grad .= (v.β .!= 0) # grad is dummy
    nnz = sum(v.grad)
    
    if v.eval_obj
        v.w .= exp.(mul!(v.w, X, v.β))
        cumsum!(v.q, v.w) # q is dummy
        get_breslow!(v.W, v.q, v.breslow)
        # println(size(v.λ)) # ()
        # println(size(v.β)) # (1301, )
        # println("sum: ", sum(v.β))
        # obj = dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W))
        # print(sum(abs.(local_arr)))
        # println(v.β
        # copyto!(v.β_prev, v.β)
        local_abs_sum = zeros(DistStat.Size())
        local_abs_sum[1+DistStat.Rank()] = sum(abs.(v.β.localarray))
        DistStat.Barrier()
        DistStat.Allreduce!(local_abs_sum)
        # pen과 unpen 될 대상이 나눠져 있는데, 이거 어떻게 처리?
        # obj = dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W)) .- v.λ .* sum(x)
        # obj = dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W)) / size(X, 1) .- v.λ .* sum(local_abs_sum)
        # obj = (dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W))) ./ size(X, 1) .- value(v.penalty, v.β, splits, DistStat.Rank())
        obj = (dot(v.δ, mul!(v.q, X, v.β) .- log.(v.W))) .- value(v.penalty, v.β, splits, DistStat.Rank())
        # println(obj)
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


function cox_one_iter!(group_info::Vector{Int}, splits::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables)
    copyto!(v.β_prev, v.β)
    cox_grad!(group_info, v, X)
    indices = [0; cumsum(splits)]

    """
    # println(indices)
    # println(DistStat.Rank(), size(v.β.localarray .+ v.σ .* v.grad.localarray))
    # println(DistStat.Rank(), size(v.β.localarray))
    s = indices[DistStat.Rank()+1] +1
    e = indices[DistStat.Rank()+2]
    i = [s; e;]
    global_β = zeros(size(v.β))
    global_β[s:e] .= v.β.localarray
    global_grad = zeros(size(v.grad))
    global_grad[s:e] .= v.grad.localarray
    DistStat.Barrier()
    DistStat.Allreduce!(global_β)
    DistStat.Allreduce!(global_grad)
    # println(global_β)
    prox!(global_β, v.penalty, global_β .+ v.σ .* global_grad, i)
    # println(global_β)
    v.β.localarray = copy(global_β[s:e])
    v.grad.localarray = copy(global_grad[s:e])
    # println(v.β)
    # update!(group_info, X, u, v)
    """

    s = indices[DistStat.Rank()+1] + 1
    e = indices[DistStat.Rank()+2]
    start_end = [s; e;]
    # println(i)
    # println("1: ", size(v.β.localarray))
    # println(size(v.grad))
    # println(size(v.penalty))
    # println("2: ", size(v.β.localarray .+ v.σ .* v.grad.localarray))
    after = v.β.localarray .+ v.σ .* v.grad.localarray
    prox!(
        v.β.localarray,
        v.penalty,
        after,
        start_end)
    return v.β
end


function loop!(group_info::Vector{Int}, splits::Vector{Int}, X::MPIArray, u, iterfun, evalfun, args...)
    converged = false
    t = 0
    while !converged && t < u.maxiter
        t += 1
        iterfun(group_info, splits, X, u, args...)
        if t % u.step == 0
            converged, monitor = evalfun(group_info, splits, X, u, args...)
            if DistStat.Rank() == 0
                println(t, ' ', monitor)
            end
        end
    end
end


function cox!(group_info::Vector{Int}, splits::Vector{Int}, X::MPIArray, u::COXUpdate, v::COXVariables)
    loop!(group_info, splits, X, u, cox_one_iter!, get_objective!, v)
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


function split_group_info(group_info::Vector{Int64}, N::Int64, unpen::Int64)
    # 그룹별 변수 개수를 계산
    group_counts = countmap(group_info)

    # 그룹별로 누적 개수를 계산
    cumulative_counts = cumsum(values(group_counts))
    
    # 균등하게 나눌 경우 각 파티션의 크기
    total_variables = length(group_info) - unpen
    target_size = ceil(Int, total_variables / N)

    partition_sizes = []

    # 현재 분할의 시작 인덱스
    current_start = 1

    # N-1개 분할 후 마지막에 몰아넣기
    for _ in 1:(N-1)
        current_end = searchsortedfirst(cumulative_counts, cumulative_counts[current_start] + target_size - 1)
        if current_end > length(cumulative_counts)
            current_end = length(cumulative_counts)
        end
        if current_start == 1
            push!(partition_sizes, [cumulative_counts[current_end]])
        else
            push!(partition_sizes, [cumulative_counts[current_end] - cumulative_counts[current_start - 1]])
        end
        current_start = current_end + 1
    end

    # 마지막 등분은 남은 모든 변수를 포함
    push!(partition_sizes, [total_variables - sum(vcat(partition_sizes...))])
    if unpen > 0
        push!(partition_sizes, [unpen])
    end
    
    return Tuple(partition_sizes)
end

function generate_groups(total::Int64, max_repeat::Int64, unpen::Int64)
    groups = []
    k = 1
    while length(groups) < total - unpen
        size = rand(1:(total - unpen - length(groups)))
        append!(groups, fill(k, size))
        k += 1
    end

    unpen_groups = []
    if unpen > 0
        append!(groups, fill(k, unpen))
    end
    return groups[1:total]
end

function generate_random_sequence(N::Int, m::Int)
    if N < m
        println("오류: N은 m보다 크거나 같아야 합니다.")
        return []
    end
    
    # 1부터 (N-m)까지의 숫자 생성
    base_numbers = collect(1:(N-m))
    
    # 결과 배열 초기화
    result = Int[]
    
    # 각 숫자마다 랜덤한 개수(1~5개)로 추가
    for num in base_numbers
        # 이 숫자가 몇 번 나타날지 랜덤하게 결정 (1~5 사이)
        count = rand(1:5)
        
        # 결정된 개수만큼 현재 숫자를 결과 배열에 추가
        append!(result, fill(num, count))
    end
    
    # 배열 섞기 (선택 사항)
    # shuffle!(result)
    
    return result
end

# Function to create MPIArray with custom splits
function create_custom_mpiarray(data::AbstractMatrix{T}, splits::Vector{Int64})
    # Check validity of splits
    total_split = sum(splits)
    size(data, 2) == total_split || error("Splits do not match the total columns of data")

    # Initialize MPI environment
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Determine local array for each process
    local_start = sum(splits[1:rank]) + 1
    local_end = sum(splits[1:rank+1])
    local_data = data[:, local_start:local_end]
    shape = size(data)
    # Create MPIArray
    rslt = MPIArray{T, length(shape), A}(splits, undef, shape...)
    rslt.localarray = local_data
    return rslt
end

"""
    gather!(out, vec, ind)

Simply returns out .= vec[ind]. Intended for extension on `CuArray`s.
"""
function gather!(out, vec, ind)
    out .= vec[ind]
end


m = opts["rows"]
n = opts["cols"]

if opts["gpu"]
    A = CuArray

    # All the GPU-related functions here
    function π_δ_kernel!(out, w, W_dist, δ, breslow, W_range)
        # fill `out` with zeros beforehand.
        idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
        stride_x = blockDim().x * gridDim().x
        W_base = minimum(W_range) - 1
        for i in idx_x:stride_x:length(out)
            for j in W_range
                @inbounds if breslow[i] <= breslow[j]
                    out[i] += δ[j] * w[i] / W_dist[j - W_base]
                end
            end
        end  
    end

    function π_δ!(out::CuArray, w::CuArray, W_dist, δ, breslow, W_range)
        numblocks = ceil(Int, length(w)/256)
        CUDA.@sync begin
            @cuda threads=256 blocks=numblocks π_δ_kernel!(out, w, W_dist.localarray, δ, breslow, W_range)
        end
        DistStat.Allreduce!(out)
        out
    end

    function breslow_kernel!(out, cumsum_w, bind)
        idx_x = (blockIdx().x-1) * blockDim().x + threadIdx().x
        stride_x = blockDim().x * gridDim().x
        for i = idx_x: stride_x:length(out)
            out[i]=cumsum_w[bind[i]]
        end
    end

    function get_breslow!(out::CuArray, cumsum_w::CuArray, bind)
        numblocks = ceil(Int, length(out)/256)
        CUDA.@sync begin
            @cuda threads=256 blocks=numblocks breslow_kernel!(out, cumsum_w, bind)
        end
        out
    end
end

# unpen = opts["unpen"]
unpen = 0

Random.seed!(seed)

###
# Example usage
data = rand(T, m, n)  # Input data
σ = 1/(2*opnorm(data)^2)
groups = generate_groups(n, unpen, 100)
group_info = Int64[vcat(groups)...]
println(group_info)
nodes = DistStat.Size()
if unpen > 0
    nodes -= 1
end
splits = split_group_info(group_info, nodes, unpen)
println(splits)

splits = vcat(splits...)
# print(splits)
X = create_custom_mpiarray(data, splits)
survival_event = rand(0:1, m)
###

###
# Vector of Vector{Any}
# variable_to_groups = Pickle.load("examples/variable_to_groups_index.pkl")
# # Vector{Int}
# group_info = Int64[vcat(variable_to_groups...)...]
# group_info_reshaped = reshape(group_info, 1, size(group_info)[1])

# X = npyread("examples/array_predictors.npy", A=A, group_info=group_info);
# survival = npzread("examples/survival.npz");
# survival_event = survival["status"]
###
δ = convert(A{T}, survival_event)

DistStat.Bcast!(δ) # synchronize the choice for δ.

uquick = COXUpdate(;maxiter=2, step=1, verbose=true)
u = COXUpdate(;maxiter=iter, step=interval, verbose=true)
# for simulation run, we just assume that the data are in reversed order of survivial time
t = convert(A{T}, collect(reverse(1:size(X,1))))
p = GroupNormL2(lambda, group_info)
gather!(p.tmp_p, p.tmp_g, p.gidx)
v = COXVariables(X, δ, lambda, t, group_info, p; σ=σ, eval_obj=eval_obj)

# variable_to_groups = [convert(Vector{Int64}, x) for x in Pickle.load("examples/variable_to_groups_index.pkl")]

cox!(group_info, splits, X, uquick, v)
reset!(v; seed=seed)

if DistStat.Rank() == 0
    println("---------------------------------")
    @time cox!(group_info, splits, X, u, v)
else
    cox!(group_info, splits, X, u, v)
end

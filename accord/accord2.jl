import MPI
import MPI: COMM_WORLD
using Random, SparseArrays, BenchmarkTools, LinearAlgebra, ArgParse, Printf, Dates

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input", "-i"
            help = "input npy file"
            default = nothing
        "--out", "-o"
            help = "name of output file"
            default = nothing
        "--l1", "-l"
            help = "lambda penalty"
            arg_type = Float64
            default = 0.1
        "--tau", "-t"
            help = "Starting step size"
            arg_type = Float64
            default = 1.0
        "--epsilon", "-e"
            help = "Stoppint criterion"
            arg_type = Float64
            default = 1e-5
        "--max_outer"
            help = "number of maximum outer iterations"
            arg_type = Int
            default = 100
        "--max_inner"
            help = "number of maximum inner iterations"
            arg_type = Int
            default = 20
        # "--gpu"
        #     help = "use gpu"
        #     action = :store_true
        # "--Float32"
        #     help = "use Float32 instead of Float64"
        #     action = :store_true
    end
    return parse_args(s)
end

include("../src/distdirectives.jl")
include("../src/distarray.jl")
include("../src/distlinalg.jl")
include("../src/reduce.jl")
include("../src/accumulate.jl")
include("../src/broadcast.jl")
include("../src/arrayfunctions.jl")
include("../src/utils.jl")
include("../src/io.jl")

MPI.Initialized() || MPI.Init()

@inline function Size()
    MPI.Comm_size(COMM_WORLD)
end

@inline function Rank()
    MPI.Comm_rank(COMM_WORLD)
end

mutable struct ACCORDUpdate
    out_iter::Int
    inner_iter::Int
    tau_start::Real
    tau_min::Real
    tol::Real
    function ACCORDUpdate(out_iter, inner_iter, tau, tau_min, tol)
        out_iter > 0 || throw(ArgumentError("iter must be greater than 0."))
        inner_iter > 0 || throw(ArgumentError("iter must be greater than 0."))
        tau > 0 || throw(ArgumentError("step size must be greater than 0."))
        tol > 0 || throw(ArgumentError("tolerance must be positive."))
        new(out_iter, inner_iter, tau, tau_min, tol)
    end
end

mutable struct ACCORDvariables{T, A}
    n::Int
    p::Int
    lambda::Real
    X::MPIMatrix{T,A}
    Y::Matrix{T}
    GT::Matrix{T}
    OmegaT::SparseMatrixCSC{T,<:Integer}
    OmegaT_old::SparseMatrixCSC{T,<:Integer}
    Identity::SparseMatrixCSC{T,<:Integer}
    function ACCORDvariables(X::MPIMatrix{T,A}, lambda::Real, OmegaT::SparseMatrixCSC{T,<:Integer}) where {T,A}
        lambda >= 0 || throw(ArgumentError("penalty lambda must be nonnegative."))
        n, p = size(X)
        Identity = SparseMatrixCSC(I, p, p)[1:p, X.partitioning[Rank() + 1][2]]

        @assert size(OmegaT, 1) == p
        OmegaT_old = deepcopy(OmegaT)

        Y = Matrix{T}(undef, n, size(OmegaT, 2))
        GT = Matrix{T}(undef, p, size(OmegaT, 2))
        new{T,A}(n, p, lambda, X, Y, GT, OmegaT, OmegaT_old, Identity)
    end
    function ACCORDvariables(X::MPIMatrix{T,A}, lambda::Real) where {T,A}
        # start with default identity
        _, p = size(X)
        OmegaT = SparseMatrixCSC{T, Integer}(I, p, p)[1:p, X.partitioning[Rank() + 1][2]]
        return ACCORDvariables(X, lambda, OmegaT)
    end
end

# TODO need to change when using replication
function compute_g!(v::ACCORDvariables{T,A}) where {T, A}
    # compute Y = X * Omega^T 
    # and return partial computation of g (smooth part of loss function)
    mul_1d!(v.Y, v.X, v.OmegaT_old) 
    return 0.5 * sum(v.Y .^ 2) / v.n 
end

function compute_grad!(v::ACCORDvariables{T,A}) where {T, A}
    # compute G^T = X^T * Y / n, gradient for g(Omega)
    mul_1d!(v.GT, transpose(v.X), v.Y / v.n)
    return
end

function compute_Omega!(v::ACCORDvariables{T,A}, tau::Real) where {T, A}
    # apply proximal update and update omega
    o_tilde = v.OmegaT_old - tau * v.GT
    v.OmegaT = sparse(map(x -> abs(x) - tau * v.lambda > 0.0 ? sign(x)*(abs(x) - tau * v.lambda) : 0.0, o_tilde))
    v.OmegaT[v.Identity] = map(x -> x + sqrt(x^2 + 4*tau), o_tilde[v.Identity])
    return
end

function compute_Q(v::ACCORDvariables{T,A}, tau::Real) where {T, A}
    # compute Q function for backtracking, also return maximum difference for stopping criterion
    D = v.OmegaT - v.OmegaT_old
    D_dot_G = sum(D .* v.GT)
    Frobenius_D = sum(D .* 2)

    partial_maxdiff = maximum(abs.(D))
    partial_Q = D_dot_G + Frobenius_D / (2.0 * tau)

    return partial_Q, partial_maxdiff
end

function update!(u::ACCORDUpdate, v::ACCORDvariables{T,A}, g_old::Real, i_outer::Integer, start_time::DateTime) where {T,A}
    tau = u.tau_start
    partial_maxdiff = 0.0
    compute_grad!(v) # need to execute compute_g! before
    for i_inner in 1:u.inner_iter
        compute_Omega!(v, tau) # update OmegaT with OmegaT_old
        partial_g = compute_g!(v)
        partial_q, partial_maxdiff = compute_Q(v, tau)
        partial_nnz_count = nnz(v.OmegaT)

        temp = [partial_g, partial_q, partial_nnz_count]
        MPI.Allreduce!(temp, MPI.SUM, MPI.COMM_WORLD)

        g = temp[1]
        Q = temp[2] + g_old
        nnz_ratio = temp[3] * 100.0 / v.n / v.p
        
        if Rank() == 0
            @printr("Round %03d.%02d [%10.4lf]: tau = %10.4lf, g = %10.4lf, Q = %10.4lf, %%nnz = %9.6lf, nnz = %10.0lf", 
                i_outer, i_inner, Dates.value(now() - start_time) * 0.001, tau, g, Q, nnz_ratio, temp[3])
        end
        if tau <= u.tau_min || g <= Q
            break
        end
        tau = tau /= 2.0
    end
    return partial_maxdiff
end

# if Rank() != 0
#     redirect_stdout(devnull)
# end

opts = parse_commandline()
if Rank() == 0
    println("world size: ", Size())
    # println(opts)
    for (arg, val) in opts
        println(" $arg => $val")
    end
end
X = npyread(opts["input"])
lambda = opts["l1"]
# if Rank() == 0
#     println("Matrix size: ", size(X))
#     println("Matrix size: ", X.partitioning)
# end

#TODO make outer loop with update!
v = ACCORDvariables(X, lambda)
if Rank() == 0
    println("Current OmegaT: ", v.OmegaT)
end
# g = compute_g(v)
# if Rank() == 0
#     println("g0: ", g)
# end
# sync()
# if Rank() == 1
#     println("g1: ", g)
# end
# partial_g = compute_g(v)
import MPI
import MPI: COMM_WORLD
using Random, SparseArrays, BenchmarkTools, LinearAlgebra, ArgParse, Printf, Dates, Format

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
            help = "starting step size"
            arg_type = Float64
            default = 1.0
        "--tau_min", "-m"
            help = "minimum step size"
            arg_type = Float64
            default = 0.0
        "--epsilon", "-e"
            help = "stopping criterion"
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
    OmegaT::SparseMatrixCSC{T,Int}
    OmegaT_old::SparseMatrixCSC{T,Int}
    diag_indx::Int #for diagonal coordinate
    function ACCORDvariables(X::MPIMatrix{T,A}, lambda::Real, OmegaT::SparseMatrixCSC{T,Int}) where {T,A}
        lambda >= 0 || throw(ArgumentError("penalty lambda must be nonnegative."))
        n, p = size(X)
        diag_indx = 1 - collect(X.partitioning[Rank() + 1][2])[1]

        @assert size(OmegaT, 1) == p
        OmegaT_old = deepcopy(OmegaT)

        Y = Matrix{T}(undef, n, size(OmegaT, 2))
        GT = Matrix{T}(undef, p, size(OmegaT, 2))
        new{T,A}(n, p, lambda, X, Y, GT, OmegaT, OmegaT_old, diag_indx)
    end
    function ACCORDvariables(X::MPIMatrix{T,A}, lambda::Real) where {T,A}
        # start with default identity
        _, p = size(X)
        OmegaT = SparseMatrixCSC{T, Int}(I, p, p)[1:p, X.partitioning[Rank() + 1][2]]
        return ACCORDvariables(X, lambda, OmegaT)
    end
end

# TODO need to change when using replication
function compute_g!(v::ACCORDvariables{T,A}) where {T, A}
    # compute Y = X * Omega^T 
    # and return partial computation of g (smooth part of loss function)
    mul_1d!(v.Y, v.X, v.OmegaT)
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
    v.OmegaT[diagind(o_tilde, v.diag_indx)] = map(x -> 0.5 * (x + sqrt(x^2 + 4*tau)), diag(o_tilde, v.diag_indx))
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
    temp = [0.0, 0.0, 0]
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
        nnz_ratio = temp[3] * 100.0 / (v.p ^ 2)
        
        if Rank() == 0
            @printf("Round %03d.%02d [%10.4lf]: tau = %10.4lf, g = %10.4lf, Q = %10.4lf, %%nnz = %9.6lf, nnz = %d\n", 
                i_outer, i_inner, Dates.value(now() - start_time) * 0.001, tau, g, Q, nnz_ratio, temp[3])
        end
        if tau <= u.tau_min || g <= Q
            break
        end
        tau /= 2.0
    end
    return partial_maxdiff, temp[1], temp[3]
end

function accord!(u::ACCORDUpdate, v::ACCORDvariables{T,A}, start_time::DateTime) where {T,A}
    temp = [compute_g!(v)]
    MPI.Allreduce!(temp, MPI.SUM, MPI.COMM_WORLD)
    g_omega = temp[1]
    omega_nnz = 0
    if Rank() == 0
        @printf("g_0 = %lf\n", g_omega)
    end
    for i_outer in 1:u.out_iter
        partial_maxdiff, g_omega, omega_nnz = update!(u, v, g_omega, i_outer, start_time)
        temp[1] = partial_maxdiff
        MPI.Allreduce!(temp, MPI.MAX, MPI.COMM_WORLD)
        v.OmegaT, v.OmegaT_old = v.OmegaT_old, v.OmegaT
        if temp[1] <= u.tol
            break
        end
    end
    return omega_nnz
end

# if Rank() != 0
#     redirect_stdout(devnull)
# end

start_time = Dates.now()

opts = parse_commandline()
# if Rank() == 0
#     println("world size: ", Size())
#     # println(opts)
#     for (arg, val) in opts
#         println(" $arg => $val")
#     end
# end

X = npyread(opts["input"])
output_dir = opts["out"]
lambda = opts["l1"]
tau_start = opts["tau"]
tau_min = opts["tau_min"]
tol = opts["epsilon"]
max_outer = opts["max_outer"]
max_inner = opts["max_inner"]

v = ACCORDvariables(X, lambda)
u = ACCORDUpdate(max_outer, max_inner, tau_start, tau_min, tol)

if Rank() == 0
    @printf("Load complete. Starting iterations [%10.4lf]:\n", Dates.value(now() - start_time) * 0.001)
end
omega_nnz = accord!(u, v, start_time)
if Rank() == 0
    @printf("Saving matrix market files. [%10.4lf]\n", Dates.value(now() - start_time) * 0.001)
end
#Convert to Omega and save in matrix market form
format_d = generate_formatter("%10d")
format_g = generate_formatter("%30.16g")
global_cord = - v.diag_indx
open(join([output_dir, "-", cfmt("%05d", Rank())]), "w") do file
    if Rank() == 0
        write(file, "%%MatrixMarket matrix coordinate real general\n")
        write(file, join([v.p, " ", v.p, " ", Int(omega_nnz), "\n"]))
    end
    for i in 1:(v.OmegaT_old.n)
        k = v.OmegaT_old.colptr[i]
        while k < v.OmegaT_old.colptr[i + 1]
            j = v.OmegaT_old.rowval[k]
            write(file, join([format_d(i + global_cord), " ", format_d(j), " ", format_g(v.OmegaT_old.nzval[k]), "\n"]))
            k += 1
        end
    end
end
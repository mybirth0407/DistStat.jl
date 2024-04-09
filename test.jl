import MPI
import MPI: COMM_WORLD
using Random, SparseArrays, BenchmarkTools, LinearAlgebra

MPI.Initialized() || MPI.Init()

@inline function Size()
    MPI.Comm_size(COMM_WORLD)
end

@inline function Rank()
    MPI.Comm_rank(COMM_WORLD)
end

include("src/distdirectives.jl")
include("src/distarray.jl")
include("src/distlinalg.jl")
include("src/reduce.jl")
include("src/accumulate.jl")
include("src/broadcast.jl")
include("src/arrayfunctions.jl")
include("src/utils.jl")
include("src/io.jl")

# aa = randn(5,5)
# aa = zeros(5,5)
# cc = zeros(5,5)
# if Rank() == 0
#     aa[1,1] = 3
# end
# if Rank() == 1
#     aa[3,3] = 2
# end
# if Rank() != 0
#     redirect_stdout(devnull)
# end
# show(aa)
# MPI.Allreduce!(aa, cc, MPI.SUM, MPI.COMM_WORLD)

aa = [1 1; 1 1]
bb = [1 2; 3 4]
cc = [0 1; 1 0]
if Rank() == 0
    println("Singple-process multiplication of AB:")
    println(aa - bb)
    # println(transpose(aa)*bb)
end
A = distribute(aa)
B = distribute(bb)
C = distribute(cc)

# if Rank() == 0
#     println("Multi-process multiplication of AB:")
# end

# mul_1d!(cc, A, bb)
twosum = [float(sum((A + B).localarray))]
println(twosum)
MPI.Allreduce!(twosum, MPI.SUM, MPI.COMM_WORLD)
println(twosum)
# show(C)
# x = npyread("../C_Xhp_5k.npy")
# println(x.localarray[1:5,1:5])
# x = npyread("../test.npy")
# show(x)
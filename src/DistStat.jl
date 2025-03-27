module DistStat

import MPI
import MPI: COMM_WORLD

export Penalty, NormL1, GroupNormL2, value, prox!

using Requires
using LinearMaps

const MapOrMatrix{T} = Union{LinearMap{T},AbstractMatrix{T}}

function __init__()
    MPI.Initialized() || MPI.Init()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin 
        include("cuda.jl")
        set_device!()
        #CuArrays.allowscalar(false)
    end

end

@inline function Size()
    MPI.Comm_size(COMM_WORLD)
end

@inline function Rank()
    MPI.Comm_rank(COMM_WORLD)
end

include("distdirectives.jl")
include("distarray.jl")
include("distlinalg.jl")
include("reduce.jl")
include("accumulate.jl")
include("broadcast.jl")
include("arrayfunctions.jl")
include("utils.jl")
include("penalties.jl")

end # module

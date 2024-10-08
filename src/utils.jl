import LinearAlgebra: diagind
import NPZ
import MPI
import MPI: COMM_WORLD
export npyread

"""
    euclidean_distance(out, A, B; tmp_m=nothing, tmp_n=nothing)

Computes all pairwise distances between two sets of data, A(p x n) and B(p x m).
temp memory: length n, length m
Output: n x m.
"""
function euclidean_distance!(out::AbstractArray, A::AbstractMatrix, B::AbstractMatrix; tmp_n::Union{AbstractArray, Nothing}=nothing, tmp_m::Union{AbstractArray, Nothing}=nothing)
    AT = typeof(B) # this choice is intentional due to the method below.
    ATN =  hasproperty(AT, :name) ? AT.name.wrapper : A

    p, n = size(A)
    p2, m = size(B)
    @assert p == p2
    @assert size(out) == (n, m)
    T = eltype(A)
    
    tmp_n = (tmp_n == nothing) ? ATN{T}(undef, n) : tmp_n
    @assert length(tmp_n) == n 
    tmp_m = (tmp_m == nothing) ? ATN{T}(undef, m) : tmp_m
    @assert length(tmp_m) == m
    
    tmp_n .= reshape(sum(A.^2; dims=1), n)
    tmp_m .= reshape(sum(B.^2; dims=1), m)

    LinearAlgebra.mul!(out, transpose(A), B)
    out .= sqrt.(max.((-2out .+ transpose(tmp_m)) .+ tmp_n, zero(T)))

    if A === B
        out[diagind(out)] .= zero(T)
    end
    out
end


"""
    euclidean_distance!(out, A)

Computes all pairwise distances between data in A (p x [n]). Output: n x [n].
"""
function euclidean_distance!(out::MPIMatrix{T,A}, data::MPIMatrix{T,A}; 
                             tmp_data::Union{A, Nothing}=nothing, tmp_dist::Union{A, Nothing}=nothing, 
                             tmp_vec1::Union{A,Nothing}=nothing, tmp_vec2::Union{A,Nothing}=nothing) where {T,A}
    p, n = size(data)
    @assert size(out) == (n, n)
    local_len = n ÷ Size()
    remainder = n % Size()
    
    local_len_p = remainder > 0 ? local_len + 1 : local_len

    tmp_data = (tmp_data == nothing) ? A{T}(undef, p*(local_len_p)) : tmp_data
    @assert length(tmp_data) >= p * local_len_p
    tmp_dist = (tmp_dist == nothing) ? A{T}(undef, local_len_p^2) : tmp_dist
    @assert length(tmp_dist) >= local_len_p^2
    tmp_vec1 = (tmp_vec1 == nothing) ? A{T}(undef, local_len_p) : tmp_vec1
    @assert length(tmp_vec1) >= local_len_p
    tmp_vec2 = (tmp_vec2 == nothing) ? A{T}(undef, local_len_p) : tmp_vec2
    @assert length(tmp_vec2) >= local_len_p


    for r in 0:Size()-1
        this = data.localarray
        if r == Rank()
            other = data.localarray
            sync()
            Bcast!(reshape(data.localarray, :); root=r)
        else
            other = r < remainder ? @view(tmp_data[1:(p*(local_len+1))]) : @view(tmp_data[1:p*local_len])
            sync()
            Bcast!(other; root=r)
            other = reshape(other, p, :)
        end
        
        tmp_dist_cols = (Rank() < remainder) ? local_len + 1 : local_len
        tmp_dist_rows = (r < remainder) ? local_len + 1 : local_len
        tmp_dist_view = reshape(@view(tmp_dist[1:(tmp_dist_rows * tmp_dist_cols)]), 
                                      tmp_dist_rows, tmp_dist_cols)
        euclidean_distance!(tmp_dist_view, other, this; 
                            tmp_n = @view(tmp_vec1[1:tmp_dist_rows]), 
                            tmp_m = @view(tmp_vec2[1:tmp_dist_cols]))
        out.localarray[out.partitioning[r+1][2], :] .= tmp_dist_view
    end
    out
end


function npyread(
    filename::AbstractString;
    root=0, A=Array, group_info::Vector{Int}=Int[]
)
    success = 0
    shape = 0
    T = nothing
    hdrend = 0
    toh = nothing

    if Rank() == root
        f = open(filename, "r")
        b = read!(f, Vector{UInt8}(undef, NPZ.MaxMagicLen))
        if NPZ.samestart(b, NPZ.NPYMagic)
            seekstart(f)
            hdr = NPZ.readheader(f)
            if hdr.fortran_order
                shape = hdr.shape
                T = eltype(hdr)
                toh = hdr.descr
                hdrend = mark(f)

                success = 1
                close(f)
            else
                success = -1
                close(f)
            end
        else
            close(f)
        end
    end
    success = MPI.bcast(success, root, MPI.COMM_WORLD)
    if success == 0
        error("not a NPY file supported: $filename")
    end
    if success == -1
        error("NPY file must be in fortran order: $filename")
    end
    shape, T, hdrend, toh = MPI.bcast(
        (shape, T, hdrend, toh), root, MPI.COMM_WORLD
    )
    rslt = MPIArray{T, length(shape), A}(undef, shape...)

    arr_skip = sum(rslt.local_lengths[1:Rank()])
    println(hdrend + arr_skip * sizeof(T))
    f = open(filename, "r")
    seek(f, hdrend + arr_skip * sizeof(T))
    rslt.localarray = map(
        toh, read!(f, Array{T}(undef, size(rslt.localarray)))
    )
    close(f)
    rslt
end
using ITensors
using Plots
using StatsBase
using CurveFit
using ITensors: linkinds
import Base.isapprox
using ITensors.HDF5

function initialize_wavefunction(; L::Int)
    @assert isodd(L) "L must be odd"
    sites = siteinds("Qubit", L)
    state_arr = ["0" for l = 1:L]
    productMPS(sites, state_arr)
end

function average_elems(A::Vector{Any})
    A0 = copy(A[1])
    for Ai in A[2:end]
        @assert size(Ai) == size(A0)
        A0 .+= Ai
    end
    A0 ./= length(A)
    return A0
end

"""
Helper function to initialize a density matrix from a wavefunction
"""
density_matrix(ψ::MPS) = outer(ψ, ψ')
density(ψ::MPS) = density_matrix(ψ)

function complement(L::Int, B::Vector{Int})
    setdiff(collect(1:L), B)
end

function partial_transpose(A::MPO, sites)
    A = copy(A)
    for n in sites
        A[n] = swapinds(A[n], siteinds(A, n)...)
    end
    return A
end

function partial_transpose(ρ::ITensor, sites::Vector{<:Index})
    primedsites = prime(sites)
    return replaceinds(ρ, [sites..., primedsites...], [primedsites..., sites...])
end

function trace_norm(ρ::ITensor)
    sites = filter(i -> hasplev(i, 0) & hastags(i, "Site"), inds(ρ))
    U, S, V = ITensors.svd(ρ, sites, cutoff=0)
    return sum(S)
end

function negativity(ρ, B)
    # compute the partial transpose
    ρT = partial_transpose(ρ, B)

    # take the trace norm
    return (trace_norm(ρT) - 1) / 2
end

"""
EN(ρ_AB) = log₂||ρ_AB^(T_B)||₁
"""
function logarithmic_negativity(ρ, B)
    # compute the partial transpose
    ρT = partial_transpose(ρ, B)

    # take the trace norm
    trnorm = trace_norm(ρT)

    # take the logarithm
    return log2(trnorm)
end

function twosite_reduced_density_matrix(ρ::MPO, A::Int, B::Int)
    ρ = copy(ρ)
    sites = siteinds(ρ)

    # new rdm
    ρAB = ITensor(1.0)

    if A == B
        return 0
    end

    if B < A
        println("For some reason, B<A")
        A, B = B, A
    end

    # trace out everything on the left and multiply into ρAB
    for i in 1:(A-1)
        ρAB = ρAB * ρ[i] * delta(sites[i][1], sites[i][2])
    end

    # don't trace out site A
    ρAB = ρAB * ρ[A]

    # trace out everything between A and B and multiply into ρAB
    for i in (A+1):(B-1)
        ρAB = ρAB * ρ[i] * delta(sites[i][1], sites[i][2])
    end

    # don't trace out site B
    ρAB = ρAB * ρ[B]

    # trace out everything to the right of B and multiply into ρB
    for i in (B+1):length(ρ)
        ρAB = ρAB * ρ[i] * delta(sites[i][1], sites[i][2])
    end

    # trace out B to get ρA
    ρA = ρAB * delta(sites[B][1], sites[B][2])

    # trace out A to get ρB
    ρB = ρAB * delta(sites[A][1], sites[A][2])

    # put everything back into MPO form
    ρA = MPO([ρA])
    ρB = MPO([ρB])
    sL = tag_and_plev(ρAB; tag="Site,n=$(A)", lev=0)
    U, S, V = ITensors.svd(ρAB, [sL, prime(sL)], cutoff=0, lefttags="Link,l=$(A)", righttags="Link,l=$(A)")
    ρAB = MPO([U, S * V])

    return ρA, ρB, ρAB
end

function von_neumann_entropy(ρ::MPO)
    T = prod(ρ)
    von_neumann_entropy(T)
end

function von_neumann_entropy(T::ITensor)
    sites = tag_and_plev(T; tag="Site", lev=0)
    U, S, V = svd(T, sites)
    SvN = 0.0
    for n = 1:ITensors.dim(S, 1)
        p = S[n, n]
        if p ≈ 0
            SvN -= 0
        else
            SvN -= p * log2(p)
        end
    end
    return SvN
end

function mutual_information(ρA, ρB, ρAB)
    SA = von_neumann_entropy(ρA)
    SB = von_neumann_entropy(ρB)
    SAB = von_neumann_entropy(ρAB)
    return SA + SB - SAB
end

function mutual_information(ρ::MPO, A::Int, B::Int)
    ρAB = twosite_reduced_density_matrix(ρ, A, B)
    ρA = reduced_density_matrix(ρ, [A])
    ρB = reduced_density_matrix(ρ, [B])
    mutual_information(ρA, ρB, ρAB)
end

function entanglement_entropy(ψ::MPS; b=nothing)
    if isnothing(b)
        b = floor(Int, length(ψ) / 2)
    end
    orthogonalize!(ψ, b)
    U, S, V = svd(ψ[b], (linkind(ψ, b - 1), siteind(ψ, b)))
    S = diag(array(S))

    SvN = 0.0
    for p in S
        # here, we DO want to square the singular values
        # this is because we are computing the entropy of a CUT
        SvN -= p^2 * log2(p^2)
    end

    return SvN
end

function second_Renyi_entropy(ρ)
    return -log(tr(apply(ρ, ρ)))
end

findnearest(A, x) = argmin(abs.(A .- x))

"""
physical_indices(ψ::MPS, idxlist::Vector{Int}, tag::String)
    Given an INTEGER list of desired indices,
    return a list of the corresponding PHYSICAL Index (struct) of the MPS
"""
function physical_indices(ψ::Union{MPS,MPO}, sitelist::Vector{Int}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψ[s])) for s in sitelist]
end

function physical_indices(ψ::Union{MPS,MPO}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψs)) for ψs in ψ]
end

function physical_indices(ψ::ITensor; tag::String="Site")
    getfirst(x -> hastags(x, tag), inds(ψ))
end

function maxlinkdim(ρ::MPO)
    Lmid = floor(Int, length(ρ) / 2)
    return ITensors.dim(linkind(ρ, Lmid))
end

function maxlinkdim(ψ::MPS)
    Lmid = floor(Int, length(ψ) / 2)
    return ITensors.dim(linkind(ψ, Lmid))
end

# here we are KEEPING the indices in indslist
function reduced_density_matrix(ρ::MPO, indslist::Vector{Int})
    # the indices to keep must be a continuous chunk
    sort!(indslist)
    @assert indslist == collect(indslist[begin]:indslist[end])
    L = length(ρ)

    # first, trace out the indices on the LHS
    linds = collect(1:indslist[begin]-1)
    if length(linds) > 0
        ρ = partial_trace(ρ, linds, "left")
    end

    # now trace out the indices on the RHS
    rinds = collect(indslist[end]+1:L) .- length(linds)
    if length(rinds) > 0
        ρ = partial_trace(ρ, rinds, "right")
    end

    return ρ
end

# here we are TRACING OUT the indices in indslist
function partial_trace(ρ::MPO, indslist::Vector{Int}, side::String)
    ρ = copy(ρ)
    s = physical_indices(ρ) # these are the physical sites

    if side == "left"
        border_idx = indslist[end] + 1
    elseif side == "right"
        border_idx = indslist[begin] - 1
    else
        @error side * " is not recognized"
    end

    orthogonalize!(ρ, border_idx)

    # trace out the indices in indslist
    for i in indslist
        ρ[i] = ρ[i] * delta(s[i], prime(s[i]))
    end

    # contract the indices in indslist
    L = ITensor(1.0)
    for i in indslist
        L *= ρ[i]
    end

    # mutliply this into the remaining ρ
    ρ[border_idx] *= L

    to_keep = setdiff(collect(1:length(ρ)), indslist)
    ρ_new = MPO(ρ[to_keep])

    return ρ_new
end


function get_D(ρ::MPO)
    L = length(ρ)
    d = ITensors.dim(siteind(ρ, 1))
    D = d^L
    return D
end

function combine_indices(ρ::MPO)
    ρ = copy(ρ)
    println("Orthogonalizing??")
    @time orthogonalize!(ρ, 1)
    # Combine the primed and unprimed indices at each site to create a super MPS
    sites = siteinds(ρ)
    for i in 1:length(ρ)
        C = combiner(sites[i]...)
        ρ[i] *= C
    end

    # Put this data into an MPS struct
    ψ = MPS(ρ.data)
    # reset_ortho_lims!(ψ)

    return ψ
end

"""
physical_indices(ψ::MPS, idxlist::Vector{Int}, tag::String)
    Given an INTEGER list of desired indices,
    return a list of the corresponding PHYSICAL Index (struct) of the MPS
"""
function physical_indices(ψ, sitelist::Vector{Int}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψ[s])) for s in sitelist]
end

function physical_indices(ψ; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψs)) for ψs in ψ]
end

function link_indices(ψ; tag::String="Link")
    taginds(ψ, tag)
end

function get_plev_inds(T::ITensor, lev::Int)
    inds(T)[findall(x -> plev(x) == lev, inds(T))]
end

function primed_inds(T::ITensor)
    return get_plev_inds(T, 1)
end

function unprimed_inds(T::ITensor)
    return get_plev_inds(T, 0)
end

function taginds(T::ITensor, tag::String)
    inds(T)[findall(x -> hastags(x, tag), inds(T))]
end

function tag_and_plev(T::ITensor; tag::String, lev::Int)
    tinds = taginds(T, tag)
    tinds[findall(x -> plev(x) == lev, tinds)]
end

function linkindT(T::ITensor)
    taginds(T, "Link")
end

function siteindT(T::ITensor)
    taginds(T, "Site")
end

# ITensors already has an isapprox function
# function (T1::ITensor, T2::ITensor; atol=1e-6)
#     # permute indices to match up
#     if inds(T1) != inds(T2)
#         T1 = permute(T1, inds(T2))
#     end

#     # extract elements and compare
#     T1_arr = [array(T1)...]
#     T2_arr = [array(T2)...]

#     for (t1, t2) in zip(T1_arr, T2_arr)
#         @assert isapprox.(t1, t2, atol=atol)
#     end
#     return true
# end

function checkpointed(path::Union{String,Nothing})
    if isnothing(path)
        return false
    end

    # check if the file exists
    if !isfile(path)
        return false
    end

    # else, try to load the ITensor
    res = load_results(path) # this is without ρ or ψ
    try
        if res.max_inner_dim == 0
            res_full = load_results(path, load_MPO=true)
            if typeof(res_full.ρ) == MPO
                return true
            end
        else
            res_full = load_results(path, load_MPS=true)
            if typeof(res_full.ρ) == MPS
                return true
            end
        end
    catch loading_ρ_error
        @show loading_ρ_error
    end

    return false
end

function splat_struct(struc)
    function Name(arg)
        string(arg)
    end
    fnames = fieldnames(typeof(struc))
    vec = []
    for fn in fnames
        push!(vec, getfield(struc, fn))
    end
    return vec
end

function save_structs(struc, path::String)
    function Name(arg)
        string(arg)
    end
    fnames = fieldnames(typeof(struc))
    for fn in fnames
        n = Name(fn)
        d = getfield(struc, fn)

        # If the file already exists, then we either append to it or overwrite
        if isfile(path)
            h5open(path, "r+") do file
                if haskey(file, n) #if key already exists, we want to rewrite
                    delete_object(file, n)
                    write(file, n, d)
                else
                    write(file, n, d)
                end
            end
        else # If the file does not exist, create it
            h5open(path, "w") do file
                write(file, n, d)
            end
        end
    end
end

function load_results(loadpath::String; load_MPO::Bool=false, load_MPS::Bool=false)
    f = h5open(loadpath, "r")
    if load_MPO
        ρ = read(f, "ρ", MPO)
    elseif load_MPS
        ρ = read(f, "ρ", MPS)
    else
        ρ = 0
    end
    d = read(f)
    close(f)
    return Results(ρ, d["L"], d["T"], d["ε"], d["maxdim"], d["max_inner_dim"], d["state_entropy"],
        d["operator_entanglement"], d["trace"], d["logarithmic_negativity"], d["mutual_information"])
end

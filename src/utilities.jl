using ITensors
using Plots 
using StatsBase 
using CurveFit
using ITensors: linkinds
import Base.isapprox

function initialize_wavefunction(;L::Int)
    @assert isodd(L) "L must be odd"
    sites = siteinds("Qubit", L)
    state_arr = ["0" for l=1:L]
    productMPS(sites,state_arr) 
end

"""
Helper function to initialize a density matrix from a wavefunction 
"""
function density_matrix(ψ::MPS)
    sites = siteinds(ψ)
    ψdag = dag(prime(ψ, sites))
    prime!(ψdag, "Link")
    return outer(ψ, ψdag)
end

function entanglement_entropy(ψ::MPS; b=nothing)
    if isnothing(b)
        b = floor(Int, length(ψ)/2)
    end
    orthogonalize!(ψ, b)
    U,S,V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ,b)))
    SvN = 0.0
    for n=1:ITensors.dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN 
end

function second_Renyi_entropy(ρ)
    return -log(tr(apply(ρ, ρ)))
end

findnearest(A,x) = argmin(abs.(A .- x))

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
    Lmid = floor(Int, length(ρ)/2)
    return ITensors.dim(linkind(ρ, Lmid))
end

function maxlinkdim(ψ::MPS)
    Lmid = floor(Int, length(ψ)/2)
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

    if side=="left"
        border_idx = indslist[end]+1
    elseif side=="right"
        border_idx = indslist[begin]-1
    else
        @error side*" is not recognized"
    end

    #orthogonalize!(ρ, border_idx)

    # trace out the indices in indslist
    for i in indslist
        ρ[i] = ρ[i]*delta(s[i],prime(s[i]))
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
    d = ITensors.dim(siteind(ρ,1))
    D = d^L
    return D 
end

function combine_indices(ρ::MPO)
    ρ = copy(ρ)
    orthogonalize!(ρ,1)
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
function physical_indices(ψ::Union{MPS,MPO}, sitelist::Vector{Int}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψ[s])) for s in sitelist]
end

function physical_indices(ψ::Union{MPS,MPO}; tag::String="Site")
    [getfirst(x -> hastags(x, tag), inds(ψs)) for ψs in ψ]
end

function get_plev_inds(T::ITensor, lev::Int)
    inds(T)[findall(x -> plev(x)==lev, inds(T))]
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
    tinds[findall(x -> plev(x)==lev, tinds)]
end

function linkindT(T::ITensor)
    taginds(T, "Link")
end

function siteindT(T::ITensor)
    taginds(T, "Site")
end

function isapprox(T1::ITensor, T2::ITensor; atol=1e-6)
    # permute indices to match up 
    if inds(T1) != inds(T2)
        T1 = permute(T1, inds(T2))
    end

    # extract elements and compare
    T1_arr = [array(T1)...]
    T2_arr = [array(T2)...]

    for (t1,t2) in zip(T1_arr,T2_arr)
        @assert isapprox.(t1, t2, atol=atol)
    end
    return true 
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

function load_results(loadpath::String; load_state=false)
    f = h5open(loadpath,"r")
    if load_state
        ρ = read(f, "ρ", MPO)
    else
        ρ = 0 
    end
    d = read(f)
    return Results(d["L"], d["T"], ρ, d["bitdist"], d["state_entropy"], 
            d["operator_entanglement"], d["trace"])
end
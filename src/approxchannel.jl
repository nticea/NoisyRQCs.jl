
using ITensors
using LinearAlgebra
using TSVD
using JuMP
using Ipopt
using Kronecker: ⊗
using Parameters

include("../src/utilities.jl")
include("../src/kraus.jl")

function centerrange(n, size)
    first = (n - size) ÷ 2 + 1
    last = first + size - 1
    return first:last
end

"""
Convert MPO to array with lists of indices converted into dimensions. Inverse of
toITensor()
"""
function toarray(M::MPO, indlists...)::Array
    T = *(M...)
    return toarray(T, indlists...)
end

"""
Convert ITensor to array with lists of indices converted into dimensions. Inverse of
toITensor()
"""
function toarray(T::ITensor, indlists...)::Array
    combiners = combiner.(filter(!isnothing, indlists))
    flattened = *(T, combiners...)
    combinds = combinedind.(combiners)
    return array(flattened, combinds)
end

"""
Convert array to ITensor with dimensions mapped to lists of indices. Inverse of toarray()
"""
function toITensor(A, indlists...)::ITensor
    combiners = combiner.(filter(!isnothing, indlists))
    combinedinds = combinedind.(combiners)
    T = ITensor(A, combinedinds...)
    return *(T, combiners...)
end

"""
leftlinkind(M::MPS, j::Integer)
leftlinkind(M::MPO, j::Integer)
Get the link or bond Index connecting the MPS or MPO tensor on site j-1 to site j.
If there is no link Index, return `nothing`.
"""
function leftlinkind(M::ITensors.AbstractMPS, j::Integer)
    (j > length(M) || j < 2) && return nothing
    return commonind(M[j-1], M[j])
end

"""
Alias for linkind(M::MPS, j::Integer)
"""
const rightlinkind = linkind

"""
Finds links one either end of a MPS or MPO slice that are loose
      |     |
=> - [_] - [_] - <=
      |     |
"""
function getlooselinks(M::ITensors.AbstractMPS)
    leftlinks = taginds(M[1], "Link")
    @assert length(leftlinks) ≤ 2
    looseleft = getfirst(!=(rightlinkind(M, 1)), leftlinks)

    rightlinks = taginds(M[end], "Link")
    @assert length(rightlinks) ≤ 2
    looseright = getfirst(!=(leftlinkind(M, length(M))), rightlinks)

    return looseleft, looseright
end

"""
Combine loose links from the ends of a range of an MPS or MPO and contract into a
tensor.
   |     |          | |
- [_] - [_] -  =>  [___]-
   |     |          | |
"""
function combineoutsidelinks(M::ITensors.AbstractMPS)
    # left and right links may be `nothing`
    leftlink, rightlink = getlooselinks(M)
    linkcombiner = combiner(filter(!isnothing, [leftlink, rightlink])...)
    rhocomb = *(M..., linkcombiner)
    return rhocomb, linkcombiner
end

"""
Approximate a final MPO with a quantum channel applied to a initial MPO.
"""
function approxquantumchannel(init::MPO, final::MPO; nkraus::Union{Nothing,Int}=nothing, silent=false)
    sites = firstsiteinds(init)

    rhocomb, rholinkcomb = combineoutsidelinks(init)
    trunccomb, trunclinkcomb = combineoutsidelinks(final)

    ρ = toarray(rhocomb, sites, sites', combinedind(rholinkcomb))
    ρ̃ = toarray(trunccomb, sites, sites', combinedind(trunclinkcomb))
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃; nkraus, silent=true)

    # Transform Kraus operator into tensor
    krausidx = Index(last(size(Ks)), KRAUS_TAG)
    K = toITensor(Ks, sites', sites, krausidx)

    return K, optloss, initloss
end

@with_kw struct TruncParams
    nsites::Int
    bonddim::Int
    nkraussites::Int
    nsitesreduce::Int = 0
    nbondstrunc::Int
    truncatedbonddim::Int
    nkraus::Int
end

@with_kw struct TruncResults
    rho::MPO
    truncrho::MPO
    K::ITensor
    kraussites::Vector{Index}
    initloss::Float64
    optloss::Float64
    initdimstrunc::Vector{Int}
end

function runtruncationapprox(params::TruncParams)::TruncResults
    @unpack nsites, bonddim, nkraussites, nsitesreduce, nbondstrunc, truncatedbonddim, nkraus = params

    # Generate random density
    nallsites = nsites + 2 * nsitesreduce
    allsites = siteinds("Qubit", nallsites)
    psi = normalize(randomMPS(ComplexF64, allsites, linkdims=bonddim))
    fullrho = density_matrix(psi)

    # Reduce outside sites, keeping only nsites
    siterange = centerrange(nallsites, nsites)
    rho = reduced_density_matrix(fullrho, collect(siterange))
    sites = allsites[siterange]

    # Take range of sites in the middle of rho onto which to apply the Kraus operators
    krausrange = centerrange(nsites, nkraussites)
    kraussites = sites[krausrange]
    rhoslice = MPO(rho[krausrange]) # the first and last tensors have loose links

    # Make truncated density matrix
    truncrange = centerrange(nkraussites, nbondstrunc + 1)
    # save initial dimensions of truncated links
    linkstotrunc = linkind.(Ref(rhoslice), truncrange[1:end-1])
    initdimstrunc = NDTensors.dim.(linkstotrunc)
    # truncate() orthogonalizes the MPO, but that is ok because we completely contract the
    # MPOs before running optimization
    trunc = truncate(rhoslice; maxdim=truncatedbonddim, site_range=truncrange)

    # Find approximate quantum channel
    K, optloss, initloss = approxqcmpo(rhoslice, trunc; nkraus)

    return TruncResults(;
        rho=rhoslice,
        truncrho=trunc,
        K,
        kraussites,
        initloss,
        optloss,
        initdimstrunc
    )
end

"""
Approximate a given density matrix with a quantum channel applied to a initial density
matrix. This is done by using non-linear optimization to finding optimal Kraus
operators.

min{Kᵢ} ‖∑ᵢKᵢρKᵢ† - ρ̃‖₂
s.t.    ∑ᵢKᵢ†Kᵢ = I
"""
function approxquantumchannel(ρ::Array, ρ̃::Array; nkraus::Union{Nothing,Int}=nothing, silent=false)
    @assert size(ρ̃) == size(ρ) "Dimensions of ρ and ρ̃ must match"
    ndim = first(size(ρ))
    @assert ispow2(ndim) "Dimension of density matrix must be a power of 2"
    @assert ndims(ρ) in [2, 3] "Input must have 2 or 3 dimensions"

    # Make all inputs 3D tensors
    if ndims(ρ) == 2
        ρ = reshape(ρ, size(ρ)..., 1)
        ρ̃ = reshape(ρ̃, size(ρ)..., 1)
    end

    # Build Krauss operator variables
    model = Model(Ipopt.Optimizer)
    silent ? set_silent(model) : nothing
    # complex array variables are not currently supported, so have to reshape
    nqubits = floor(Int64, log(2, ndim))
    maxnkraus = (2^nqubits)^2
    nkraus = isnothing(nkraus) ? maxnkraus : nkraus
    Ksdims = (ndim, ndim, nkraus)
    # the optimizer needs help with starting from a feasible point, so we initialize with
    # valid complex-valued Kraus operators.
    # TODO: explore effect of initialization on optimizations
    σy = [0.0 -1.0; 1.0 0.0]
    # initKs = cat(repeat(sqrt(1 / nkraus) * 1 / sqrt(2) * (I + (1.0im * (σy ⊗ nqubits))), outer=[1, 1, nkraus]), dims=3)
    ident = Array(I, ndim, ndim)
    zero = zeros(ndim, ndim)
    initKs = cat(ident, repeat(zero, outer=[1, 1, nkraus - 1]), dims=3)
    Ks = reshape([
            @variable(model, set = ComplexPlane(), start = initKs[i, j, k])   #sqrt(1 / nkraus) * initK[i, j])
            for (i, j, k) in Tuple.(CartesianIndices(Ksdims))
        ], Ksdims)

    # Define Krauss operators contraint: ∑ᵢKᵢ†Kᵢ = I
    @constraint(model, sum(K' * K for K in eachslice(Ks, dims=3)) .== I)

    # Find the difference between the approximation and tsvd matrix and compute the
    # Frobenius norm: ∑ᵢKᵢρKᵢ† - ρ̃.
    approxs = [
        @expression(model, sum(K * ρi * K' for K in eachslice(Ks, dims=3)))
        for ρi in eachslice(ρ, dims=3)]
    diffs = [@expression(model, approxs[i] - ρ̃[:, :, i]) for i in 1:length(approxs)]

    # Compute the Frobenius norm. This will have quartic terms, so we have to use
    # NLexpression. NLexpression does not yet support complex variables :(
    flatdiffs = Iterators.flatten(diffs)
    diffelems = vcat(real.(flatdiffs), imag.(flatdiffs))
    obj = @NLexpression(model, sum(el^2 for el in diffelems))
    @NLobjective(model, Min, obj)

    # Setup callback to record optimization iteration data
    iterdata = []
    function recorditerdata(data...)
        push!(iterdata, data)
        return true
    end
    MOI.set(model, Ipopt.CallbackFunction(), recorditerdata)

    # Let's optimize!
    optimize!(model)

    # Calculate initial objective value for comparison
    initloss = sum(norm.(ρ - ρ̃) .^ 2)

    optloss = objective_value(model)

    return value.(Ks), optloss, initloss, iterdata, model
end

using ITensors
using LinearAlgebra
using ITensors.HDF5

include("circuit_elements.jl")
include("utilities.jl")
include("results.jl")

"""
Some notes about MPDOs from https://arxiv.org/pdf/1804.09796.pdf
    - Do not conserve positivity. Checking for positivity is an NP-hard problem
    - Alternative approaches: quantum trajectories and locally purified tensor networks (LPTNs)

"""

INNER_TAG = "Inner"
OUTER_TAG = "Link"
SITE_TAG = "Site"

mutable struct MPDO <: ITensors.AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

function MPDO(A::Vector{<:ITensor}; ortho_lims::UnitRange=1:length(A))
    return MPDO(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end


function MPDO(ψ::MPS)
    # Take the input state and add a dummy index (for now just dim=1)
    mpdo = Array{ITensor}(undef, length(ψ))
    for m in eachindex(ψ)
        M = ψ[m]
        new_inds = [inds(M)..., Index(1, "$(INNER_TAG),n=$(m)")]
        new_arr = reshape(array(M), size(M)..., 1)
        mpdo[m] = ITensor(new_arr, new_inds)
    end
    return MPDO(mpdo)
end


function ITensors.siteind(state::MPDO, i::Int)
    return getfirst(i -> hastags(i, SITE_TAG), inds(state[i]))
end

ITensors.siteinds(state::MPDO) = siteinds(first, state)

function ITensors.siteinds(::typeof(first), state::MPDO)
    return [siteind(state, i) for i in eachindex(state)]
end

function innerind(state::MPDO, i::Int)
    return getfirst(i -> hastags(i, INNER_TAG), inds(state[i]))
end

## Metrics

function LinearAlgebra.tr(state::MPDO, range)
    return tr(ITensor(1.0), state, range)
end

function LinearAlgebra.tr(curr::ITensor, state::MPDO, range)
    for i in range
        # contract next tensor into L one at a time to reduce intermediate size
        curr *= state[i]
        # prime the outer tags so that they are not contracted
        curr *= prime(dag(state[i]), tags=OUTER_TAG)
    end
    return curr
end

function LinearAlgebra.tr(A::MPDO)
    return real(scalar(tr(A, eachindex(A))))
end

"""
Given a slice of a MPDO, and contracted sites to the left and right of the slice, and two
pauli opertators, compute trace.
"""
function twosite_pauli_trace(L::ITensor, state::Vector{<:ITensor}, R::ITensor, leftpauli::ITensor, rightpauli::ITensor)
    # Contract pauli with tensor and adjoint
    L *= leftpauli * prime(state[1], tags=INNER_TAG)
    L *= prime(dag(state[1]))

    # Continue tracing out sites between the left and right sites
    for T in state[2:end-1]
        # contract next tensor into L one at a time to reduce intermediate size
        L *= T
        L *= prime(dag(T), tags=OUTER_TAG)
    end

    # Contract pauli with tensor and adjoint
    L *= rightpauli * prime(state[end], tags=INNER_TAG)
    L *= prime(dag(state[end]))

    # Finish by contracting the the traced out right tensor
    L *= R

    return (1 / 2^2) * scalar(L)
end

"""
Compute reduced density using two-site tomography. This may be more memory efficient than
using simple contraction.
"""
function twosite_tomography(L::ITensor, state::Vector{<:ITensor}, R::ITensor)
    # Find site indices
    lind = getfirst(i -> hastags(i, SITE_TAG), inds(state[1]))
    rind = getfirst(i -> hastags(i, SITE_TAG), inds(state[end]))

    # Build pauli basis
    leftpaulis = [ITensor(σᵢ, lind', lind) for σᵢ in paulis]
    rightpaulis = [ITensor(σᵢ, rind', rind) for σᵢ in paulis]
    paulipairs = Iterators.product(leftpaulis, rightpaulis)
    basis = Base.splat(*).(paulipairs)

    # Apply each pair of paulis to perform tomography
    apply_tom(paulipair) = twosite_pauli_trace(L, state, R, paulipair...)
    weights = apply_tom.(paulipairs)

    # Build reduced density
    ρ = sum(weights .* basis)
    # Swap primes on site indices to recover original prime order
    return swapprime(ρ, 0, 1)
end

"""
Compute reduced density by simply contracting from left to right, not tracing out the
site indices of the leftmost and rightmost sites.
"""
function twosite_reduced_density(L::ITensor, state::Vector{<:ITensor}, R::ITensor)
    L *= state[1]
    L *= prime(prime(dag(state[1]), tags=OUTER_TAG), tags=SITE_TAG)
    for T in state[2:end-1]
        # contract next tensor into L one at a time to reduce intermediate sizes
        L *= T
        L *= prime(dag(T), tags=OUTER_TAG)
    end
    L *= state[end]
    L *= prime(prime(dag(state[end]), tags=OUTER_TAG), tags=SITE_TAG)
    return L * R
end

function twosite_reduced_density(state::MPDO, site1::Int, site2::Int; tom=false)
    # Ensure left site < right site
    lsite, rsite = sort([site1, site2])
    @assert lsite !== rsite

    # Trace and contract sites left of left site and right of right site
    L = tr(state, 1:lsite-1)
    R = tr(state, length(state):-1:rsite+1)

    if tom
        return twosite_tomography(L, state[lsite:rsite], R)
    else
        return twosite_reduced_density(L, state[lsite:rsite], R)
    end
end

function von_neumann_entropy(state::MPDO, i)
    orthogonalize!(state, i)
    # TODO: do we need to put the inner index somewhere specific?
    U, S, V = svd(state[i], [linkind(state, i - 1), siteind(state, i)])
    SvN = 0.0
    for n = 1:ITensors.dim(S, 1)
        p = S[n, n]^2
        if p ≈ 0
            SvN -= 0
        else
            SvN -= p * log2(p)
        end
    end
    return SvN
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, M::MPDO)
    g = create_group(parent, name)
    attributes(g)["type"] = "MPDO"
    attributes(g)["version"] = 1
    N = length(M)
    write(g, "length", N)
    write(g, "rlim", M.rlim)
    write(g, "llim", M.llim)
    for n in 1:N
        write(g, "MPDO[$(n)]", M[n])
    end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{MPDO})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "MPDO"
        error("HDF5 group or file does not contain MPDO data")
    end
    N = read(g, "length")
    rlim = read(g, "rlim")
    llim = read(g, "llim")
    v = [read(g, "MPDO[$(i)]", ITensor) for i in 1:N]
    return MPDO(v, llim, rlim)
end



function make_kraus_operator(s, ε::Real)
    D = ITensors.dim(s)

    # Make the Kraus operators
    K_elems = [sqrt(1 - ε) * Id, sqrt(ε / 3) * σx, sqrt(ε / 3) * σy, sqrt(ε / 3) * σz]

    # Make the ITensor object
    Ks = []
    for i in eachindex(K_elems)
        push!(Ks, ITensor(K_elems[i], s, prime(s)))
    end

    return Ks
end

function make_kraus_operators(sites, ε::Real)
    Ks = []
    for s in sites
        push!(Ks, make_kraus_operator(s, ε))
    end
    return Ks
end

function apply_noise(ψ::MPDO, Ks; inner_dim::Union{Int,Nothing}=2)
    ψ̃ = copy(ψ)
    sites = siteinds(first, ψ)

    for j in eachindex(ψ)
        K = copy(Ks[j])
        T = copy(ψ[j])
        T = prime(T, sites[j])

        T̃ = T * K[1]
        inneridx1 = taginds(T̃, INNER_TAG)[1]
        inneridx2 = copy(inneridx1)
        for i in 2:length(K)
            TKi = T * K[i]
            T̃, inneridx2 = directsum(T̃ => inneridx2, TKi => inneridx1, tags="$(INNER_TAG),n=$(j)")
        end

        # Perform SVD on inner index
        U, S, V = ITensors.svd(T̃, uniqueinds(T̃, inneridx2), maxdim=inner_dim, righttags="$(INNER_TAG),n=$(j)")
        ψ̃[j] = U * S
    end

    return ψ̃
end

function apply_depolarizing_noise(ψ::MPDO, ε::Real; inner_dim::Union{Int,Nothing}=2)
    sites = siteinds(first, ψ)
    Ks = make_kraus_operators(sites, ε)
    return apply_noise(ψ, Ks; inner_dim)
end

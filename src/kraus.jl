
using ITensors

include("channel-analysis.jl")

const KRAUS_TAG = "Kraus"

function getkrausind(K)
    allinds = inds(K)
    return allinds[findfirst(ind -> hastags(ind, KRAUS_TAG), allinds)]
end

"""
Returns true is K is a valid Kraus tensor. Checks completeness.
"""
function isvalidkraus(K::ITensor, sites)
    # Put Kraus tensor into canonical form
    # K = getcanonicalkraus(K)
    krausidx = getkrausind(K)

    # Check completeness with tensors
    Kdag = swapprime(dag(K), 0 => 1) * δ(krausidx, krausidx')
    complete = apply(Kdag, K)
    delt = *([δ(ind, ind') for ind in sites]...)
    return complete ≈ delt
end

"""
Transform Kraus operator tensor to canonical form with SVD. The quantum channel defined
by the set of Kraus operators K_i is invariant under unitary transformation on the virtual
index. This allows us to SVD the Kraus tensor on the virtual index and drop the U tensor.
"""
function getcanonicalkraus(Ks)
    krausind = getkrausind(Ks)
    U, S, V = svd(Ks, krausind)
    sind = uniqueind(S, V)
    return δ(krausind, sind) * S * V
end

function dephasing_noise(sites, ε::Float64)
    CS = combiner(sites...) # make a combiner tensor for the inds
    cS = combinedind(CS) # make a new label for the combined indices

    # Make the kraus operators
    Id = Matrix(I, 2, 2)
    σx = [0.0 1.0
        1.0 0.0]
    σy = [0.0 -1.0im
        -1.0im 0.0]
    σz = [1.0 0.0
        0.0 -1.0]

    Ids = sqrt(1 - ε) .* copy(Id)
    σxs = sqrt(ε / 3) .* copy(σx)
    σys = sqrt(ε / 3) .* copy(σy)
    σzs = sqrt(ε / 3) .* copy(σz)

    for _ in 2:length(sites)
        # Build up the total operator
        Ids = Ids ⊗ Id
        σxs = σxs ⊗ σx
        σys = σys ⊗ σy
        σzs = σzs ⊗ σz
    end

    # Stack them together
    K_elems = cat(collect(Ids), collect(σxs), collect(σys), collect(σzs), dims=3)

    # Turn this into an ITensor with the appropriate indices
    sum_idx = Index(4, tags=KRAUS_TAG)
    KC = ITensor(K_elems, cS, prime(cS), sum_idx)
    K = KC * CS * prime(CS)
    return K
end

"""
Build depolarizing channel Kraus operator for single qubit site.
"""
function single_site_depolarizing_noise(site, ϵ)
    # Build pauli operators for the site: [I, σx, σy, σz]
    paulis = buildpaulibasis(site)

    # Scale paulis with given ϵ
    coeffI = sqrt(1 - 3ϵ / 4)
    coeffx = sqrt(ϵ) / 2
    coefs = [coeffI, coeffx, coeffx, coeffx]
    ops = coefs .* paulis

    # Combine operators into tensor
    krausind = Index(4, tags=KRAUS_TAG)
    K = sum(i -> ops[i] * onehot(krausind => i), eachindex(ops))
    return K
end

function depolarizing_noise(sites, ϵ)
    # Build depolarizing channel for each site
    Ks = single_site_depolarizing_noise.(sites, Ref(ϵ))

    # Make tensor product of single-site channels. Prime kraus inds to prevent contraction
    primedKs = [setprime(Ks[i], i - 1, tags=KRAUS_TAG) for i in eachindex(Ks)]
    krausinds = getkrausind.(primedKs)
    combinedkrausind = combiner(krausinds..., tags=KRAUS_TAG)
    K = *(primedKs...) * combinedkrausind
    return K
end

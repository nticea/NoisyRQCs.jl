
using ITensors
using LinearAlgebra
using Kronecker: ⊗

"""
Computed Frobenius norm of a tensor, keeping indices provided
"""
function frobneiusnorm(K, onindices...)
    allinds = collect(inds(K))
    indstonorm = allinds[allinds.∉Ref([Kindex])]

    # For some reason it's not happy two broadcasts at the same time :(
    normed = norm.(K)
    normsquared = normed .^ 2
    summed = normsquared * ITensor(1.0, indstonorm)

    return sqrt.(summed)
end

"""
Computes matrix of pauli decomposition coefficients of tensor on given indices and their primes
C_i = 1/n^2 Tr(Kᵢ ⋅ σᵃ ⊗ σᵇ)
"""
function getpaulicoeffs(K, sites, psites)
    # Pauli operators
    Id = [1.0 0.0; 0.0 1.0]
    σx = [0.0 1.0; 1.0 0.0]
    σy = [0.0 1.0im; -1.0im 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    paulis = [Id, σx, σy, σz]

    # Build pauli itensors for each site
    sitepaulis = [
        [ITensor(pauli, sites[i], psites[i]) for pauli in paulis]
        for i in eachindex(sites)
    ]

    # Build tensor products of all combinations of paulis across sites
    multisitepaulis = [*(ops...) for ops in Iterators.product(sitepaulis...)]

    # Compute pauli decomposition coefficients: C_i = 1/n^2 Tr(Kᵢ ⋅ σᵃ ⊗ σᵇ)
    # TODO: compute normalization for general number of sites, currently hardcoded for 2 sites
    Cs = 1 / 4 .* Ref(K) .* multisitepaulis

    return Cs, multisitepaulis
end

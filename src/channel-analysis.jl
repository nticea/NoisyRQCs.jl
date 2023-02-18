
using ITensors
using LinearAlgebra
using Kronecker: ⊗

"""
Computed Frobenius norm of a tensor, keeping indices provided
"""
function frobneiusnorm(K, onindices...)
    allinds = collect(inds(K))
    indstonorm = allinds[allinds.∉Ref(onindices)]

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
function paulidecomp(K, sites)
    # Pauli operators
    Id = Matrix(I, 2, 2)
    σx = [0.0 1.0
        1.0 0.0]
    σy = [0.0 -1.0im
        1.0im 0.0]
    σz = [1.0 0.0
        0.0 -1.0]
    paulis = [Id, σx, σy, σz]

    # Build pauli itensors for each site
    sitebases = [
        [ITensor(pauli, site, site') for pauli in paulis]
        for site in sites
    ]

    sitelabels = [
        [l for l in ["I", "x", "y", "z"]]
        for _ in eachindex(sites)
    ]

    # Build tensor products of all combinations of paulis across sites
    basis = [*(ops...) for ops in Iterators.product(sitebases...)]
    labels = [*(ops...) for ops in Iterators.product(sitelabels...)]

    # Compute pauli decomposition coefficients: C_i = 1/2^n Tr(Kᵢ ⋅ σᵃ ⊗ σᵇ)
    # where n is the number of sites.
    nsites = length(sites)

    Cs = (1 / 2^nsites) .* Ref(K) .* basis

    # Transpose basis for reconstruction
    recbasis = swapprime.(basis, Ref(1 => 0))

    return Cs, recbasis, labels
end

"""
Reshape a 2D matrix of 1D tensors to a 3D array
"""
function tensmatrix_to_arr(Ts::Matrix{ITensor})
    coeffsraw = array.(Ts) # Matrix{Vector}
    n3d = size(coeffsraw[1, 1])[1]
    return cat([getindex.(coeffsraw, Ref(i)) for i in 1:n3d]..., dims=3)
end

"""
Visualize Pauli decomposition of tensor on sites
"""
function visualize_paulidecomp(K, sites; title::String="Pauli Decomposition", clims=nothing)
    Cs, basis, labels = paulidecomp(K, sites)

    # Reshape to 3D array with Kraus index as 3rd dimension
    coeffsraw = array.(Cs) # Matrix{Vector}
    nkraus = size(coeffsraw[1, 1])[1]
    coeffarr = cat([getindex.(coeffsraw, Ref(i)) for i in 1:nkraus]..., dims=3)

    coeffnorms = norm.(coeffarr)

    ndims = size(coeffnorms)[1]
    ps = []
    for n in 1:nkraus
        p = heatmap(coeffnorms[:, :, n], aspect_ratio=:equal, clims=clims, c=:bluesreds, yflip=true)
        ann = [(j, i, text("$n" * labels[i, j], :white, :center)) for i in 1:ndims for j in 1:ndims]
        p = annotate!(p, ann, linecolor=:white, yflip=:true)
        push!(ps, p)
    end
    s = (500 * nkraus / 2, 1000)
    p = plot(ps...,
        layout=Plots.grid(2, floor(Int, nkraus / 2), widths=[1 / floor(Int, nkraus / 2) for _ in 1:floor(Int, nkraus / 2)]), size=s, plot_title=title)

    return p
end

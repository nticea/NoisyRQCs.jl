
using ITensors
using LinearAlgebra
using Kronecker: ⊗

"""
Computed Frobenius norm of a tensor, keeping indices provided
"""
function frobneiusnorm(K, onindices...)
    allinds = collect(inds(K))
    indstonorm = allinds[allinds.∉Ref(onindices)]

    # note: real() is used only to convert tensor type. norm() should already be real
    return sqrt.((real(norm.(K)) .^ 2) * ITensor(1.0, indstonorm))
end

function buildpaulibasis(site)
    Id = Matrix(I, 2, 2)
    σx = [0.0 1.0
        1.0 0.0]
    σy = [0.0 -1.0im
        1.0im 0.0]
    σz = [1.0 0.0
        0.0 -1.0]

    paulis = [Id, σx, σy, σz]

    return [ITensor(pauli, site, site') for pauli in paulis]
end

function paulibasislabels(n::Int)
    sitelabels = [
        [l for l in ["I", "x", "y", "z"]]
        for _ in eachindex(sites)
    ]

    return [*(ops...) for ops in Iterators.product(sitelabels...)]
end

"""
Computes matrix of pauli decomposition coefficients of tensor on given indices and their primes
C_i = 1/n^2 Tr(Kᵢ ⋅ σᵃ ⊗ σᵇ)
"""
function paulidecomp(K, sites)
    # Build pauli itensors for each site
    sitebases = buildpaulibasis.(sites)

    # Build tensor products of all combinations of paulis across sites
    basis = [*(ops...) for ops in Iterators.product(sitebases...)]

    # Compute pauli decomposition coefficients: C_i = 1/2^n Tr(Kᵢ ⋅ σᵃ ⊗ σᵇ)
    # where n is the number of sites.
    nsites = length(sites)

    Cs = (1 / 2^nsites) .* Ref(K) .* basis

    # Transpose basis for reconstruction
    recbasis = swapprime.(basis, Ref(1 => 0))

    labels = paulibasislabels(length(sites))

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
function visualize_kraus(K, sites; title::String="Pauli Decomposition", clims=nothing)
    pdecomp, basis, labels = paulidecomp(K, sites)
    pdnorms = norm.(pdecomp)
    plot_paulidecomp(pdecomp; clims)
end

function plot_paulidecomp(pdnorms; clims=nothing, title="Pauli Decomposition")
    nkraus = size(pdnorms)[3]
    ndimens = size(pdnorms)[1]
    nsites = Int(log(2, ndimens))
    labels = paulibasislabels(nsites)

    ps = []

    ny = 4
    nx = ceil(Int, nkraus / ny)

    height = 250 * ny
    width = 250 * nx + 20 * nx^2 + 20

    for n in 1:nkraus
        p = heatmap(
            pdnorms[:, :, n],
            c=:blues,
            clims=clims,
            title=n,
            framestyle=:none,
        )
        ann = [(j, i, text("$n" * labels[i, j], :white, :center, 8)) for i in 1:ndimens for j in 1:ndimens]
        p = annotate!(p, ann, linecolor=:white, yflip=true)
        push!(ps, p)
    end

    plot(
        ps...,
        size=(width, height),
        layout=(ny, nx),
        plot_title=title,
        padding=5 * Plots.mm
    )
end

"""
Run basic analysis on Kraus tensor
    - Pauli decomposition
    - Distribution of norms
"""
function analyzekraus(K, sites; usecanonical=true)
    # Transform into canonical form
    K = usecanonical ? getcanonicalkraus(K) : K

    # perform pauli decomposition
    pdtens, basis, lbls = paulidecomp(K, sites)
    pdarr = tensmatrix_to_arr(pdtens)

    # Get norm distribution
    normtensor = frobneiusnorm(K, getkrausind(K))
    norms = array(normtensor)
    relnorms = norms / sum(norms)

    return pdarr, relnorms
end

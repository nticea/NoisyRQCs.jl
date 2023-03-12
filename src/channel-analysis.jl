
using ITensors
using LinearAlgebra
using Plots
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
        for _ in 1:n
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

function plot_paulidecomp(pdnorms; zerolims=false, title="Pauli Decomposition", plotnorms=false)
    nkraus = size(pdnorms)[3]
    ndimens = size(pdnorms)[1]
    nsites = Int(log(2, ndimens))
    labels = paulibasislabels(nsites)

    ps = []

    nx = 4
    ny = ceil(Int, nkraus / nx)

    height = 250 * ny
    width = 250 * nx * 1.35

    for n in 1:nkraus
        data = round.(pdnorms[:, :, n], digits=5)
        maxval = maximum(data)
        minval = minimum(data)
        computedclims = zerolims ? (0, maxval) : (minval, maxval)
        Plots.gr_cbar_width[] = 0.005
        p = heatmap(
            data,
            c=:blues,
            clims=computedclims,
            title=n,
            framestyle=:none,
        )
        ann = [(j, i, text(labels[i, j], :white, :center, 8)) for i in 1:ndimens for j in 1:ndimens]
        p = annotate!(p, ann, linecolor=:white, yflip=true)
        push!(ps, p)
    end

    l = @layout [
        title{0.001h}
        grid(ny, nx)
    ]

    # show a barplot of matrix norms
    if plotnorms
        relnorms = [sqrt(sum(norm.(m) .^ 2)) for m in eachslice(pdnorms, dims=3)]
        normplot = bar(
            relnorms,
            titlefont=font(9),
            legend=:none
        )
        ps = vcat(normplot, ps)
        l = @layout [
            title{0.001h}
            n{0.11w} grid(ny, nx)
        ]
        width = width + 300
    end

    return plot(
        plot(title=title, grid=false, showaxis=false, ticks=false),
        ps...,
        size=(width, height),
        layout=l,
        topmargin=4 * Plots.mm
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
    pdtens, basis, labels = paulidecomp(K, sites)
    pdarr = tensmatrix_to_arr(pdtens)

    # Get norm distribution
    normtensor = frobneiusnorm(K, getkrausind(K))
    norms = array(normtensor)
    relnorms = norms / sum(norms)

    return pdarr, relnorms
end

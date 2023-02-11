
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
    sum_idx = Index(4, tags="Kraus")
    KC = ITensor(K_elems, cS, prime(cS), sum_idx)
    K = KC * CS * prime(CS)
    return K
end

function random_noise(sites, nkraus::Int)
    CS = combiner(sites...) # make a combiner tensor for the inds
    cS = combinedind(CS) # make a new label for the combined indices

    # Make the kraus operators
    ms = [rand(Complex{Float64}, 2, 2) for _ in 1:nkraus]

    for _ in 2:length(sites)
        # Build up the total operator
        for i in 1:length(ms)
            ms[i] = ms[i] ⊗ rand(Complex{Float64}, 2, 2)
        end
    end

    # Stack them together
    K_elems = cat([collect(m) for m in ms]..., dims=3)

    # Turn this into an ITensor with the appropriate indices
    sum_idx = Index(nkraus, tags="Kraus")
    KC = ITensor(K_elems, cS, prime(cS), sum_idx)
    K = KC * CS * prime(CS)
    return K
end

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

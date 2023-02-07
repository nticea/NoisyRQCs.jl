
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
function paulidecomp(K, sites)
    psites = prime.(sites) # we will work with the convention that MPO has primed inds on one side and unprimed on the other 

    # Pauli operators
    Id = Matrix(I, 2, 2)
    σx = [0.0 1.0
        1.0 0.0]
    σy = [0.0 -1.0im
        -1.0im 0.0]
    σz = [1.0 0.0
        0.0 -1.0]
    paulis = [Id, σx, σy, σz]

    # Build pauli itensors for each site
    sitebases = [
        [ITensor(pauli, sites[i], psites[i]) for pauli in paulis]
        for i in eachindex(sites)
    ]

    sitelabels = [
        [l for l in ["I", "x", "y", "z"]]
        for _ in eachindex(sites)
    ]

    # Build tensor products of all combinations of paulis across sites
    basis = [*(ops...) for ops in Iterators.product(sitebases...)]
    basis_labels = [*(ops...) for ops in Iterators.product(sitelabels...)]

    # Compute pauli decomposition coefficients: C_i = 1/2^n Tr(Kᵢ ⋅ σᵃ ⊗ σᵇ)
    # where n is the number of sites.
    nsites = length(sites)
    N = 2^nsites # defining some useful constants 
    @show inds(K)

    Cs = (1 / N) .* Ref(K) .* basis

    # for each Ki in the stack of Kraus operators, plot its projection onto the coefficients
    kraus_dim = ITensors.dim.(inds(Cs[1, 1]))[1] # the kraus dimension 
    K_projs_real = zeros(Float64, kraus_dim, N, N)
    K_projs_imag = zeros(Float64, kraus_dim, N, N)
    labels = ["" for _ in 1:kraus_dim, _ in 1:N, _ in 1:N]
    for c1 in 1:N
        for c2 in 1:N
            for i in 1:kraus_dim # iterate through the dimensions of the kraus operator 
                K_projs_real[i, c1, c2] = real(Cs[c1, c2][i])
                K_projs_imag[i, c1, c2] = imag(Cs[c1, c2][i])
                labels[i, c1, c2] = "K$(i)" * basis_labels[c1, c2]
            end
        end
    end

    return K_projs_real, K_projs_imag, labels
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

function visualize_paulidecom(K, sites; title::String="Pauli Decomposition", clims::Tuple{Real,Real}=(-1, 1))
    Kreal, Kimag, labels = paulidecomp(K, sites)
    numkraus = size(Kreal)[1]
    ndims = size(Kreal)[2]
    ps = []
    for n in 1:numkraus
        p = heatmap(Kreal[n, :, :] .^ 2 + Kimag[n, :, :] .^ 2, aspect_ratio=:equal, clim=clims, c=:bluesreds, yflip=true)
        ann = [(j, i, text(labels[n, i, j], :white, :center)) for i in 1:ndims for j in 1:ndims]
        p = annotate!(p, ann, linecolor=:white, yflip=:true)
        push!(ps, p)
    end
    s = (500 * nkraus / 2, 1000)
    p = plot(ps...,
        layout=Plots.grid(2, floor(Int, nkraus / 2), widths=[1 / floor(Int, nkraus / 2) for _ in 1:floor(Int, nkraus / 2)]), size=s, plot_title=title)

    return p
end

using ITensors

const KRAUS_TAG = "Kraus"

function getkrausind(K)
    allinds = inds(K)
    return allinds[findfirst(ind -> hastags(ind, KRAUS_TAG), allinds)]
end
s
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
    sum_idx = Index(nkraus, tags=KRAUS_TAG)
    KC = ITensor(K_elems, cS, prime(cS), sum_idx)
    K = KC * CS * prime(CS)
    return K
end

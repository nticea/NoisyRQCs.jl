
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors
using Plots
using Parameters

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/channel-analysis.jl")

@with_kw struct TruncParams
    nsites::Int
    bonddim::Int
    nkraussites::Int
    nbondstrunc::Int
    truncatedbonddim::Int
    nkraus::Int
end

@with_kw struct TruncResults
    reducedrho::MPO
    truncrho::MPO
    K::ITensor
    krausidx::Index
    kraussites::Vector{Index}
    lossratio::Float64
    initdimstrunc::Vector{Int}
end

function centerrange(n, size)
    first = (n - size) ÷ 2 + 1
    last = first + size - 1
    return first:last
end

params = TruncParams(;
    nsites=6,
    bonddim=100,
    nkraussites=2,
    nbondstrunc=1,
    truncatedbonddim=1,
    nkraus=4
)

function runtruncationapprox(params::TruncParams)
    @unpack nsites, bonddim, nkraussites, nbondstrunc, truncatedbonddim, nkraus = params

    # Generate random density
    sites = siteinds("Qubit", nsites)
    print("")
    psi = normalize(randomMPS(ComplexF64, sites, linkdims=bonddim))
    rho = density_matrix(psi)

    # Take subset of sites in the middle. Can choose to trace out other sites
    # shouldpartialtrace = false TODO: include non-partial trace case
    krausrange = centerrange(nsites, nkraussites)
    kraussites = sites[krausrange]
    reducedrho = reduced_density_matrix(rho, collect(krausrange))

    # Make truncated density matrix
    truncrange = centerrange(nkraussites, nbondstrunc + 1)
    truncrho = copy(reducedrho)
    NDTensors.truncate!(truncrho, maxdim=truncatedbonddim, site_range=truncrange)

    # Find approximate quantum channel
    ρ = Matrix(reducedrho)
    ρ̃ = Matrix(truncrho)
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃, nkraus=nkraus)

    lossratio = (initloss - optloss) / initloss
    @show lossratio

    # Transform Kraus operator into tensor
    krausidx = Index(last(size(Ks)), "Kraus")
    Kraw = toITensor(Ks, prime.(kraussites), kraussites, krausidx)
    K = getcanonicalkraus(Kraw, krausidx)

    return TruncResults(;
        reducedrho,
        truncrho,
        K,
        krausidx,
        kraussites,
        lossratio,
        initdimstrunc=[]
    )
end

res = runtruncationapprox(params)
@unpack K, krausidx, kraussites = res

# Get norm distribution. The U should be a diagonal matrix with entries with norm 1.
U, S, V = svd(K, krausidx)
mags = array(diag(S))
relmags = mags / sum(mags)
p = bar(
    relmags,
    ylim=[0, 1],
    ylabel="Relative magnitude",
    xlabel="Kraus operators",
    title="Relative magnitudes of Kraus operators",
    legend=:none,
)
display(p)

# Plot pauli decompositions
p = visualize_paulidecomp(K, kraussites, title="Pauli decomposition for truncation channel")
display(p)

"""
Ideas:

Average over many MPDOs

Look at:
- distribution of norms
- clustering of pauli decompositions vs random noise

optimal loss with 2 vs 4 site Kraus operators

difference in entanglement entropy between truncated and approximation

different truncation dimensions
"""


using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors
using Plots
using Parameters
using DataFrames
using StatsPlots

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/channel-analysis.jl")
include("../src/kraus.jl")

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

function runtruncationapprox(params::TruncParams)
    @unpack nsites, bonddim, nkraussites, nbondstrunc, truncatedbonddim, nkraus = params

    # Generate random density
    sites = siteinds("Qubit", nsites)
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
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃, nkraus=nkraus, silent=true)

    lossratio = (initloss - optloss) / initloss

    # Transform Kraus operator into tensor
    krausidx = Index(last(size(Ks)), KRAUS_TAG)
    K = toITensor(Ks, prime.(kraussites), kraussites, krausidx)

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

function buildnormsdf(normslist)
    nkraus = size(first(normslist))[1]
    cols = string.(collect(1:nkraus))
    normsdata = DataFrame([getindex.(normslist, i) for i in 1:nkraus], cols, copycols=false)
    normsdata.n = axes(normsdata, 1)
    normsdf = stack(normsdata, cols, variable_name="krausind", value_name="relnorm")
    normsdf.krausind = parse.(Int, normsdf.krausind)
    return normsdf
end

function plotnorms(normsdf, type::String)
    @df normsdf boxplot(
        :krausind,
        :relnorm,
        ylabel="Relative magnitude",
        xlabel="Kraus operators",
        title="Relative magnitudes of $type Kraus operators",
        legend=:none,
        titlefont=font(11),
    )
end

channelparams = TruncParams(;
    nsites=6,
    bonddim=100,
    nkraussites=2,
    nbondstrunc=1,
    truncatedbonddim=1,
    nkraus=4
)

# Truncation Channel
ntruncsamples = 200

pdecomps = []
relnorms = []
for _ in 1:ntruncsamples
    @unpack K, kraussites, lossratio = runtruncationapprox(channelparams)
    csarr, relmags = analyzekraus(K, kraussites)
    push!(pdecomps, csarr)
    push!(relnorms, relmags)
end

# Relative norms
relnormsdf = buildnormsdf(relnorms)
plotnorms(relnormsdf, "truncation-approximating")

# Pauli Decomposition
pdplots = []
pdarr = cat(pdecomps..., dims=4)
pdnorms = norm.(pdarr)
pdmean = mean(pdnorms, dims=4)
push!(pdplots, plot_paulidecomp(pdmean, title="Pauli decomposition norm means", clims=:auto))
push!(pdplots, plot_paulidecomp(pdmean, title="Pauli decomposition norm means (zeroed)"))
pdstd = std(pdnorms, dims=4)
push!(pdplots, plot_paulidecomp(pdstd, title="Pauli decomposition norm std", clims=:auto))
push!(pdplots, plot_paulidecomp(pdstd, title="Pauli decomposition norm std (zeroed)"))
pdstdratio = pdstd ./ pdmean
push!(pdplots, plot_paulidecomp(pdstdratio, title="Pauli decomposition norm std to mean ratio", clims=:auto))
push!(pdplots, plot_paulidecomp(pdstdratio, title="Pauli decomposition norm std to mean ratio (zeroed)"))

plot(
    pdplots...,
    layout=(length(pdplots), 1),
    size=(1450, length(pdplots) * 300)
)

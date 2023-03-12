
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors
using Plots
using Parameters
using DataFrames
using StatsPlots
using FileIO, JLD2

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/channel-analysis.jl")
include("../src/kraus.jl")

@with_kw struct TruncParams
    nsites::Int
    bonddim::Int
    nkraussites::Int
    nsitesreduce::Int = 0
    nbondstrunc::Int
    truncatedbonddim::Int
    nkraus::Int
end

@with_kw struct TruncResults
    rho::MPO
    truncrho::MPO
    K::ITensor
    kraussites::Vector{Index}
    initloss::Float64
    optloss::Float64
    initdimstrunc::Vector{Int}
end

function centerrange(n, size)
    first = (n - size) ÷ 2 + 1
    last = first + size - 1
    return first:last
end

function toarray(T::ITensor, indlists...)::Array
    combiners = combiner.(filter(!isnothing, indlists))
    flattened = *(T, combiners...)
    combinds = combinedind.(combiners)
    return array(flattened, combinds)
end

function toITensor(A, indlists...)::ITensor
    combiners = combiner.(filter(!isnothing, indlists))
    combinedinds = combinedind.(combiners)
    T = ITensor(A, combinedinds...)
    return *(T, combiners...)
end

"""
leftlinkind(M::MPS, j::Integer)
leftlinkind(M::MPO, j::Integer)
Get the link or bond Index connecting the MPS or MPO tensor on site j-1 to site j.
If there is no link Index, return `nothing`.
"""
function leftlinkind(M::ITensors.AbstractMPS, j::Integer)
    (j > length(M) || j < 2) && return nothing
    return commonind(M[j-1], M[j])
end

"""
Alias for linkind(M::MPS, j::Integer)
"""
const rightlinkind = linkind

"""
Finds links one either end of a MPS or MPO slice that are loose
      |     |
=> - [_] - [_] - <=
      |     |
"""
function getlooselinks(M::ITensors.AbstractMPS)
    leftlinks = taginds(M[1], "Link")
    @assert length(leftlinks) ≤ 2
    looseleft = getfirst(!=(rightlinkind(M, 1)), leftlinks)

    rightlinks = taginds(M[end], "Link")
    @assert length(rightlinks) ≤ 2
    looseright = getfirst(!=(leftlinkind(M, length(M))), rightlinks)

    return looseleft, looseright
end

"""
Combine loose links from the ends of a range of an MPS or MPO and contract into a
tensor.
   |     |          | |
- [_] - [_] -  =>  [___]-
   |     |          | |
"""
function combineoutsidelinks(M::ITensors.AbstractMPS)
    # left and right links may be `nothing`
    leftlink, rightlink = getlooselinks(M)
    linkcombiner = combiner(filter(!isnothing, [leftlink, rightlink])...)
    rhocomb = *(M..., linkcombiner)
    return rhocomb, linkcombiner
end

function approxqcmpo(init::MPO, final::MPO; nkraus=nothing)
    sites = firstsiteinds(init)

    rhocomb, rholinkcomb = combineoutsidelinks(init)
    trunccomb, trunclinkcomb = combineoutsidelinks(final)

    ρ = toarray(rhocomb, sites, sites', combinedind(rholinkcomb))
    ρ̃ = toarray(trunccomb, sites, sites', combinedind(trunclinkcomb))
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃; nkraus, silent=true)

    # Transform Kraus operator into tensor
    krausidx = Index(last(size(Ks)), KRAUS_TAG)
    K = toITensor(Ks, sites', sites, krausidx)

    return K, optloss, initloss
end

function runtruncationapprox(params::TruncParams)
    @unpack nsites, bonddim, nkraussites, nsitesreduce, nbondstrunc, truncatedbonddim, nkraus = params

    # Generate random density
    nallsites = nsites + 2 * nsitesreduce
    allsites = siteinds("Qubit", nallsites)
    psi = normalize(randomMPS(ComplexF64, allsites, linkdims=bonddim))
    fullrho = density_matrix(psi)

    # Reduce outside sites, keeping only nsites
    siterange = centerrange(nallsites, nsites)
    rho = reduced_density_matrix(fullrho, collect(siterange))
    sites = allsites[siterange]

    # Take range of sites in the middle of rho onto which to apply the Kraus operators
    krausrange = centerrange(nsites, nkraussites)
    kraussites = sites[krausrange]
    rhoslice = MPO(rho[krausrange]) # the first and last tensors have loose links

    # Make truncated density matrix
    truncrange = centerrange(nkraussites, nbondstrunc + 1)
    # save initial dimensions of truncated links
    linkstotrunc = linkind.(Ref(rhoslice), truncrange[1:end-1])
    initdimstrunc = NDTensors.dim.(linkstotrunc)
    # truncate() orthogonalizes the MPO, but that is ok because we completely contract the
    # MPOs before running optimization
    trunc = truncate(rhoslice; maxdim=truncatedbonddim, site_range=truncrange)

    # Find approximate quantum channel
    K, optloss, initloss = approxqcmpo(rhoslice, trunc; nkraus)

    return TruncResults(;
        rho=rhoslice,
        truncrho=trunc,
        K,
        kraussites,
        initloss,
        optloss,
        initdimstrunc
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

# Compute truncation channels
channelparams = TruncParams(;
    nsites=2,
    nsitesreduce=0,
    bonddim=100,
    nkraussites=2,
    nbondstrunc=1,
    truncatedbonddim=1,
    nkraus=4
)
ntruncsamples = 10
results = [runtruncationapprox(channelparams) for _ in 1:ntruncsamples]
# results = vcat(results, loadedresults) # uncomment to save previous results

# save results
datadir = "outputs"
datafilename = "rand-mpo-data-2-3.jld2"
savefile = joinpath(@__DIR__, "..", datadir, datafilename)
tosave = Dict("results" => results)
save(savefile, tosave)

# load data
savedata = load(savefile)
loadedresults = savedata["results"]

# Loss ratios
lossratios = [round(r.optloss, digits=5) / r.initloss for r in loadedresults]
bins = 0:0.01:1.0
histogram(
    lossratios,
    title="Loss ratios",
    normalize=:pdf,
    xlabel="Loss ratio",
    ylabel="%",
    legend=false;
    bins
)
savefig(joinpath(@__DIR__, "..", datadir, "loss"))


# run analysis
analyses = [analyzekraus(r.K, r.kraussites) for r in loadedresults]
pdecomps = [a[1] for a in analyses]
relnorms = [a[2] for a in analyses]

# Relative norms
relnormsdf = buildnormsdf(relnorms)
plotnorms(relnormsdf, "truncation-approximating")
savefig(joinpath(@__DIR__, "..", datadir, "norms"))

# Pauli Decomposition means
meanplots = []
pdarr = cat(pdecomps..., dims=4)
pdnorms = norm.(pdarr)
pdmean = mean(pdnorms, dims=4)
push!(meanplots, plot_paulidecomp(pdmean, title="Pauli decomposition norm means"))
push!(meanplots, plot_paulidecomp(pdmean, title="Pauli decomposition norm means (zeroed)", zerolims=true))
pdstd = std(pdnorms, dims=4)
push!(meanplots, plot_paulidecomp(pdstd, title="Pauli decomposition norm std"))
push!(meanplots, plot_paulidecomp(pdstd, title="Pauli decomposition norm std (zeroed)", zerolims=true))
pdstdratio = pdstd ./ pdmean
push!(meanplots, plot_paulidecomp(pdstdratio, title="Pauli decomposition norm std to mean ratio"))
push!(meanplots, plot_paulidecomp(pdstdratio, title="Pauli decomposition norm std to mean ratio (zeroed)", zerolims=true))

plot(
    meanplots...,
    layout=(length(meanplots), 1),
    size=(1400, 1.36 * length(meanplots) * 215)
)
savefig(joinpath(@__DIR__, "..", datadir, "means"))


# Samples
nsamples = 3
mininds = partialsortperm(lossratios, 1:nsamples)
maxinds = partialsortperm(lossratios, 1:nsamples, rev=true)
sampleinds = [mininds..., reverse(maxinds)...]
sampleplots = [
    plot_paulidecomp(
        norm.(pdecomps[i]),
        title="Sample $i: loss = $(round(lossratios[i], digits=4))",
        plotnorms=true
    )
    for i in sampleinds
]
plot(
    sampleplots...,
    layout=(length(sampleplots), 1),
    size=(2000, 1.6 * length(sampleplots) * 215)
)
savefig(joinpath(@__DIR__, "..", datadir, "samples"))

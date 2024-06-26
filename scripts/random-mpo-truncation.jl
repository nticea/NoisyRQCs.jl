
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

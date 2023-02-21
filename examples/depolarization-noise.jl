
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

function plotnorms(normsdf, type::String)
    @df normsdf bar(
        :krausind,
        :relnorm,
        ylabel="Relative Frobenius norm",
        xlabel="Kraus operators",
        title="Relative Frobenius norms of $type Kraus operators",
        titlefont=font(11),
        legend=:none,
    )
end

# Depolarizing Channel
nsites = 2
ϵ = 0.1
sites = siteinds("Qubit", nsites)
K = depolarizing_noise(sites, ϵ)
pdecomp, relnorms = analyzekraus(K, sites, usecanonical=true)

# Kraus operator norms
normsdf = DataFrame(krausind=eachindex(relnorms), relnorm=relnorms)
plotnorms(normsdf, "depolarizing")

# Kraus operator Pauli decomposition
pdnorms = norm.(pdecomp)
plot_paulidecomp(pdnorms)

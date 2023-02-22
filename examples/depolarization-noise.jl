
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

# Depolarizing Channel
nsites = 2
ϵ = 0.1
sites = siteinds("Qubit", nsites)
K = depolarizing_noise(sites, ϵ)
pdecomp, relnorms = analyzekraus(K, sites, usecanonical=true)

# Kraus operator norms
bar(
    relnorms,
    ylabel="Relative Frobenius norm",
    xlabel="Kraus operators",
    title="Relative Frobenius norms of depolarizing Kraus operators",
    titlefont=font(11),
    legend=:none,
)

# Kraus operator Pauli decomposition
pdnorms = norm.(pdecomp)
plot_paulidecomp(pdnorms)

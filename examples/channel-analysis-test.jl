using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/channel-analysis.jl")

# Build Kraus operator for testing
nsites = 2
nkraus = (2^nsites)^2
bonddim = 8
sites = siteinds("S=1/2", nsites)
Kindex = Index(nkraus, "K index")
psites = prime(sites)
K = randomITensor(ComplexF64, sites, psites, Kindex)

Cs, ops = paulidecomp(K, sites, psites)

norms = frobneiusnorm(K, Kindex)

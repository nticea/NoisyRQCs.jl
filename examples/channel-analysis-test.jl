using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/channel-analysis.jl")

# Build Kraus operator for testing
nsites = 2
nkraus = nsites^2
bonddim = 8
sites = siteinds("S=1/2", nsites)
Kindex = Index(nkraus, "K index")
psites = prime(sites)
K = randomITensor(ComplexF64, sites, psites, Kindex)

Cs, ops = getpaulicoeffs(K, sites, psites)

norms = frobneiusnorm(K, Kindex)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/channel-analysis.jl")

# Build Kraus operator for testing
nsites = 2
nkraus = (2^nsites)^2
bonddim = 8
sites = siteinds("S=1/2", nsites)
krausidx = Index(nkraus, "Kraus")
K = randomITensor(ComplexF64, sites, prime(sites), krausidx)

K_projs_real, K_projs_imag, labels = paulidecomp(K, sites)

norms = frobneiusnorm(K, krausidx)

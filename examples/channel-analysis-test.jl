using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/channel-analysis.jl")

# Build Kraus operator for testing
nsites = 1
nkraus = (2^nsites)^2
bonddim = 8
sites = siteinds("S=1/2", nsites)
krausidx = Index(nkraus, "Kraus")
K = randomITensor(ComplexF64, sites, prime(sites), krausidx)

Cs, basis, labels = paulidecomp(K, sites)

# Check that we can reconstruct the Kraus tensor
reconstruction = sum(Cs .* basis)
@assert reconstruction ≈ K

norms = frobneiusnorm(K, krausidx)

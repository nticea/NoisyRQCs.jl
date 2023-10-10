
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")

"""
Tests the channel approximation function.
"""

# 1. generate random density matrices
n = 3
nsites = 2
bonddim = 8
sites = siteinds("S=1/2", nsites)
psis = [randomMPS(sites, bonddim) for i in 1:n]
rhos = density_matrix.(psis)

# 2. Make truncated density matrix
truncatedbonddim = 1
trhos = truncate.(rho; maxdim=Ref(truncatedbonddim))

# 3. Find approximate quantum channel
ρ = cat(toarray.(rhos, Ref(sites),  Ref(sites'))..., dims=3)
ρ̃ = cat(toarray.(trhos, Ref(sites),  Ref(sites'))..., dims=3)
Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃)

@show Ks
@show (initloss - optloss) / initloss

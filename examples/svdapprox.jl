
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")

"""
Computed a quantum channel that best approximates bond truncation between two sites of a
random spin-1/2 MPS
"""

# 1. generate random density matrix
nsites = 2
bonddim = 8
sites = siteinds("S=1/2", nsites)
psi = randomMPS(sites, bonddim)
rho = density_matrix(psi)

# 2. Make truncated density matrix
truncatedbonddim = 2
trho = copy(rho)
NDTensors.truncate!(trho, maxdim=truncatedbonddim)

# 3. Find approximate quantum channel
ρ = Matrix(rho)
ρ̃ = Matrix(trho)
Ks, iterdata, model = approxquantumchannel(ρ̃, ρ)

initialobjvalue = sum((ρ - ρ̃) .^ 2)
@show initialobjvalue
@show Ks
@show last(iterdata)

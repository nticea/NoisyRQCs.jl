## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 20
ε = 0
maxdim = nothing
truncdim = 1

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

# Apply the MPDO circuit
ψ = apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim, benchmark=true)
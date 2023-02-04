## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")
include("../src/results.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 100
ε = 0.001
maxdim = 16
max_inner_dim = 2

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

# Apply the MPDO circuit
ψ, state_entanglement, op_entanglement, trace = apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim, max_inner_dim=max_inner_dim, benchmark=true)

plot_entropy(state_entanglement, L)

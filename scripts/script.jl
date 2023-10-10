## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 20
ε = 1e-3
χ = nothing
κ = 4

ψ0 = initialize_wavefunction(L=L)
ψ, _, _, _, _, _ = apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=χ,
    max_inner_dim=κ, benchmark=false, normalize_ρ=true,
    checkpoint_path=nothing, save_path=nothing,
    tensors_path=nothing)

# For each pair of sites, extract the two-site tensors 
# Orthogonalize about site i 
# Combine the physical and link indices 
# Initialize U₁
# Construct the environment tensor 
# SVD the environment tensor 
# Update U 
# Continue iterating 






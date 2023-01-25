## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include("../src/lagrange.jl")
include("../src/utilities.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 20
ε = 0
maxdim = nothing
truncdim = 1

# Initialize the wavefunction to be all zeros 
ψ0 = initialize_wavefunction(L=L)

# Apply the circuit 
ρ, SR2, SvN_op, t  = apply_circuit_truncation_channel(ψ0, T, truncdim, ε=ε, benchmark=true, maxdim=maxdim)

## We are not computing the right pseudoinverse! it should be a pseudoinverse
# for every value of the input and output link indices 
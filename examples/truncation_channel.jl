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
ρ, all_Ks, all_loss_hist = apply_circuit_truncation_channel(ψ0, T, truncdim, ε=ε, maxdim=maxdim)

p = plot()
cmap = cgrad(:acton, length(all_loss_hist), categorical = true)
for (i,lh) in enumerate(all_loss_hist)
    plot!(p, lh, c=cmap[i], label="t=$(i)")
end
title!(p, "Training loss")
xlabel!(p, "Iteration")
plot(p)
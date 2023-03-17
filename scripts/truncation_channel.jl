## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/lagrange.jl")
include("../src/utilities.jl")
include("../src/channel-analysis.jl")
include("../src/kraus.jl")

using Plots
using Images, ImageTransformations
using Random

ITensors.set_warn_order(50)
Random.seed!(12345)

## PARAMETERS ##
L = 9
T = 20
ε = 0
maxdim = nothing
truncdim = 1
truncidx = floor(Int, L / 2)
nkraus = 4

# Initialize the wavefunction to be all zeros
ψ0 = initialize_wavefunction(L=L)
sites = physical_indices(ψ0)[truncidx:truncidx+1]

# Apply the circuit
ρ, all_Ks, all_optloss, all_initloss, all_loss_hist = apply_circuit_truncation_channel(ψ0, T, truncdim, truncidx, nkraus, ε=ε, maxdim=maxdim)

p = plot()
cmap = cgrad(:acton, length(all_loss_hist), categorical=true)
for (i, lh) in enumerate(all_loss_hist)
    plot!(p, lh, c=cmap[i], legend=false)
end
title!(p, "Training loss")
xlabel!(p, "Iteration")
plot(p)

## FOR REFERENCE -- construct various types of noise ##
Kdephasing = dephasing_noise(sites, 0.5)
p_dephasing = visualize_paulidecomp(Kdephasing, sites, title="Pauli decomposition for dephasing noise")
plot(p_dephasing)

## Random noise
Krandom = random_noise(sites, 4)
p_random = visualize_paulidecomp(Krandom, sites, title="Pauli decomposition for random noise")
plot(p_random)

K = all_Ks[3]
p_K = visualize_paulidecomp(K, sites, title="Pauli decomposition for truncation channel", clims=(-0.1, 0.1))
plot(p_K)

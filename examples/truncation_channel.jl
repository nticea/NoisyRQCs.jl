## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/lagrange.jl")
include("../src/utilities.jl")
include("../src/channel-analysis.jl")

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
nkraus = 2

# Initialize the wavefunction to be all zeros 
ψ0 = initialize_wavefunction(L=L)

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
sites = physical_indices(ρ)[truncidx:truncidx+1]
Kdephasing = dephasing_noise(sites, 0.5)
Kdephasing_projs_real, Kdephasing_projs_imag, labels = paulidecomp(Kdephasing, sites)

Krandom = random_noise(sites, 4)
Krandom_projs_real, Krandom_projs_imag, labels = paulidecomp(Krandom, sites)

## DEPHASING NOISE ## 
ps = [heatmap(Kdephasing_projs_imag[n, :, :], aspect_ratio=:equal, clim=(-1, 1), c=:bluesreds, yflip=true) for n in 1:4]
p = plot(ps...,
    layout=Plots.grid(2, 2, widths=[1 / 2 for _ in 1:2]), size=(1000, 1000))

## RANDOM NOISE ##
ps = [heatmap(Krandom_projs_real[n, :, :], aspect_ratio=:equal, clim=(-1, 1), c=:bluesreds, yflip=true) for n in 1:4]
p = plot(ps...,
    layout=Plots.grid(2, 2, widths=[1 / Nsqrt for _ in 1:2]), size=(1000, 1000))

## TRUNCATION CHANNEL APPROXIMATION ## 
K = all_Ks[19]
K_projs_real, K_projs_imag, labels = paulidecomp(K, sites)

ps = [heatmap(K_projs_real[n, :, :], aspect_ratio=:equal, clim=(-1, 1), c=:bluesreds, yflip=true) for n in 1:nkraus]
p = plot(ps...,
    layout=Plots.grid(2, floor(Int, nkraus / 2), widths=[1 / floor(Int, nkraus / 2) for _ in 1:floor(Int, nkraus / 2)]), size=(500 * nkraus / 2, 1000))


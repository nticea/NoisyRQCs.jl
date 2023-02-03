## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/lagrange.jl")
include("../src/utilities.jl")

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

# Initialize the wavefunction to be all zeros 
ψ0 = initialize_wavefunction(L=L)

# Apply the circuit 
ρ, all_Ks, all_optloss, all_initloss, all_loss_hist = apply_circuit_truncation_channel(ψ0, T, truncdim, ε=ε, maxdim=maxdim)

p = plot()
cmap = cgrad(:acton, length(all_loss_hist), categorical=true)
for (i, lh) in enumerate(all_loss_hist)
    plot!(p, lh, c=cmap[i], legend=false)
end
title!(p, "Training loss")
xlabel!(p, "Iteration")
plot(p)

Δloss = convert.(Float64, log10.(cat(all_initloss, all_optloss, dims=2)))[2:end, :]'
plot(Δloss, legend=false, xticks=(1:2, ["Initial loss", "Final loss"]))
scatter!(Δloss, legend=false)
ylabel!("Loss (log10)")


# function perfract(x, t, m=0.7, M=1)
#     x = x / t
#     return m + (M - m) * (x - floor(x))
# end

# function domcol(w; n=10)
#     logm = log.(abs.(w)) # for lines of constant modulus
#     H = angle.(w) * 180 / π #compute argument of  w within interval [-180, 180], iei the Hue

#     V = perfract.(logm, 2π / n) # lines of constant log-modulus
#     arr = permutedims(cat(H, ones(size(H)), V, dims=3), [3, 1, 2]) #HSV-array

#     return RGB.(colorview(HSV, arr[:, end:-1:1, :]))
# end

# idx = 10
# s = 1
# Ks = all_Ks[idx]
# Ki = Ks[:, :, s]
# domcol(Ki)
# img = domcol(Ki; n=13)

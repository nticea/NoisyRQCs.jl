## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include("../src/circuit.jl")
include("../src/utilities.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 5
T = 100
ε = 0
nsamples = 50

## CODE ## 

"""
Checks to do:
    2. Run forward and then backwards, make sure i am getting out the same initial state
    4. Compute the second Renyi entropy 

TODO
    0.5 Finish reading the papers!! 
    1. Implement negativity 
    2. Implement MPO entanglement entropy
"""
bitdist = zeros(nsamples, 4^L)
entropy = zeros(nsamples, T)
for n in 1:nsamples
    print(n,"-")
    # Initialize the wavefunction to be all zeros 
    ψ0 = initialize_wavefunction(L=L)

    # Apply the circuit 
    ρ, S = apply_circuit(ψ0, T, ε=ε, benchmark=true)

    # get the distribution over bitstrings
    bdist = bitstring_distribution(ρ)

    bitdist[n,:] = bdist
    entropy[n,:] = S
end

_porter_thomas_fit(vec(bitdist), 4^L, true)
entropy_avg = vec(mean(entropy, dims=1))
plot_entropy(entropy_avg)

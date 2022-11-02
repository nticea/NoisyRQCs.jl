## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include("../src/circuit.jl")
include("../src/utilities.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 15 
T = 10

## CODE ## 

# Initialize the wavefunction to be all zeros 
ψ0 = initialize_wavefunction(L=L)

# Apply the circuit 
ρ = apply_circuit(ψ0, T)

"""
Checks to do:
    1. Output of the circuit when we are not applying any noise. Do we get the expected Haar statistics
    (Porter-Thomas distribution) 
    2. Run forward and then backwards, make sure i am getting out the same initial state
    3. Look at the growth of entanglement entropy over time 
    4. Compute the second Renyi entropy 
    2. Check -- are we applying two-qubit noise, or just single-qubit noise? 
        Single site noise is okay 

TODO
    0.5 Finish reading the papers!! 
    MAKE U DAGGER !!! 
    1. Implement negativity 
    2. Implement MPO entanglement entropy

"""
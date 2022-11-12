## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,"../.."))
include("../../src/circuit.jl")
include("../../src/utilities.jl")
using Dates
ITensors.set_warn_order(50)

## SAVING OUT ## 
do_save = true

## PARAMETERS ## 
L = 11
T = 100
ε = 0
nsamples = 5

## CODE ## 
for _ in 1:nsamples

    # Saving info 
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")  
    paramstamp = "$(L)L_$(T)T"
    save_path = joinpath(@__DIR__,timestamp*"_"*paramstamp*".h5")
    println("Saving to", save_path)

    # Initialize the wavefunction to be all zeros 
    ψ0 = initialize_wavefunction(L=L)

    # Apply the circuit 
    ρ, entropy = apply_circuit(ψ0, T, ε=ε, measure_entropy=true)
    bitdist = bitstring_distribution(ρ)

    # Save the results 
    results = Results(L, T, ρ, bitdist, entropy)
    if do_save 
        save_structs(results, save_path)
    end

end

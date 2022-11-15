## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,"../.."))
include("../../src/circuit.jl")
include("../../src/utilities.jl")
using Dates
ITensors.set_warn_order(50)

## SAVING OUT ## 
do_save = true
save_ρ = false

## PARAMETERS ## 
L = 25 
T = 100
ε = 0.1

## CODE ## 

# Saving info 
timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")  
paramstamp = "$(L)L_$(T)T_$(ε)ε"
save_path = joinpath(@__DIR__,timestamp*"_"*paramstamp*".h5")
println("Saving to", save_path)

# Initialize the wavefunction to be all zeros 
ψ0 = initialize_wavefunction(L=L)

# Apply the circuit 
ρ, entropy = apply_circuit(ψ0, T, ε=ε, benchmark=false)
bitdist = bitstring_distribution(ρ)

# Save the results 
if save_ρ
    results = Results(L, T, ρ, bitdist, entropy)
else
    results = Results(L, T, nothing, bitdist, entropy)
end

if do_save 
    save_structs(results, save_path)
end



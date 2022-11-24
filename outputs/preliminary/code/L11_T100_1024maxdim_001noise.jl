## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,"../../.."))
using Dates
include("../../../src/circuit.jl")
include("../../../src/utilities.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 11
T = 100
ε = 0.01
nsamples = 10
maxdim = nothing

## SAVING INFO ## 
save_ρ = false
do_save = true 

## CODE ## 

for n in 1:nsamples
    print(n,"-")
    # Initialize the wavefunction to be all zeros 
    ψ0 = initialize_wavefunction(L=L)

    # Apply the circuit 
    ρ, SR2, SvN_op, t  = apply_circuit(ψ0, T, ε=ε, benchmark=true, maxdim=maxdim)

    # get the distribution over bitstrings
    bdist = bitstring_distribution(ρ)

    # Save the results 
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")  
    paramstamp = "$(L)L_$(T)T_$(ε)ε_$(maxdim)maxdim"
    save_path = joinpath(@__DIR__,"../results",timestamp*"_"*paramstamp*".h5")
    println("Saving to", save_path)
    if save_ρ
        results = Results(L, T, ρ, bdist, SR2, SvN_op, t)
    else
        results = Results(L, T, 0, bdist, SR2, SvN_op, t)
    end

    if do_save 
        save_structs(results, save_path)
    end
end
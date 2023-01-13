## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include("../src/circuit.jl")
include("../src/utilities.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 100
ε = 1e-5
maxdims = [nothing]
nsamples = length(maxdims)

## SAVING INFO ## 
save_ρ = false
do_save = false 
save_path = @__DIR__

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
state_entropy = zeros(nsamples, T)
operator_entanglement = zeros(nsamples, T, L-3)
trace = zeros(nsamples, T)

for (n,maxdim) in enumerate(maxdims)
    print(n,"-")
    # Initialize the wavefunction to be all zeros 
    ψ0 = initialize_wavefunction(L=L)

    # Apply the circuit 
    ρ, SR2, SvN_op, t  = apply_circuit(ψ0, T, ε=ε, benchmark=true, maxdim=maxdim)

    # get the distribution over bitstrings
    bdist = bitstring_distribution(ρ)

    # Save the results 
    if save_ρ
        results = Results(L, T, ρ, bdist, SR2, SvN_op, t)
    else
        results = Results(L, T, 0, bdist, SR2, SvN_op, t)
    end

    if do_save 
        save_structs(results, save_path)
    end

    # record the results 
    bitdist[n,:] = bdist
    state_entropy[n,:] = SR2
    operator_entanglement[n,:,:] = SvN_op
    trace[n,:] = t
end

state_entropy_avg = vec(mean(state_entropy, dims=1))
operator_entanglement_avg = mean(operator_entanglement, dims=1)[1,:,:]
midpoint_op_entanglement = operator_entanglement_avg[:, floor(Int, (L-3)/2)]
trace_avg = vec(mean(trace, dims=1))

# make some plots 
_porter_thomas_fit(vec(bitdist), 2^L, true)
plot_entropy(state_entropy_avg, L)
plot(midpoint_op_entanglement)
plot(operator_entanglement_avg[end,:])



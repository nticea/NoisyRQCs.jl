using ITensors
using Distributions
using LinearAlgebra
using StatsBase
include("utilities.jl")
include("circuit_elements.jl")

"""
Make a layer of noise gates. Act on each site with this gateset between unitary evolution 
"""
function noise_layer(sites, ε::Real)
    gates = []
    for s in sites
        push!(gates, make_kraus_gate(s, ε))
    end
    return gates
end

"""
Apply a random circuit to the wavefunction ψ0
"""
function apply_circuit(ψ0::MPS, T::Int; random_type="Haar", ε=0, benchmark=false,
    maxdim=nothing, disentangler_channel::Bool=false, nkraus::Int=4, normalize_ρ::Bool=true,
    checkpoint_path::Union{String,Nothing}, save_path::Union{String,Nothing})

    # check whether there exists a checkpointed MPDO
    if checkpointed(checkpoint_path)
        return apply_circuit_mpo_checkpointed(checkpoint_path=checkpoint_path, save_path=save_path, random_type=random_type, benchmark=benchmark, normalize_ρ=normalize_ρ)
    end

    L = length(ψ0)
    if isnothing(maxdim)
        println("No truncation")
    else
        println("Truncating at m=$(maxdim)")
    end

    ρ = density_matrix(copy(ψ0)) # Make the density matrix 
    sites = siteinds(ψ0)

    state_entanglement = zeros(Float64, T)
    operator_entanglement = zeros(Float64, T, L - 3)
    trace = zeros(Float64, T)
    lognegs = zeros(Float64, T)
    MIs = zeros(Float64, T, L)

    # Iterate over all time steps 
    for t in 1:T
        # print results
        print(t, "-")
        flush(stdout)

        # benchmarking 
        if benchmark

            # trace
            trace[t] = real.(tr(ρ))
            @show trace[t], maxlinkdim(ρ)

            # Calculate the second Renyi entropy (state entanglement)
            ρ_A = reduced_density_matrix(ρ, collect(1:floor(Int, L / 2)))
            SR2 = second_Renyi_entropy(ρ_A)
            state_entanglement[t] = real(SR2)
            if normalize_ρ
                state_entanglement[t] *= trace[t]
            end

            # Calculate the operator entropy
            Ψ = combine_indices(ρ)
            SvN = []
            Threads.@threads for b in 2:(L-2)
                push!(SvN, entanglement_entropy(Ψ, b=b))
            end
            operator_entanglement[t, :] = SvN

            # Compute the logarithmic negativity
            lognegs[t] = logarithmic_negativity(ρ, collect(1:floor(Int, L / 2)))

            # mutual information 
            A = 1
            Threads.@threads for B in collect(2:L)
                ρA, ρB, ρAB = twosite_reduced_density_matrix(ρ, A, B)

                # Compute the mutual information 
                MIs[t, B] = mutual_information(ρA, ρB, ρAB)
            end

            # update the results 
            if !isnothing(save_path)
                results = Results(0, L, T, ε, maxdim, 0, state_entanglement, operator_entanglement, trace, lognegs, MIs)
                save_structs(results, save_path)
            end

        else
            # still need to keep track of the trace somehow 
            trace[t] = -1
        end

        # At each time point, make a layer of random unitary gates 
        unitary_gates = unitary_layer(sites, t, random_type)

        # Now apply the gates to the wavefunction (alternate odd and even) 
        Threads.@threads for u in unitary_gates
            ρ = apply_twosite_gate(ρ, u, maxdim=maxdim)
        end

        # Make the noise gates for this layer 
        noise_gates = noise_layer(sites, ε)

        Threads.@threads for n in noise_gates # Now apply the noise layer 
            ρ = apply_onesite_gate(ρ, n)
        end

        # save results
        if !isnothing(checkpoint_path)
            results = Results(ρ, L, T, ε, maxdim, 0, state_entanglement, operator_entanglement, trace, lognegs, MIs)
            save_structs(results, checkpoint_path)
        end
    end

    if benchmark
        return ρ, state_entanglement, operator_entanglement, lognegs, MIs, trace
    end

    return ρ
end

function apply_circuit_mpo_checkpointed(; checkpoint_path::String, save_path::Union{String,Nothing}=nothing, random_type::String="Haar", benchmark::Bool=false, normalize_ρ::Bool=false)
    # extract the results from the checkpointed path
    results = load_results(checkpoint_path, load_MPO=true)
    ρ, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs = splat_struct(results)
    T0 = findfirst(x -> x == 0, trace) # this is how far we've simulated already 
    sites = physical_indices(ρ)

    # if we've already evolved this wavefunction all the way through, do nothing 
    if isnothing(T0)
        if benchmark
            return ρ, state_entanglement, operator_entanglement, lognegs, MIs, trace
        end

        return ρ
    end

    # Iterate over all time steps 
    for t in T0:T
        # print results
        print(t, "-")
        flush(stdout)

        # benchmarking 
        if benchmark
            # trace
            trace[t] = real.(tr(ρ))
            @show trace[t], maxlinkdim(ρ)

            # Calculate the second Renyi entropy (state entanglement)
            ρ_A = reduced_density_matrix(ρ, collect(1:floor(Int, L / 2)))
            SR2 = second_Renyi_entropy(ρ_A)
            state_entanglement[t] = real(SR2)
            if normalize_ρ
                state_entanglement[t] *= trace[t]
            end

            # Calculate the operator entropy
            Ψ = combine_indices(ρ)
            SvN = []
            Threads.@threads for b in 2:(L-2)
                push!(SvN, entanglement_entropy(Ψ, b=b))
            end
            operator_entanglement[t, :] = SvN

            # Compute the logarithmic negativity
            lognegs[t] = logarithmic_negativity(ρ, collect(1:floor(Int, L / 2)))

            # mutual information 
            A = 1
            Threads.@threads for B in collect(2:L)
                ρA, ρB, ρAB = twosite_reduced_density_matrix(ρ, A, B)

                # Compute the mutual information 
                MIs[t, B] = mutual_information(ρA, ρB, ρAB)
            end

            # update the results 
            if !isnothing(save_path)
                results = Results(0, L, T, ε, maxdim, 0, state_entanglement, operator_entanglement, trace, lognegs, MIs)
                save_structs(results, save_path)
            end

        else
            # still need to keep track of the trace somehow 
            trace[t] = -1
        end

        # At each time point, make a layer of random unitary gates 
        unitary_gates = unitary_layer(sites, t, random_type)

        # Now apply the gates to the wavefunction (alternate odd and even) 
        Threads.@threads for u in unitary_gates
            ρ = apply_twosite_gate(ρ, u, maxdim=maxdim)
        end

        # Make the noise gates for this layer 
        noise_gates = noise_layer(sites, ε)

        Threads.@threads for n in noise_gates # Now apply the noise layer 
            ρ = apply_onesite_gate(ρ, n)
        end

        # save results
        if !isnothing(checkpoint_path)
            results = Results(ρ, L, T, ε, maxdim, 0, state_entanglement, operator_entanglement, trace, lognegs, MIs)
            save_structs(results, checkpoint_path)
        end
    end

    if benchmark
        return ρ, state_entanglement, operator_entanglement, lognegs, MIs, trace
    end

    return ρ

end
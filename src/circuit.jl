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
function apply_circuit(ψ0::MPS, T::Int; random_type="Haar", ε=0.05, benchmark=false, maxdim=nothing)
    L = length(ψ0)
    if isnothing(maxdim)
        println("No truncation")
    else
        println("Truncating at m=$(maxdim)")
    end

    ρ = density_matrix(copy(ψ0)) # Make the density matrix 
    sites = siteinds(ψ0)

    if benchmark
        state_entanglement = zeros(Float64, T)
        operator_entanglement = zeros(Float64, T, L - 3)
        trace = zeros(Float64, T)
        lognegs = zeros(Float64, T, L)
        MIs = zeros(Float64, T, L)
    end

    # Iterate over all time steps 
    for t in 1:T
        print(t, "-")

        # benchmarking 
        if benchmark
            # The maximum link dimension
            @show maxlinkdim(ρ)

            # Calculate the second Renyi entropy (state entanglement)
            ρ_A = reduced_density_matrix(ρ, collect(1:floor(Int, L / 2)))
            SR2 = second_Renyi_entropy(ρ_A)
            state_entanglement[t] = real(SR2)

            # Calculate the operator entropy
            Ψ = combine_indices(ρ)
            SvN = []
            for b in 2:(L-2)
                push!(SvN, entanglement_entropy(Ψ, b=b))
            end
            operator_entanglement[t, :] = SvN

            # trace
            trace[t] = real.(tr(ρ))
            @show trace[t]

            # logarithmic negativity and mutual information 
            A = 1
            for B in collect(2:L)
                ρAB = twosite_reduced_density_matrix(ρ, A, B)
                ρA = reduced_density_matrix(ρ, [A])
                ρB = reduced_density_matrix(ρ, [B])

                @show inds(ρAB)
                @show inds(ρA)
                @show inds(ρB)
                @assert 1 == 0

                # Compute the logarithmic negativity
                lognegs[t, B] = logarithmic_negativity(ρAB, [1])

                # Compute the mutual information 
                MIs[t, B] = mutual_information(ρA, ρB, ρAB)

            end
            @show lognegs[t, :]
            @show MIs[t, :]
        end

        # At each time point, make a layer of random unitary gates 
        unitary_gates = unitary_layer(sites, t, random_type)

        # Now apply the gates to the wavefunction (alternate odd and even) 
        for u in unitary_gates
            ρ = apply_twosite_gate(ρ, u, maxdim=maxdim)
        end

        # Make the noise gates for this layer 
        noise_gates = noise_layer(sites, ε)

        # Now apply the noise layer 
        for n in noise_gates
            ρ = apply_onesite_gate(ρ, n)
        end
    end

    @show tr(ρ)

    if benchmark
        return ρ, state_entanglement, operator_entanglement, lognegs, MIs, trace
    end

    return ρ
end
include("circuit.jl")
include("utilities.jl")
include("approxchannel.jl")

using ITensors
using LinearAlgebra
using Kronecker
using JuMP, Ipopt


"""
Apply a random circuit to the wavefunction ψ0
"""
function apply_circuit_truncation_channel(ψ0::MPS, T::Int, truncdim::Int, truncidx::Int, nkraus::Int; random_type="Haar", ε=0.05, maxdim=nothing)
    if isnothing(maxdim)
        println("No truncation")
    else
        println("Truncating at m=$(maxdim)")
    end

    ρ = density_matrix(copy(ψ0)) # Make the density matrix
    sites = siteinds(ψ0)

    # Iterate over all time steps
    all_Ks = []
    all_loss_hist = []
    all_optloss = []
    all_initloss = []
    for t in 1:T
        print(t, "-")

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

        # Perform the optimization 
        Ks, optloss, initloss, loss_hist = truncation_quantum_channel_rdm(ρ, truncdim, truncidx, nkraus, apply_gate=false)

        # record the data 
        push!(all_Ks, Ks)
        push!(all_loss_hist, loss_hist)
        push!(all_optloss, optloss)
        push!(all_initloss, initloss)
    end

    @show tr(ρ)

    return ρ, all_Ks, all_optloss, all_initloss, all_loss_hist
end

function truncation_quantum_channel(ρ::MPO, truncdim::Int, truncidx::Int, nkraus::Int; apply_gate::Bool=false)
    ρ = copy(ρ)
    sites = physical_indices(ρ) # qubit sites unprimed 
    sL = noprime(sites[truncidx]) # site on the left 
    sR = noprime(sites[truncidx+1]) # site on the right 

    # Orthogonalize the MPS around this site 
    orthogonalize!(ρ, truncidx)
    @show linkdim(ρ, truncidx)

    # snip two sites out of ρ
    ρ_ij = ρ[truncidx] * ρ[truncidx+1]

    ## IGNORE THIS ## 
    if apply_gate
        println("Applying gate")
        g = make_unitary_gate(sL, sR, "Haar")
        ρ_ij *= g
        # Lower the prime level by 1 to get back to what we originally had
        ρ_ij = replaceprime(ρ_ij, 3 => 1)
        ρ_ij = replaceprime(ρ_ij, 2 => 0)
    end

    # SVD the resulting tensor 
    inds3 = uniqueinds(ρ[truncidx], ρ[truncidx+1])
    U, S, V = ITensors.svd(ρ_ij, inds3, maxdim=truncdim, lefttags="Link,l=$(truncidx-1)", righttags="Link,l=$(truncidx+1)")

    # Create the target tensor 
    ρ̃_ij = U * S * V

    ## Tie the indices together and make sure all the indices are the same ##

    # Extract the link indices 
    lL = taginds(ρ_ij, "Link,l=$(truncidx-1)")
    rL = taginds(ρ_ij, "Link,l=$(truncidx+1)")
    @assert length(lL) > 0 && length(rL) > 0

    # These combiners will tie the indices together 
    cL = combiner(lL, rL)
    cX = combiner(sL, sR)
    cX1 = combiner(prime(sL), prime(sR))
    iL = combinedind(cL)
    iX = combinedind(cX)
    iX1 = combinedind(cX1)

    # apply the combiners 
    ρ_ij = ρ_ij * cL * cX * cX1
    ρ̃_ij = ρ̃_ij * cL * cX * cX1

    # permute the indices 
    ρ_ij = permute(ρ_ij, iX, iX1, iL)
    ρ̃_ij = permute(ρ̃_ij, iX, iX1, iL)

    # find the nearest CPTP map
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(array(ρ_ij), array(ρ̃_ij), nkraus=nkraus)
    # objective value is the 3rd entry
    loss_hist = map(x -> x[3], iterdata)

    # Turn the Ks into ITensors 
    krausidx = Index(size(Ks)[3], tags="Kraus")
    Ks = ITensor(Ks, iX, iX1, krausidx)

    # We want to SVD along the Kraus dimension to pick out components
    KKdag = Ks * dag(prime(Ks, iX, iX1))
    U, S, V = svd(KKdag, prime(iX), prime(iX1), righttags="Kraus")

    # Finally, decombine the indices 
    Ks = V * cX * cX1

    return Ks, optloss, initloss, loss_hist
end

function truncation_quantum_channel_rdm(ρ::MPO, truncdim::Int, truncidx::Int, nkraus::Int; apply_gate::Bool=false)
    ρ = copy(ρ)

    # Take the reduced density matrix
    rdm = reduced_density_matrix(ρ, [truncidx, truncidx + 1])
    sinds = siteinds(rdm)
    sL = unique(noprime(sinds[1]))
    sR = unique(noprime(sinds[2]))
    Linds = uniqueinds(rdm[1], rdm[2])

    @show maxlinkdim(rdm)

    # This is the starting density matrix
    ρtr = (rdm[1] * rdm[2])

    if apply_gate
        println("Applying gate")
        g = make_unitary_gate(sL, sR, "Haar")
        ρtr *= g
        # Lower the prime level by 1 to get back to what we originally had
        ρtr = replaceprime(ρtr, 3 => 1)
        ρtr = replaceprime(ρtr, 2 => 0)
    end

    # Now create the target tensor
    Ũ, S̃, Ṽ = ITensors.svd(ρtr, Linds, maxdim=truncdim)
    ρ̃tr = Ũ * S̃ * Ṽ

    # tie indices together
    cX = combiner(sL, sR)
    cX1 = combiner(prime(sL), prime(sR))
    iX = combinedind(cX)
    iX1 = combinedind(cX1)
    ρtr = ρtr * cX * cX1
    ρ̃tr = ρ̃tr * cX * cX1

    # permute the indices 
    ρtr = permute(ρtr, iX, iX1)
    ρ̃tr = permute(ρ̃tr, iX, iX1)

    # find the nearest CPTP map
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(array(ρtr), array(ρ̃tr), nkraus=nkraus)
    # objective value is the 3rd entry
    loss_hist = map(x -> x[3], iterdata)

    # Turn the Ks into ITensors 
    krausidx = Index(size(Ks)[3], tags="Kraus")
    Ks = ITensor(Ks, iX, iX1, krausidx)

    # We want to SVD along the Kraus dimension to pick out components
    KKdag = Ks * dag(prime(Ks, iX, iX1))
    U, S, V = svd(KKdag, prime(iX), prime(iX1), righttags="Kraus")

    # Finally, decombine the indices 
    Ks = V * cX * cX1

    return Ks, optloss, initloss, loss_hist
end

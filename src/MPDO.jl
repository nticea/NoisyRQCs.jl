using ITensors
using LinearAlgebra

include("circuit_elements.jl")
include("utilities.jl")

"""
Some notes about MPDOs from https://arxiv.org/pdf/1804.09796.pdf 
    - Do not conserve positivity. Checking for positivity is an NP-hard problem
    - Alternative approaches: quantum trajectories and locally purified tensor networks (LPTNs)

"""

struct MPDO <: ITensors.AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

function MPDO(ψ::MPS)
    return MPDO(ψ.data, ψ.llim, ψ.rlim)
end

function density_matrix_mpdo(ψ::MPS)
    # create a new density matrix MPO 
    sites = physical_indices(ψ)
    ρ = randomMPO(sites)
    orthogonalize!(ρ, 1)
    orthogonalize!(ψ, 1)

    # make the combiners 
    Cinds = []
    for n in 1:length(ψ)-1
        link_ind = taginds(ψ[n], "Link,l=$(n)")
        push!(Cinds, combiner(link_ind, prime(link_ind), tags="Link,l=$(n)"))
    end

    # Iterate through all the sites and construct the corresponding density matrix matrices 
    for n in 1:length(ψ)
        A = ψ[n]
        Adag = dag(prime(A, sites[n]))
        rind = taginds(A, "Link,l=$(n)")
        lind = taginds(A, "Link,l=$(n-1)")

        # make combiners 
        if length(rind) > 0
            Rind = Cinds[n] #combiner(rind, prime(rind), tags="Link,l=$(n)")
            Adag = prime(Adag, rind)
        end
        if length(lind) > 0
            Lind = Cinds[n-1] #combiner(lind, prime(lind), tags="Link,l=$(n-1)")
            Adag = prime(Adag, lind)
        end

        # Multiply A with A*
        AAdag = A * Adag

        # Apply combiner 
        if length(rind) > 0
            AAdag = AAdag * Rind
        end
        if length(lind) > 0
            AAdag = AAdag * Lind
        end

        # Update ρ
        ρ[n] = AAdag

    end

    return ρ
end

function make_kraus_operator(s, ε::Real)
    D = ITensors.dim(s)

    # Make the Kraus operators
    Id = sqrt(1 - ε) * Matrix(I, D, D)
    σx = sqrt(ε / 3) * [0.0 1.0
        1.0 0.0]
    σy = sqrt(ε / 3) * [0.0 -1.0im
        -1.0im 0.0]
    σz = sqrt(ε / 3) * [1.0 0.0
        0.0 -1.0]
    # Stack them together 
    K_elems = [Id, σx, σy, σz]

    # Make the ITensor object 
    Ks = []
    for i in 1:length(K_elems)
        push!(Ks, ITensor(K_elems[i], s, prime(s)))
    end

    return Ks
end

function make_kraus_operators(sites, ε::Real)
    Ks = []
    for s in sites
        push!(Ks, make_kraus_operator(s, ε))
    end
    return Ks
end

function apply_noise_mpdo(ψ::MPS, Ks; inner_dim::Union{Int,Nothing}=1)
    ψ̃ = copy(ψ)
    sites = physical_indices(ψ)

    for m in 1:length(ψ)
        K = Ks[m]
        M = ψ[m]
        M = prime(M, sites[m])

        M̃ = M * K[1]
        inneridx1 = taginds(M̃, "Inner")[1]
        inneridx2 = copy(inneridx1)
        for i in 2:length(K)
            MKi = M * K[i]
            M̃, inneridx2 = directsum(M̃ => inneridx2, MKi => inneridx1, tags="Inner,n=$(m)")
        end

        # perform truncation 
        @show inds(M̃)
        linds = linkindT(M̃)
        MMdag = M̃ * prime(dag(M̃), sites[m], linds...)
        if !isnothing(inner_dim)
            # V is the new value for M 
            U, S, V = ITensors.svd(MMdag, prime(sites[m]), prime.(linds)..., maxdim=inner_dim, righttags="Inner,n=$(m)")
            @show inds(V)
            ψ̃[m] = V
        end
    end

    return ψ̃
end

function apply_circuit_mpdo(ψ::MPS, T::Int; maxdim::Union{Nothing,Int}=nothing,
    max_inner_dim::Union{Nothing,Int}=nothing, random_type::String="Haar",
    ε::Real=0.05, benchmark::Bool=false)

    # Housekeeping 
    L = length(ψ)
    sites = siteinds(ψ)
    if isnothing(maxdim)
        println("No truncation")
    else
        println("Truncating at m=$(maxdim)")
        if !isnothing(max_inner_dim)
            @assert max_inner_dim <= 2 * maxdim^2
        end
    end

    # Make the noise gates for this layer 
    Ks = make_kraus_operators(sites, ε)

    # For benchmarking 
    if benchmark
        state_entanglement = zeros(Float64, T)
        operator_entanglement = zeros(Float64, T, L - 3)
        trace = zeros(Float64, T)
    end

    ## Transform ψ into an MPDO ##
    # Take the input state and add a dummy index (for now just dim=1) 
    for m in 1:length(ψ)
        M = ψ[m]
        new_inds = [inds(M)..., Index(1, "Inner,n=$(m)")]
        new_arr = reshape(array(M), size(M)..., 1)
        ψ[m] = ITensor(new_arr, new_inds)
    end

    for t in 1:T
        print(t, "-")

        # benchmarking 
        if benchmark
            # Convert MPDO into density matrix 
            ρ = density_matrix_mpdo(ψ)

            # The maximum link dimension
            @show maxlinkdim(ρ)

            # Calculate the second Renyi entropy (state entanglement)
            ρ_A = reduced_density_matrix(ρ, collect(1:floor(Int, L / 2)))
            SR2 = second_Renyi_entropy(ρ_A)
            state_entanglement[t] = real(SR2)
            @show real(SR2)

            # Calculate the operator entropy
            Ψ = combine_indices(ρ)
            SvN = []
            for b in 2:(L-2)
                push!(SvN, entanglement_entropy(Ψ, b=b))
            end
            operator_entanglement[t, :] = SvN
            @show SvN

            # trace
            trace[t] = real(tr(ρ))
            @show trace[t]
        end

        ## Apply a layer of unitary evolution to the MPS ##
        # At each time point, make a layer of random unitary gates 
        unitary_gates = unitary_layer(sites, t, random_type)

        # Now apply the gates to the wavefunction (alternate odd and even) 
        for u in unitary_gates
            ψ = apply_twosite_gate(ψ, u, maxdim=maxdim)
        end

        # Apply the noise layer 
        #ψ = apply_noise_mpdo(ψ, Ks)

        # Perform a canonicalization from L → R 

        # Truncate the virtual bonds

        # Truncate the inner indices 
    end

    return ψ

end

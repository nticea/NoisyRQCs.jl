using ITensors
using LinearAlgebra

include("circuit_elements.jl")
include("utilities.jl")
include("results.jl")

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

function apply_noise_mpdo(ψ::MPS, Ks; inner_dim::Union{Int,Nothing}=2)
    ψ̃ = copy(ψ)
    sites = physical_indices(ψ)

    for j in 1:length(ψ)
        K = copy(Ks[j])
        T = copy(ψ[j])
        T = prime(T, sites[j])

        T̃ = T * K[1]
        inneridx1 = taginds(T̃, "Inner")[1]
        inneridx2 = copy(inneridx1)
        for i in 2:length(K)
            TKi = T * K[i]
            T̃, inneridx2 = directsum(T̃ => inneridx2, TKi => inneridx1, tags="Inner,n=$(j)")
        end

        # Perform SVD on inner index
        U, S, V = ITensors.svd(T̃, uniqueinds(T̃, inneridx2), maxdim=inner_dim, righttags="Inner,n=$(j)")
        ψ̃[j] = U * S
    end

    return ψ̃
end

function build_mpdo(ψ::MPS)
    # Take the input state and add a dummy index (for now just dim=1)
    mpdo = Array{ITensor}(undef, length(ψ))
    for m in eachindex(ψ)
        M = ψ[m]
        new_inds = [inds(M)..., Index(1, "Inner,n=$(m)")]
        new_arr = reshape(array(M), size(M)..., 1)
        mpdo[m] = ITensor(new_arr, new_inds)
    end
    return MPS(mpdo)
end


function apply_circuit_mpdo_checkpointed(; checkpoint_path::String, save_path::Union{String,Nothing}=nothing, tensors_path::Union{String,Nothing}=nothing, random_type::String="Haar", benchmark::Bool=false, normalize_ρ::Bool=false)

    # extract the results from the checkpointed path
    results = load_results(checkpoint_path, load_MPS=true)
    ψ, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs = splat_struct(results)
    T0 = findfirst(x -> x == 0, trace) # this is how far we've simulated already

    # prepare the noise gates
    sites = siteinds(ψ)
    Ks = make_kraus_operators(sites, ε)

    # if we've already evolved this wavefunction all the way through, do nothing
    if isnothing(T0)
        return ψ, state_entanglement, operator_entanglement, lognegs, MIs, trace
    end

    for t in T0:T
        # print results
        print(t, "-")
        flush(stdout)

        # benchmarking
        if benchmark
            # Convert MPDO into density matrix
            ρ = density_matrix_mpdo(ψ)

            # trace
            trace[t] = real(tr(ρ))
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
            for b in 2:(L-2)
                push!(SvN, entanglement_entropy(Ψ, b=b))
            end
            operator_entanglement[t, :] = SvN

            # Compute the logarithmic negativity
            lognegs[t] = logarithmic_negativity(ρ, collect(1:floor(Int, L / 2)))

            # mutual information
            A = 1
            for B in collect(2:L)
                ρA, ρB, ρAB = twosite_reduced_density_matrix(ρ, A, B)

                # Compute the mutual information
                MIs[t, B] = mutual_information(ρA, ρB, ρAB)
            end

            # update the results
            if !isnothing(save_path)
                results = Results(0, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs)
                save_structs(results, save_path)
            end

        else
            # still need to keep track of the trace
            trace[t] = real(tr(ρ))
            @show trace[t], maxlinkdim(ρ)
        end

        ## Apply a layer of unitary evolution to the MPS ##

        # At each time point, make a layer of random unitary gates
        unitary_gates = unitary_layer(sites, t, random_type)

        for u in unitary_gates
            ψ = apply_twosite_gate(ψ, u, maxdim=maxdim)
        end

        # Apply the noise layer
        ψ = apply_noise_mpdo(ψ, Ks, inner_dim=max_inner_dim)

        # save results
        if !isnothing(checkpoint_path)
            results = Results(ψ, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs)
            save_structs(results, checkpoint_path)
        end

        if !isnothing(tensors_path)
            tpath = joinpath(tensors_path, "_$(T)T.h5")
            results = Results(ψ, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs)
            save_structs(results, tpath)
        end
    end

    return ψ, state_entanglement, operator_entanglement, lognegs, MIs, trace
end

function apply_circuit_mpdo(ψ::MPS, T::Int; maxdim::Union{Nothing,Int}=nothing,
    max_inner_dim::Union{Nothing,Int}=nothing, random_type::String="Haar",
    ε::Real=0, benchmark::Bool=false, normalize_ρ::Bool=false,
    checkpoint_path::Union{String,Nothing}=nothing, save_path::Union{String,Nothing}=nothing,
    tensors_path::Union{String,Nothing}=nothing)

    # check whether there exists a checkpointed MPDO
    if checkpointed(checkpoint_path)
        return apply_circuit_mpdo_checkpointed(checkpoint_path=checkpoint_path, save_path=save_path, tensors_path=tensors_path, random_type=random_type, benchmark=benchmark, normalize_ρ=normalize_ρ)
    end

    # Housekeeping
    L = length(ψ)
    sites = siteinds(ψ)
    if isnothing(maxdim)
        println("No truncation")
        maxdim = 2^((L - 1) / 2) # accounts for the fact that the MPDO bonds are doubled relative to the MPS bonds
    else
        println("Truncating at m=$(maxdim)")
        if !isnothing(max_inner_dim)
            @assert max_inner_dim <= 2 * maxdim^2
        end
    end

    # Make the noise gates for this layer
    Ks = make_kraus_operators(sites, ε)

    # For benchmarking
    state_entanglement = zeros(Float64, T)
    operator_entanglement = zeros(Float64, T, L - 3)
    trace = zeros(Float64, T)
    lognegs = zeros(Float64, T)
    MIs = zeros(Float64, T, L)

    ## Transform ψ into an MPDO ##
    # Take the input state and add a dummy index (for now just dim=1)
    for m in 1:length(ψ)
        M = ψ[m]
        new_inds = [inds(M)..., Index(1, "Inner,n=$(m)")]
        new_arr = reshape(array(M), size(M)..., 1)
        ψ[m] = ITensor(new_arr, new_inds)
    end

    for t in 1:T
        # print results
        print(t, "-")
        flush(stdout)

        # benchmarking
        if benchmark
            # Convert MPDO into density matrix
            println("Making density matrix")
            @time ρ = density_matrix_mpdo(ψ)

            # trace
            println("Calculating trace")
            @time trace[t] = real(tr(ρ))
            @show trace[t], maxlinkdim(ρ)

            # Calculate the second Renyi entropy (state entanglement)
            println("Making reduced density matrix")
            @time ρ_A = reduced_density_matrix(ρ, collect(1:floor(Int, L / 2)))

            println("Calculating second renyi entropy")
            @time SR2 = second_Renyi_entropy(ρ_A)
            state_entanglement[t] = real(SR2)
            if normalize_ρ
                state_entanglement[t] *= trace[t]
            end

            # Calculate the operator entropy
            println("Combining indices")
            @time Ψ = combine_indices(ρ)
            SvN = entanglement_entropy(Ψ, b=floor(Int, L / 2))
            operator_entanglement[t, floor(Int, L / 2)] = SvN

            ## THIS IS A COSTLY OPERATION, SO DON'T COMPUTE FOR EVERY BOND
            # SvN = []
            # for b in 2:(L-2)
            #     println("Computing entanglement entropy for bond $b")
            #     @time push!(SvN, entanglement_entropy(Ψ, b=b))
            # end
            # operator_entanglement[t, :] = SvN

            # Compute the logarithmic negativity
            println("Calculating logarithmic negativity")
            @time lognegs[t] = logarithmic_negativity(ρ, collect(1:floor(Int, L / 2)))

            # mutual information
            A = 1
            for B in collect(2:L)
                println("Making reduced density matrix with sites $A, $B")
                @time ρA, ρB, ρAB = twosite_reduced_density_matrix(ρ, A, B)

                # Compute the mutual information
                println("Computing MI of sites $A, $B")
                MIs[t, B] = mutual_information(ρA, ρB, ρAB)
            end

            # update the results
            if !isnothing(save_path)
                results = Results(0, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs)
                save_structs(results, save_path)
            end

        else
            # still need to keep track of the trace somehow
            trace[t] = -1
        end

        ## Apply a layer of unitary evolution to the MPS ##

        # At each time point, make a layer of random unitary gates
        unitary_gates = unitary_layer(sites, t, random_type)

        for u in unitary_gates
            ψ = apply_twosite_gate(ψ, u, maxdim=maxdim)
        end

        # if multithread
        #     ψnew = copy(ψ)
        #     Threads.@threads for u in unitary_gates
        #         ρL, ρR, cL, cR = apply_twosite_gate_multithread(ψ, u, maxdim=maxdim)
        #         ψnew[cL] = ρL
        #         ψnew[cR] = ρR
        #     end
        #     ψ = copy(ψnew)

        # else
        #     for u in unitary_gates
        #         ψ = apply_twosite_gate(ψ, u, maxdim=maxdim)
        #     end
        # end

        # Apply the noise layer
        ψ = apply_noise_mpdo(ψ, Ks, inner_dim=max_inner_dim)

        # save results
        if !isnothing(checkpoint_path)
            results = Results(ψ, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs)
            save_structs(results, checkpoint_path)
        end

        if !isnothing(tensors_path)
            tpath = joinpath(tensors_path, "state_$(t)T.h5")
            println("Saving data at timestep $t at $tpath")
            results = Results(ψ, L, T, ε, maxdim, max_inner_dim, state_entanglement, operator_entanglement, trace, lognegs, MIs)
            save_structs(results, tpath)
        end
    end

    return ψ, state_entanglement, operator_entanglement, lognegs, MIs, trace

end

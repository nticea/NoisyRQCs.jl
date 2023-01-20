include("circuit.jl")
include("utilities.jl")

using ITensors
using LinearAlgebra
using Kronecker
using NLsolve
using Einsum 
using Convex, SCS # convex solvers
using ADNLPModels
using NLPModelsIpopt
using JuMP 
using Ipopt


"""
Apply a random circuit to the wavefunction ψ0
"""
function apply_circuit_truncation_channel(ψ0::MPS, T::Int; random_type="Haar", ε=0.05, benchmark=false, maxdim=nothing)
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
        operator_entanglement = zeros(Float64, T, L-3)
        trace = zeros(Float64, T)
    end
    
    # Iterate over all time steps 
    for t in 1:T
        print(t,"-")

        # benchmarking 
        if benchmark
            # The maximum link dimension
            @show maxlinkdim(ρ)

            # Calculate the second Renyi entropy (state entanglement)
            ρ_A = partial_trace(ρ, collect(1:floor(Int, L/2)))
            SR2 = second_Renyi_entropy(ρ_A)
            state_entanglement[t] = real(SR2)

            # Calculate the operator entropy
            Ψ = combine_indices(ρ)
            SvN = []
            for b in 2:(L-2)
                push!(SvN, entanglement_entropy(Ψ, b=b))
            end
            operator_entanglement[t,:] = SvN
            
            # trace
            trace[t] = real.(tr(ρ))
            @show trace[t]
        end

        # At each time point, make a layer of random unitary gates 
        unitary_gates = unitary_layer(sites, t, random_type)

        # Now apply the gates to the wavefunction (alternate odd and even) 
        for u in unitary_gates
            ρ = truncation_channel(ρ, u, maxdim=maxdim)
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
        return ρ, state_entanglement, operator_entanglement, trace 
    end

    return ρ
end

"""
Apply a quantum channel ("gate") to a density matrix 
"""
function truncation_channel(ρ::MPO, G::ITensor; maxdim=nothing)
    ρ̃ = copy(ρ)
    L = length(ρ)

    # Extract the common indices where we will be applying the channel 
    c = findall(x -> hascommoninds(G, ρ[x]), collect(1:length(ρ)))
    @assert length(c) == 2
    c1, c2 = c

    # Orthogonalize the MPS around this site 
    orthogonalize!(ρ,c1)

    # Apply the gate 
    ρ2 = (ρ̃[c1] * ρ̃[c2]) * G

    # Lower the prime level by 1 to get back to what we originally had 
    ρ2 = replaceprime(ρ2, 3 => 1, tags="Site")
    ρ2 = replaceprime(ρ2, 2 => 0, tags="Site")

    # SVD the resulting tensor 
    Linds = uniqueinds(ρ̃[c1], ρ̃[c2])
    # If maxdim is nothing, then implement no truncation cutoff
    if isnothing(maxdim)
        U,S,V = ITensors.svd(ρ2,Linds,cutoff=0)
    else
        U,S,V = ITensors.svd(ρ2,Linds,maxdim=maxdim)
    end

    # Make the SVD channel 
    if c1>1 && c2<L
        ρ_targ = U*S*V # this is our target tensor 
        ρ_targ = replaceprime(ρ_targ, 1 => 3, tags="Site")
        ρ_targ = replaceprime(ρ_targ, 0 => 2, tags="Site")

        Φ = CPTP_approximation_JuMP(ρ2, ρ_targ) # find the nearest CPTP map
    end

    # Update the original MPO 
    ρ̃[c1] = U
    ρ̃[c2] = S*V

    return ρ̃
end

function initialize_channel(SInds; random_init=false)
    CS = combiner(SInds...) # make a combiner tensor for the inds
    cS = combinedind(CS) # make a new label for the combined indices 

    # Make the kraus operators
    Id = sqrt(1-ε) * Matrix(I, 2, 2)
    σx = sqrt(ε/3) * [0.0 1.0 
          1.0 0.0] 
    σy = sqrt(ε/3) * [0.0 -1.0im 
          -1.0im 0.0]
    σz = sqrt(ε/3) * [1.0 0.0 
          0.0 -1.0]

    Ids = copy(Id)
    σxs = copy(σx)
    σys = copy(σy)
    σzs = copy(σz)

    for _ in 2:length(SInds)
        # Build up the total operator 
        Ids = Ids ⊗ Id 
        σxs = σxs ⊗ σx
        σys = σys ⊗ σy 
        σzs = σzs ⊗ σz 
    end

    # Stack them together 
    K_elems = cat(collect(Ids), collect(σxs), collect(σys), collect(σzs), dims=3)

    if random_init
        K_elems = rand(size(K_elems))
    end

    # Turn this into an ITensor with the appropriate indices 
    sum_idx = Index(4, tags="Kraus")
    KdagC = ITensor(K_elems, cS, prime(cS), sum_idx)
    Kdag = KdagC * CS * prime(CS)
    Kdag = prime(Kdag, 1, plev=1)
    K = prime(dag(Kdag))
    K = replaceprime(K, 1 => 0, tags="Kraus")

    return K, Kdag
end

function get_siteinds(ρ::ITensor)
    return tag_and_plev(ρ, tag="Site", lev=0)
end

function check_isometry(K::ITensor, Kdag::ITensor)
    @error "TODO"
end

function identity_operator(K::ITensor, Kdag::ITensor)
    @error "TODO"
end

function get_all_indices(K::ITensor, Kdag::ITensor, ρ::ITensor, ρ̃::ITensor)
    L1, R1, L3, R3, S = inds(K)
    slink = taginds(K, "Kraus")[1]
    @assert S==slink 

    L, R, L2, R2, S̃ = inds(Kdag)
    @assert S̃==slink 

    lL, lR, L̃, R̃, L̃1, R̃1 = inds(ρ)
    lLlink,lRlink = taginds(ρ, "Link")
    @assert (lL==lLlink && lR==lRlink) || (lL==lRlink && lR==lLlink)

    @assert L==L̃ && R==R̃ && L1==L̃1 && R1==R̃1 

    if inds(ρ̃) != (lL, lR, L2, R2, L3, R3)
        ρ̃ = permute(ρ̃, lL, lR, L2, R2, L3, R3)
    end

    return K, Kdag, ρ, ρ̃, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR
end

function CPTP_approximation_JuMP(ρ::ITensor, ρ̃::ITensor)
    K,Kdag = initialize_channel(get_siteinds(ρ), random_init=false)
    
    """
    After processing...
        K has indices L1, R1, L3, R3, S
        Kdag has indices L, R, L2, R2, S 
        ρ has indices lL, lR, L, R, L1, R1
        ρ̃ has indices lL, lR, L2, R2, L3, R3
        Id has indices L, R, L1, R1, L2, R2, L3, R3 
    """
    # Permute the tensors into standard form 
    K, Kdag, ρ, ρ̃, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR = get_all_indices(K,Kdag,ρ,ρ̃)
    dS = ITensors.dim(S)

    ## COMBINING LEGS ## 
    # These are the combiner tensors
    X = combiner(L,R)
    Y = combiner(L2,R2)
    X1 = combiner(L1,R1)
    Y1 = combiner(L3,R3)
    B = combiner(lL,lR)
    # These are the indices 
    Xc = combinedind(X)
    Yc = combinedind(Y)
    X1c = combinedind(X1)
    Y1c = combinedind(Y1)
    Bc = combinedind(B)
    # These are the dimensions of all the indices 
    dX = ITensors.dim(Xc)
    dY = ITensors.dim(Yc)
    dX1 = ITensors.dim(X1c)
    dY1 = ITensors.dim(Y1c)
    dB = ITensors.dim(Bc)

    ## REFERENCE RESULTS ## 
    Nρ = K*ρ*Kdag
    if inds(Nρ) != (lL, lR, L2, R2, L3, R3)
        Nρ = permute(Nρ, lL, lR, L2, R2, L3, R3)
    end
    loss = norm(Nρ - ρ̃)
    @show loss  

    ## Combine the legs ## 
    K = K*X1*Y1
    Kdag = Kdag*X*Y 
    ρ = ρ*B*X*X1
    ρ̃ = ρ̃*B*Y*Y1
    Id = delta(Xc,X1c)
    Nρ = Nρ * B * Y * Y1 # for reference only 

    # checking for isometry 
    KdagK = Kdag*delta(Yc,Y1c)*K
    @assert array(Id)==array(KdagK)

    ## PERMUTE ALL THE INDICES TO MAKE SURE WE HAVE WHAT WE WANT ## 
    if inds(K) != (X1c, Y1c, S)
        K = permute(K, X1c, Y1c, S)
    end

    if inds(Kdag) != (Xc, Yc, S)
        Kdag = permute(Kdag, Xc, Yc, S)
    end

    if inds(ρ) != (Bc, Xc, X1c)
        ρ = permute(ρ, Bc, Xc, X1c)
    end

    if inds(ρ̃) != (Bc, Yc, Y1c)
        ρ̃ = permute(ρ̃, Bc, Yc, Y1c)
    end

    if inds(Id) != (Xc, X1c)
        Id = permute(Id, Xc, X1c)
    end

    ## EXTRACT THE TENSORS FROM THE ITENSOR OBJECTS ##
    K_arr = array(K) # dX1, dY1, dS
    K_arr_flat = reshape(K_arr, dX1*dY1*dS) # dX1*dY1*dS
    Kdag_arr = array(Kdag) # dX, dY, dS
    ρ_arr = array(ρ) # dB, dX, dX1
    ρ̃_arr = array(ρ̃) # dB, dY, dY1
    Id_arr = array(Id) # dX, dX1 

    ## OPTIMIZATION ##
    # very relevant: https://github.com/jump-dev/JuMP.jl/issues/2060 
    model = Model(Ipopt.Optimizer) 

    # Define K, the variable being optimized over 
    # Initialize with the 'identity' version of K 
    K = [@variable(model, set = ComplexPlane(), start=K_arr_flat[n]) for n in 1:dX1*dY1*dS] # dX*dY*dS 
    K = reshape(K, dX1, dY1, dS) # now has shape dX1, dY1, dS
    
    # Make K†
    Kdag = LinearAlgebra.conj(K) # has shape dX, dY, dS 

    ## CONSTRAINTS ##
    numconstraints = 0
    # We are performing Kdag[x, y, s] * K[x1, y1, s] * δ[y, y1]
    for x in 1:dX
        for x1 in 1:dX1

            # Sum over the contracted indices (y, y1, s)
            KdagK_elem = @expression(model, K[x1, 1, 1] * Kdag[x, 1, 1])
            for y in 2:dY # Only need to do the y sum. The sum over y1 picks out all the y1==y terms 
                for s in 2:dS # Do the sum over the Kraus index 
                    inc = @expression(model, K[x1, y, s] * Kdag[x, y, s]) # how much we are incrementing by 
                    KdagK_elem = @expression(model, KdagK_elem + inc)
                end
            end

            # we add the constraints here 
            Id_elem = Id_arr[x, x1]
            @constraint(model, KdagK_elem==Id_elem)
            numconstraints += 1

        end
    end

    ## OBJECTIVE ## 
    @NLexpression(model, loss, 0)
    numsquares = 0 
    # We are computing K[x1, y1, s] * ρ[b, x, x1] * Kdag[x, y, s]
    for y in 1:dY # Y, Y1, and B are the free indices 
        for y1 in 1:dY1
            for b in 1:dB

                # Now sum over the contracted indices 
                KρKdag_elem = @expression(model, K[1, y1, 1] * ρ_arr[b, 1, 1] * Kdag[1, y, 1])
                for x in 2:dX
                    for x1 in 2:dX1
                        for s in 2:dS # Kraus index 
                            inc = @expression(model, K[x1, y1, s] * ρ_arr[b, x, x1] * Kdag[x, y, s])
                            KρKdag_elem = @expression(model, KρKdag_elem + inc)
                        end
                    end
                end

                # Now take the difference 
                Δ = KρKdag_elem - ρ̃_arr[b, y, y1]
                Δreal = real(Δ)
                Δcomp = imag(Δ)
                Δsquared = @NLexpression(model, Δreal^2 + Δcomp^2)
                loss = @NLexpression(model, loss+Δsquared)
                numsquares += 1

            end
        end
    end

    @show numconstraints
    @show numsquares

    @NLobjective(model, Min, loss)
    optimize!(model)

    @assert 1==0
end



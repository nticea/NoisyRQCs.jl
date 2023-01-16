include("circuit.jl")
include("utilities.jl")

using ITensors
using LinearAlgebra
using Kronecker
using NLsolve
using Einsum 
using Convex, SCS # convex solvers 

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

        Φ = CPTP_approximation(ρ2, ρ_targ) # find the nearest CPTP map
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
    # get some relevant indices 
    kinds = tag_and_plev(Kdag; tag="Site", lev=0)

    # contract over the summation index 
    KKdag = K*Kdag

    # combine the in and out indices 
    c_in = combiner(kinds, prime(kinds))
    c_out = combiner(prime(kinds,2), prime(kinds,3))

    # check whether the contracted kraus operators give the identity
    Id = delta(combinedind(c_in), combinedind(c_out))
    return array(Id)==array(KKdag*c_in*c_out)
end

function identity_operator(K::ITensor, Kdag::ITensor)
    # get some relevant indices 
    kinds = tag_and_plev(Kdag; tag="Site", lev=0)

    # combine the in and out indices 
    c_in = combiner(kinds, prime(kinds))
    c_out = combiner(prime(kinds,2), prime(kinds,3))

    # check whether the contracted kraus operators give the identity
    Id = delta(combinedind(c_in), combinedind(c_out))
    Id = Id * c_in * c_out  
    return Id 
end

function frobenius_inner(A::ITensor, B::ITensor)
    return tr(dag(A)*B)
end

const finner = frobenius_inner # alias for the frobenius inner product 

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

    Id = identity_operator(K, Kdag)
    L̃, R̃, L̃1, R̃1, L̃2, R̃2, L̃3, R̃3 = inds(Id)
    @assert L==L̃ && R==R̃ && L1==L̃1 && R1==R̃1 && L2==L̃2 && R2==R̃2 && L3==L̃3 && R3==R̃3

    return K, Kdag, ρ, ρ̃, Id, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR
end

function flatten(A)
    A_flat = reshape(A, prod(size(A)))
    Ã = unflatten(A_flat, size(A))
    @assert A==Ã
    return A_flat
end

function unflatten(A, dimsA...)
    reshape(A, dimsA...)
end

function flatten_K(K::ITensor, Kdag::ITensor, ρ::ITensor, ρ̃::ITensor)
    # Permute the tensors into standard form 
    K, Kdag, ρ, ρ̃, Id, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR = get_all_indices(K,Kdag,ρ,ρ̃)
    dL, dR, dL1, dR1, dL2, dR2, dL3, dR3, dS, dlL, dlR = ITensors.dim.([L, R, L1, R1, L2, R2, L3, R3, S, lL, lR])

    # Turn K into a 1D vector for compatibility with nonlinear eqn solver  
    Kflat = zeros(ComplexF64, dL1*dR1*dL3*dR3*dS)
    step = dL1*dR1*dL3*dR3
    for (i,s) in zip(1:step:length(Kflat),1:dS)
        Ki = array(K)[:,:,:,:,s]
        Kflat[i:i+step-1] = flatten(Ki) 
    end

    # # An extra slot for λ
    # append!(Kflat,1)

    return Kflat 
end

function reconstruct_K(Kflat::AbstractArray, K::ITensor, Kdag::ITensor, ρ::ITensor, ρ̃::ITensor)
    # Permute the tensors into standard form 
    K, Kdag, ρ, ρ̃, Id, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR = get_all_indices(K,Kdag,ρ,ρ̃)
    dL, dR, dL1, dR1, dL2, dR2, dL3, dR3, dS, dlL, dlR = ITensors.dim.([L, R, L1, R1, L2, R2, L3, R3, S, lL, lR])

    # Kflat = init_x[1:end-1]
    # Take input Kflat and reconstruct all the Ki 
    K̃_array = zeros(size(array(K)))
    step = dL1*dR1*dL3*dR3
    for (i,s) in zip(1:step:length(Kflat),1:dS)
        Ki = unflatten(Kflat[i:i+step-1], dL1, dR1, dL3, dR3)
        K̃_array[:,:,:,:,s] = Ki
    end

    return K̃_array
end

function reconstruct_K(Kflat::AbstractArray, dL1::Int, dR1::Int, dL3::Int, dR3::Int, dS::Int)
    # Permute the tensors into standard form 

    # Kflat = init_x[1:end-1]
    # Take input Kflat and reconstruct all the Ki 
    K̃_array = zeros(dL1, dR1, dL3, dR3, dS)
    step = dL1*dR1*dL3*dR3
    for (i,s) in zip(1:step:length(Kflat),1:dS)
        Ki = unflatten(Kflat[i:i+step-1], dL1, dR1, dL3, dR3)
        K̃_array[:,:,:,:,s] = Ki
    end

    return K̃_array
end


"""
ρ is the tensor to which we are applying the quantum channel
ρ̃ is the desired output upon application of the quantum channel
"""
function CPTP_approximation(ρ::ITensor, ρ̃::ITensor)
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
    K, Kdag, ρ, ρ̃, Id, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR = get_all_indices(K,Kdag,ρ,ρ̃)
    dL, dR, dL1, dR1, dL2, dR2, dL3, dR3, dS, dlL, dlR = ITensors.dim.([L, R, L1, R1, L2, R2, L3, R3, S, lL, lR])
    ρ_arr = array(ρ)
    ρ̃_arr = array(ρ̃)
    Id_arr = array(Id)
    @assert check_isometry(K, Kdag)

    # checking that we are getting what we expect 
    ## ACTUAL ## 
    KρKdag = K*ρ*Kdag
    if inds(KρKdag) != (lL, lR, L2, R2, L3, R3)
        KρKdag = permute(KρKdag, lL, lR, L2, R2, L3, R3)
    end
    loss = norm(KρKdag - ρ̃)
    @show loss 

    X = combiner(L,R)
    Y = combiner(L2,R2)
    X1 = combiner(L1,R1)
    Y1 = combiner(L3,R3)
    B = combiner(lL,lR)

    Xc = combinedind(X)
    Yc = combinedind(Y)
    X1c = combinedind(X1)
    Y1c = combinedind(Y1)
    Bc = combinedind(B)

    KρKdag = KρKdag * B * Y * Y1 
    

    ## ELEMENTWISE ## 
    K = K*X1*Y1
    Kdag = Kdag*X*Y
    ρ = ρ*B*X*X1
    ρ̃ = ρ̃*B*Y*Y1

    if inds(K) != (X1c, Y1c, S)
        K = permute(K, X1c, Y1c, S)
    end

    if inds(Kdag) != (Xc, Yc, S)
        Kdag = permute(Kdag, Xc, Yc, S)
    end

    if inds(ρ) != (Bc, Xc, X1c)
        ρ = permute(ρ, Bc, Xc, X1c)
    end

    if inds(ρ̃) != (B, Yc, Y1c)
        ρ̃ = permute(ρ̃, Bc, Yc, Y1c)
    end

    K_arr = array(K)
    Kdag_arr = array(Kdag)
    ρ_arr = array(ρ)
    ρ̃_arr = array(ρ̃)  

    # useful dimensions 
    X = dL*dR
    Y = dL2*dR2
    X1 = dL1*dR1
    Y1 = dL3*dR3
    B = dlL*dlR

    Nρ = zeros(ComplexF64, B, Y, Y1)

    # Iterate through all of the link indices
    for l in 1:B
        # Iterate through all of the ys 
        for y in 1:Y
            for y1 in 1:Y1
                # Iterate through all of the Kraus operators 
                for s in 1:dS
                    for x in 1:X
                        for x1 in 1:X1
                            @assert K_arr[x1, y1, s] == array(K)[x1, y1, s]
                            @assert Kdag_arr[x, y1, s] == array(Kdag)[x, y1, s]
                            @assert ρ_arr[l,x,x1] == array(ρ)[l,x,x1]
                        end
                    end
                    #Nρ[l, y, y1] += transpose(K_arr[:, y1, s]) * @assert K_arr[x1, y1, s] == array(K)[x1, y1, s] * Kdag_arr[:, y, s]
                end
            end
        end
    end

    loss2 = norm(Nρ - ρ̃_arr)
    @show loss2 
    println("")

    # # Check that reconstruction works 
    # Kflat = flatten_K(K, Kdag, ρ, ρ̃)
    # K̃_array = reconstruct_K(Kflat, K, Kdag, ρ, ρ̃)
    # @assert K̃_array==array(K)

    # K̃_array = reshape(Kflat, dL1, dR1, dL3, dR3, dS)
    # @assert K̃_array==array(K)

    # ## TESTING ## 
    # K_flatarr = reshape(array(K), dL1 * dR1 * dL3 * dR3 * dS)
    # K_arr = reshape(K_flatarr, dL1, dR1, dL3, dR3, dS)
    # @assert K_arr == array(K)
    # Kdag_arr = conj(K_arr)
    # @assert Kdag_arr == array(Kdag)

    # @einsum Nρ[lL, lR, L2, R2, L3, R3] := K_arr[L1, R1, L3, R3, S] * ρ_arr[lL, lR, L, R, L1, R1] * Kdag_arr[L, R, L2, R2, S]
    # @show size(Nρ)

    # KρKdag = K*ρ*Kdag
    # if inds(KρKdag) != (lL, lR, L2, R2, L3, R3)
    #     KρKdag = permute(KρKdag, lL, lR, L2, R2, L3, R3)
    # end
    # @assert Nρ == array(KρKdag)

    # X = dL*dR
    # Y = dL2*dR2
    # X1 = dL1*dR1
    # Y1 = dL3*dR3
    # l = dlL*dlR

    # Kdag_arr = reshape(Kdag_arr, X*Y, dS)
    # K_arr = reshape(K_arr, X1*Y1, dS)
    # ρ_arr = reshape(ρ_arr, l, X*X1)
    
    # KdagK_arr = Kdag_arr * K_arr'
    # KdagK_arr = reshape(KdagK_arr, X, Y, X1, Y1)
    # KdagK_arr = permutedims(KdagK_arr, [1,3,2,4])
    # KdagK_arr = reshape(KdagK_arr, X*X1, Y*Y1)
    
    # Nρ_arr = ρ_arr*KdagK_arr #l, Y*Y1
    # Nρ_arr = reshape(Nρ_arr, dlL, dlR, dL2, dR2, dL3, dR3)
    # @show size(Nρ_arr)

    # @assert Nρ_arr == array(KρKdag)

    ### COMPLEX CONVEX SOLVER ###
    # objective = Variable()
    # constraints = []

    # # useful dimensions 
    # X = dL*dR
    # Y = dL2*dR2
    # X1 = dL1*dR1
    # Y1 = dL3*dR3
    # B = dlL*dlR

    # # Define the complex variable     
    # K_arr = [ComplexVariable(X1, Y1) for _ in 1:dS] #secretly is X1, Y1, dS
    
    # # Reshape the density matrices 
    # ρ_arr = reshape(ρ_arr, B, X, X1)
    # ρ̃_arr = reshape(ρ̃_arr, B, X, X1)
    
    # constraints = Convex.EqConstraint[]
    
    # # Iterate over the link index in ρ
    # for l in 1:B
    #     ρl = ρ_arr[l,:,:] # is size X, X1 
    #     ρ̃l = ρ̃_arr[l,:,:] # also is size X, X1 
        
    #     # Iterate over the Kraus sumlink Index
    #     for s in 1:dS
    #         Ki = K_arr[s] # is size X1, Y1 

    #         # the objective 
    #         rec_loss = ρ̃l - quadform(Ki', ρl) # is size Y, Y1 
    #         objective += sum(abs2(rec_loss))

    #         # the constraint 
    #         KdagK = quadform(Ki, Matrix(I, size(ρl)...))
    #         push!(constraints, KdagK == Matrix(I, X1, Y1))
    #     end
    # end

    # # Define the program using objective and constraints
    # p = minimize(objective, constraints)

    # # Solve the program 
    # solve!(p, SCS.Optimizer; silent_solver = true)

    # Evaluate the program 
    

    ## Do it using Lagrange multipliers
    
    ## Check that reconstruction works 
    # K̃_array = reconstruct_K(initial_x, K, Kdag, ρ, ρ̃)
    # @assert K̃_array==array(K)

    # initial_x = init_x(K, Kdag, ρ, ρ̃)
    # function f!(F, x)
    #     # The x vector is structured like [K1..., K2..., Kn..., Kdag1..., Kdagn..., λ]

    #     # first, let us regenerate our Kraus operators 
    #     K̃_array = reconstruct_K(x, K, Kdag, ρ, ρ̃)

    #     # The ∂L/∂K_i equations
    #     for s in 1:S
    #         Ki = K̃_array[:,:,:,:,dS]
            
    #         A[lL, lR, L, R, L3, R3] := Ki[L1, R1, L3, R3] * ρ[lL, lR, L, R, L1, R1]
    #     end

    #     # The ∂L/∂λ equation   
    #     F[end] = 0

    # end
    #nlsolve(f!, initial_x, autodiff = :forward)

end



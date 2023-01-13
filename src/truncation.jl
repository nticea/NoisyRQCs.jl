include("circuit.jl")
include("utilities.jl")

using ITensors
using LinearAlgebra
using Kronecker

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

        Φ, err = project_CPTP(ρ2, ρ_targ) # find the nearest CPTP map
    end

    # Update the original MPO 
    ρ̃[c1] = U
    ρ̃[c2] = S*V

    return ρ̃
end

function projective_SVD(ρ_init::ITensor, ρ_targ::ITensor, Kdag::ITensor)
    # Make the environment tensor
    Γ = Kdag * ρ_init * ρ_init

    # SVD the environment tensor 
    Linds = [tag_and_plev(Γ, tag="Site", lev=2)..., taginds(Γ, "Kraus")]
    U,S,V = ITensors.svd(Γ, Linds)
    ulink, vlink = inds(S)

    # FIXED, I THINK!! 
    Kdag = V*dag(U)*delta(ulink,vlink) 
    K = prime(dag(Kdag)) 

    return K, Kdag 
end

function initialize_channel(SInds)

    CS = combiner(SInds...) # make a combiner tensor for the inds
    cS = combinedind(CS) # make a new label for the combined indices 

    # Make the kraus operators
    Id = sqrt(1-ε) * Matrix(I, D, D)
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

    # Turn this into an ITensor with the appropriate indices 
    sum_idx = Index(4, tags="Kraus")
    KdagC = ITensor(K_elems, cS, prime(cS), sum_idx)
    Kdag = KdagC * CS * prime(CS)
    Kdag = prime(Kdag, 1, plev=1)
    K = prime(dag(Kdag))

    return K, Kdag
end


"""
Implements projective truncation via svd
See Algorithms for TNR (G. Evenbly 2017), section C PRB for more information
"""
function project_CPTP(ρ_init::ITensor, ρ_targ::ITensor; niter::Int=10, num_kraus::Int=4)
    # Initialize our guess for K 
    # in_inds = tag_and_plev(ρ_init, tag="Site", lev=0)
    # slink = Index(num_kraus, tags="Kraus")
    # Kdag = randomITensor(in_inds, prime(in_inds, 2), slink)
    # K = prime(dag(Kdag))

    # for benchmarking 
    ρ_init0 = replaceprime(ρ_init, 1 => 3, tags="Site")
    ρ_init0 = replaceprime(ρ_init0, 0 => 2, tags="Site")

    K,Kdag = initialize_channel(tag_and_plev(ρ_init, tag="Site", lev=0))
    slink = getfirst(x -> hastags(x, "Kraus"), inds(Kdag))

    # For benchmarking (assess projection error)
    err = []
    err0 = []
    for n in 1:niter
        # benchmarking 
        KKdag = K*Kdag*delta(slink,prime(slink)) 
        ρ̂ = ρ_init*KKdag

        push!(err, norm(ρ̂-ρ_targ))
        push!(err0, norm(ρ_init0-ρ_targ))

        # projective SVD step 
        K, Kdag  = projective_SVD(ρ_init, ρ_targ, Kdag)
    end
    @show err 
    @show err0
    return K*Kdag, err
end

# function pseudoinverse(A::ITensor, Linds, Rinds; tol=1e-5)
#     Lis = commoninds(A, Linds)
#     Ris = commoninds(A, Rinds)

#     # We will combine all the left inds together, and all the right inds together 
#     CL = combiner(Lis...) # make a combiner tensor for the inds
#     CR = combiner(Ris...)
#     cL = combinedind(CL) # make a new label for the combined indices 
#     cR = combinedind(CR)
#     AC = A * CR * CL # combine the indices by applying the combiner tensor 

#     # Next we identify the fixed indices -- those that index the identity matrices 
#     fixed = uniqueinds(A, Linds..., Rinds...)
#     @assert length(fixed)==2 "Only 2 link indices supported at the moment"
#     df1, df2 = ITensors.dim.(fixed)
#     dcL, dcR = ITensors.dim(cL), ITensors.dim(cR)

#     # Permute so that the extracted Julia array has dimensions fixed inds x cL x cR 
#     if inds(AC) != (fixed..., cL, cR)
#       AC = permute(AC, fixed..., cL, cR)
#     end
    
#     # Extract the regular Julia tensor from AC 
#     ACarr = array(AC)

#     # Construct the pseudoinverse 
#     pinvACarr = zeros(ComplexF64, size(ACarr))
    
#     # Compute the pseudoinverse (or true inverse, if dim(cL)=dim(cR))
#     for i in 1:df1 # fix the first link index 
#         for j in 1:df2 # fix the second link index 
#             ACarr_ij = ACarr[i,j,:,:] # extract the matrix we wish to invert 
#             # if dcL == dcR # square matrix, so compute the true inverse 
#             #     invarr = inv(ACarr_ij)
#             # else # compute the psuedoinverse 
#             #     invarr = pinv(ACarr_ij)
#             # end
#             invarr = pinv(ACarr_ij)

#             # check the reconstruction error
#             rec = invarr * ACarr_ij
#             err = norm(rec - Matrix(I, size(rec)...)) #how far away from I it is

#             if err > tol # we'll try again with another block 
#                 return
#             end
#             pinvACarr[i,j,:,:] = sqrt.(invarr) # update the tensor of inverses  
#         end
#     end

#     # Make the new inverse tensor 
#     pinvAC = ITensor(pinvACarr, fixed..., cL, cR)

#     # Decombine the indices 
#     pinvA = pinvAC * CR * CL

#     # Raise the indices appropriately 
#     pinvAprime = prime(pinvA, 2, tags="Site") 

#     ## TESTING ##
#     invA = pinvA * pinvAprime
#     A = prime(invA, -2, tags="Site")
#     @show inds(A)

#     AC = A * CR * CL # combine the indices by applying the combiner tensor 

#     # Next we identify the fixed indices -- those that index the identity matrices 
#     fixed = uniqueinds(A, Linds..., Rinds...)
#     @assert length(fixed)==2 "Only 2 link indices supported at the moment"
#     df1, df2 = ITensors.dim.(fixed)
#     dcL, dcR = ITensors.dim(cL), ITensors.dim(cR)

#     # Permute so that the extracted Julia array has dimensions fixed inds x cL x cR 
#     if inds(AC) != (fixed..., cL, cR)
#       AC = permute(AC, fixed..., cL, cR)
#     end
    
#     # Extract the regular Julia tensor from AC 
#     ACarr = array(AC)
    
#     # Compute the pseudoinverse (or true inverse, if dim(cL)=dim(cR))
#     for i in 1:df1 # fix the first link index 
#         for j in 1:df2 # fix the second link index 
#             ACarr_ij = ACarr[i,j,:,:] # extract the matrix we wish to invert 
            
#             err = norm(ACarr_ij - Matrix(I, size(rec)...)) #how far away from I it is

#             @show err 
#         end
#     end


#     ## END TESTING ##

#     return pinvA * pinvAprime 
# end

# # function pseudoinverse(A::ITensor, Linds...; tol::Real=1e-5)
# #     # We are taking the pseudoinverse of A and multiplying it with B 
  
# #     Lis = commoninds(A, Linds)
# #     Ris = uniqueinds(A, Lis)
  
# #     CL = combiner(Lis...)
# #     CR = combiner(Ris...)
  
# #     AC = A * CR * CL
  
# #     cL = combinedind(CL)
# #     cR = combinedind(CR)

# #     if inds(AC) != (cL, cR)
# #       AC = permute(AC, cL, cR)
# #     end

# #     # Get the matrix inverse 
# #     pinvA = pinv(array(AC)) 
# #     rec = pinvA * array(AC)
# #     err = norm(rec - Matrix(I, size(rec)...)) #how far away from I it is
# #     if err < tol
# #         # give it the right indices 
# #         pinvT̃ = ITensor(pinvA, cL, cR) 
# #         pinvT = pinvT̃ * CL * CR

# #         # permute so that pinvT and A have the same index order 
# #         # if inds(pinvT) != inds(A)
# #         #     pinvT = permute(pinvT, inds(A)...)
# #         # end

# #         return pinvT
# #     end
# # end

# function channel_truncation(A::ITensor, B::ITensor; tol::Real=1e-5)
#     A = copy(A)
#     B = copy(B)

#     Linds = primed_inds(A)
#     Rinds = noprime.(Linds)

#     # Take the pseudoinverse of A 
#     pinvA = pseudoinverse(A, Linds, Rinds, tol=tol)
#     if !isnothing(pinvA)
#         # Set the prime levels on the physical indices of B to be 2 higher 
#         B = prime(B, 2, tags="Site")  
        
#         ## TESTING ## 
#         AAinv = A * pinvA
#         @show inds(AAinv)
#         #@show AAinv
#         @assert 1==0


#         # @assert 1==0
#         ## END TESTING ##

#         # Compute the channel 
#         Φ = pinvA * B
        
#         # Compare the output with the true SVD 
#         B̃ = Φ*A # Apply the SVD channel  
#         Δ = B̃ - B
#         @show array(Δ)[1:10]
#         @show norm(Δ)

#     end
# end

# # function channel_truncate(A::ITensor, B::ITensor, Linds...; tol::Real=1e-5)
# #     # We are taking the pseudoinverse of A and multiplying it with B 
  
# #     Lis = commoninds(A, Linds)
# #     Ris = uniqueinds(A, Lis)
  
# #     CL = combiner(Lis...)
# #     CR = combiner(Ris...)
  
# #     AC = A * CR * CL
# #     BC = B * CR * CL 
  
# #     cL = combinedind(CL)
# #     cR = combinedind(CR)

# #     if inds(AC) != (cL, cR)
# #       AC = permute(AC, cL, cR)
# #     end

# #     if inds(BC) != (cL, cR)
# #         BC = permute(BC, cL, cR)
# #     end

# #     # Get the matrix inverse of A 
# #     pinvA = pinv(array(AC)) 
# #     rec = pinvA * array(AC) # reconstruction
# #     err = norm(rec - Matrix(I, size(rec)...)) #how far away from I it is
# #     @show err 
# #     if err < tol
# #         # to make the channel, multiply with B 
# #         Φ_arr = pinvA * array(BC)
# #         SInds = primed_inds(A)
# #         Φ = ITensor(Φ_arr, [noprime(SInds), SInds, prime(SInds), prime(prime(SInds))]...)
# #         return Φ
# #     end
# # end

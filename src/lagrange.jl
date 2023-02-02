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
function apply_circuit_truncation_channel(ψ0::MPS, T::Int, truncdim::Int; random_type="Haar", ε=0.05, maxdim=nothing)
    L = length(ψ0)
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
        Ks, optloss, initloss, loss_hist = truncation_quantum_channel(ρ, truncdim, apply_gate=false)

        # record the data 
        push!(all_Ks, Ks)
        push!(all_loss_hist, loss_hist)
        push!(all_optloss, optloss)
        push!(all_initloss, initloss)
    end

    @show tr(ρ)

    return ρ, all_Ks, all_optloss, all_initloss, all_loss_hist
end

function truncation_quantum_channel(ρ::MPO, truncdim::Int; apply_gate::Bool=false, truncidx::Union{Int,Nothing}=nothing)
    ρ = copy(ρ)
    L = length(ρ)

    if isnothing(truncidx)
        truncidx = floor(Int, L / 2)
    end
    sites = physical_indices(ρ)
    sL = noprime(sites[truncidx])
    sR = noprime(sites[truncidx+1])

    # Orthogonalize the MPS around this site 
    orthogonalize!(ρ, truncidx)
    @show linkdim(ρ, truncidx)

    ρ_ij = ρ[truncidx] * ρ[truncidx+1] # this is our original tensor
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
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(array(ρ_ij), array(ρ̃_ij), nkraus=4)
    # objective value is the 3rd entry
    loss_hist = map(x -> x[3], iterdata)

    return Ks, optloss, initloss, loss_hist
end

function truncation_quantum_channel_rdm(ρ::MPO, truncdim::Int; apply_gate::Bool=false, truncidx::Union{Int,Nothing}=nothing)
    ρ = copy(ρ)
    L = length(ρ)

    if isnothing(truncidx)
        truncidx = floor(Int, L / 2)
    end

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
    X = combiner(sL, sR)
    X1 = combiner(prime(sL), prime(sR))
    ρtr = ρtr * X * X1
    ρ̃tr = ρ̃tr * X * X1

    if inds(ρtr) != inds(ρ̃tr)
        ρtr = permute(ρtr, inds(ρ̃tr))
    end

    # find the nearest CPTP map
    Ks, optloss, initloss, iterdata, model = approxquantumchannel(array(ρtr), array(ρ̃tr), nkraus=4)
    # objective value is the 3rd entry
    loss_hist = map(x -> x[3], iterdata)

    return Ks, optloss, initloss, loss_hist
end


# function frobenius_norm(A, B)
#     Δ = A - B
#     Δreal = real(Δ)
#     Δimag = imag(Δ)
#     Δnorm = sum(Δreal[i]^2 for i in CartesianIndices(Δreal)) +
#             sum(Δimag[i]^2 for i in CartesianIndices(Δimag))
#     return sqrt(Δnorm)
# end

# function approximate_tsvd(ρ, ρ̃; nkraus::Int=4)

#     # 3. Approximate truncated density matrix with quantum operation by finding optimal Kraus
#     #    operators using non-convex optimization (quartic objective with quadratic constraint)
#     #
#     #                                    min{Kᵢ} ‖∑ᵢKᵢρKᵢ† - ρ̃‖₂
#     #                                    s.t.    ∑ᵢKᵢ†Kᵢ = I

#     # This is the initial difference between ρ and ρ̃
#     initial_loss = (frobenius_norm(ρ, ρ̃))^2

#     # Initialize the JuMP model
#     model = Model(Ipopt.Optimizer)

#     ndims_out, ndims_out2 = size(ρ)
#     @assert ndims_out==ndims_out2
#     @assert size(ρ̃)==(ndims_out,ndims_out)

#     # a. Build Krauss operator variables
#     # complex array variables are not currently supported, so have to reshape
#     Ksdims = (ndims_out, ndims_out, nkraus)
#     # Optimizer needs help with starting from a feasible point, using Kᵢ = I
#     Ks = reshape([
#             @variable(model, set = ComplexPlane(), start = sqrt(1/nkraus)*I[i, j])
#             for (i, j, _) in Tuple.(CartesianIndices(Ksdims))
#         ], Ksdims)

#     # b. define Krauss operators contraint: ∑ᵢKᵢ†Kᵢ = I
#     @constraint(model, sum(K' * K for K in eachslice(Ks, dims=3)) .== I)

#     # c. Find the difference between the approximation and tsvd matrix and compute Frobenius norm
#     #                                    ∑ᵢKᵢρKᵢ† - ρ̃.
#     approx = @expression(model, sum(K * ρ * K' for K in eachslice(Ks, dims=3)))
#     diff = @expression(model, approx - ρ̃)

#     # d. Compute the Frobenius norm. This will have quartic terms, so we have to use NLexpression
#     # NLexpression does not support complex variables :(
#     diffreal = real(diff)
#     diffimag = imag(diff)
#     fnorm = @NLexpression(model,
#         sum(diffreal[i]^2 for i in CartesianIndices(diffreal))
#         +
#         sum(diffimag[i]^2 for i in CartesianIndices(diffimag))
#     )
#     @NLobjective(model, Min, fnorm)

#     loss_hist = []

#     # callback function for recording results
#     function my_callback(alg_mod::Cint,iter_count::Cint,obj_value::Float64,
#         inf_pr::Float64,inf_du::Float64,mu::Float64,d_norm::Float64,
#         regularization_size::Float64,alpha_du::Float64,
#         alpha_pr::Float64,ls_trials::Cint)

#         push!(loss_hist, obj_value)
#         return true
#     end

#     # set callback function for this model
#     MOI.set(model, Ipopt.CallbackFunction(), my_callback)

#     # e. Let's optimize!
#     optimize!(model)

#     # 4. Process results
#     @show initial_loss
#     @show objective_value(model)

#     # return results
#     return value.(Ks), loss_hist
# end

function initialize_channel(SInds; random_init=false)
    CS = combiner(SInds...) # make a combiner tensor for the inds
    cS = combinedind(CS) # make a new label for the combined indices

    # Make the kraus operators
    Id = Matrix(I, 2, 2)
    σx = [0.0 1.0
        1.0 0.0]
    σy = [0.0 -1.0im
        -1.0im 0.0]
    σz = [1.0 0.0
        0.0 -1.0]

    Ids = sqrt(1 - ε) .* copy(Id)
    σxs = sqrt(ε / 3) .* copy(σx)
    σys = sqrt(ε / 3) .* copy(σy)
    σzs = sqrt(ε / 3) .* copy(σz)

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

function initialize_channel_identity(SInds; random_init=false, num_kraus::Int=4)
    CS = combiner(SInds...) # make a combiner tensor for the inds
    cS = combinedind(CS) # make a new label for the combined indices

    # Make the kraus operators
    Id = Matrix(I, 2, 2)
    Ids = sqrt(1 / 4) * copy(Id)

    for _ in 2:length(SInds)
        # Build up the total operator
        Ids = Ids ⊗ Id
    end

    # Stack them together
    stacked = [collect(Ids) for _ in 1:num_kraus]
    K_elems = cat(stacked..., dims=3)

    if random_init
        K_elems = rand(size(K_elems))
    end

    # Turn this into an ITensor with the appropriate indices
    sum_idx = Index(num_kraus, tags="Kraus")
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

# function check_isometry(K::ITensor, Kdag::ITensor)
#     @error "TODO"
# end

# function identity_operator(K::ITensor, Kdag::ITensor)
#     @error "TODO"
# end

# function get_all_indices(K::ITensor, Kdag::ITensor, ρ::ITensor, ρ̃::ITensor)
#     L1, R1, L3, R3, S = inds(K)
#     slink = taginds(K, "Kraus")[1]
#     @assert S==slink

#     L, R, L2, R2, S̃ = inds(Kdag)
#     @assert S̃==slink

#     lL, lR, L̃, R̃, L̃1, R̃1 = inds(ρ)
#     lLlink,lRlink = taginds(ρ, "Link")
#     @assert (lL==lLlink && lR==lRlink) || (lL==lRlink && lR==lLlink)

#     @assert L==L̃ && R==R̃ && L1==L̃1 && R1==R̃1

#     if inds(ρ̃) != (lL, lR, L2, R2, L3, R3)
#         ρ̃ = permute(ρ̃, lL, lR, L2, R2, L3, R3)
#     end

#     return K, Kdag, ρ, ρ̃, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR
# end

# """
# Maybe try implementing this: https://apps.dtic.mil/sti/pdfs/ADA528339.pdf
# """
# function CPTP_approximation_JuMP(ρ::ITensor, ρ̃::ITensor)
#     K,Kdag = initialize_channel(get_siteinds(ρ), random_init=false)
#     #K,Kdag = initialize_channel_identity(get_siteinds(ρ), random_init=false)

#     """
#     After processing...
#         K has indices L1, R1, L3, R3, S
#         Kdag has indices L, R, L2, R2, S
#         ρ has indices lL, lR, L, R, L1, R1
#         ρ̃ has indices lL, lR, L2, R2, L3, R3
#         Id has indices L, R, L1, R1, L2, R2, L3, R3
#     """
#     # Permute the tensors into standard form
#     K, Kdag, ρ, ρ̃, L, R, L1, R1, L2, R2, L3, R3, S, lL, lR = get_all_indices(K,Kdag,ρ,ρ̃)
#     dS = ITensors.dim(S)

#     ## COMBINING LEGS ##
#     # These are the combiner tensors
#     X = combiner(L,R)
#     Y = combiner(L2,R2)
#     X1 = combiner(L1,R1)
#     Y1 = combiner(L3,R3)
#     B = combiner(lL,lR)
#     # These are the indices
#     Xc = combinedind(X)
#     Yc = combinedind(Y)
#     X1c = combinedind(X1)
#     Y1c = combinedind(Y1)
#     Bc = combinedind(B)
#     # These are the dimensions of all the indices
#     dX = ITensors.dim(Xc)
#     dY = ITensors.dim(Yc)
#     dX1 = ITensors.dim(X1c)
#     dY1 = ITensors.dim(Y1c)
#     dB = ITensors.dim(Bc)

#     ## REFERENCE RESULTS ##
#     Nρ = K*ρ*Kdag
#     if inds(Nρ) != (lL, lR, L2, R2, L3, R3)
#         Nρ = permute(Nρ, lL, lR, L2, R2, L3, R3)
#     end
#     initial_loss = (norm(Nρ - ρ̃))^2

#     ## Combine the legs ##
#     K = K*X1*Y1
#     Kdag = Kdag*X*Y
#     ρ = ρ*B*X*X1
#     ρ̃ = ρ̃*B*Y*Y1
#     Id = delta(Xc,X1c)
#     Nρ = Nρ * B * Y * Y1 # for reference only

#     # checking for isometry
#     KdagK = Kdag*delta(Yc,Y1c)*K
#     @show array(KdagK)
#     @assert isapprox(KdagK, Id)

#     ## PERMUTE ALL THE INDICES TO MAKE SURE WE HAVE WHAT WE WANT ##
#     if inds(K) != (X1c, Y1c, S)
#         K = permute(K, X1c, Y1c, S)
#     end

#     if inds(Kdag) != (Xc, Yc, S)
#         Kdag = permute(Kdag, Xc, Yc, S)
#     end

#     if inds(ρ) != (Bc, Xc, X1c)
#         ρ = permute(ρ, Bc, Xc, X1c)
#     end

#     if inds(ρ̃) != (Bc, Yc, Y1c)
#         ρ̃ = permute(ρ̃, Bc, Yc, Y1c)
#     end

#     if inds(Id) != (Xc, X1c)
#         Id = permute(Id, Xc, X1c)
#     end

#     if inds(Nρ) != (Bc, Yc, Y1c)
#         Nρ = permute(Bc, Yc, Y1c)
#     end

#     ## EXTRACT THE TENSORS FROM THE ITENSOR OBJECTS ##
#     K_arr = array(K) # dX1, dY1, dS
#     K_arr_flat = reshape(K_arr, dX1*dY1*dS) # dX1*dY1*dS
#     Kdag_arr = array(Kdag) # dX, dY, dS
#     ρ_arr = array(ρ) # dB, dX, dX1
#     ρ̃_arr = array(ρ̃) # dB, dY, dY1
#     Id_arr = array(Id) # dX, dX1
#     Δ_arr = array(Nρ) - ρ̃_arr # dB, dY, dY1
#     true_loss = (norm(ρ_arr-ρ̃_arr))^2

#     ## OPTIMIZATION ##
#     # very relevant: https://github.com/jump-dev/JuMP.jl/issues/2060
#     model = Model(Ipopt.Optimizer)

#     # Define K, the variable being optimized over
#     # Initialize with the 'identity' version of K
#     K = [@variable(model, set = ComplexPlane(), start=K_arr_flat[n]) for n in 1:dX1*dY1*dS] # dX*dY*dS
#     K = reshape(K, dX1, dY1, dS) # now has shape dX1, dY1, dS

#     # Make K†
#     Kdag = LinearAlgebra.conj(K) # has shape dX, dY, dS

#     ## CONSTRAINTS ##
#     num_isometry_constraints = 0
#     num_hermitian_constraints = 0
#     # We are performing Kdag[x, y, s] * K[x1, y1, s] * δ[y, y1]
#     for x in 1:dX
#         for x1 in 1:dX1

#             # Sum over the contracted indices (y, y1, s)
#             KdagK_elem = @expression(model, 0)

#             # debugging
#             KdagK_elem_debug = 0

#             for y in 1:dY # Only need to do the y sum. The sum over y1 picks out all the y1==y terms
#                 for s in 1:dS # Do the sum over the Kraus index
#                     inc = @expression(model, K[x1, y, s] * Kdag[x, y, s])
#                     KdagK_elem = @expression(model, KdagK_elem + inc)

#                     # Hermiticity constraint
#                     if x1<y
#                         if x==1
#                             @constraint(model, K[x1, y, s]==conj(K[y, x1, s]))
#                             num_hermitian_constraints += 1
#                         end
#                     end

#                     # debugging
#                     inc_debug = K_arr[x1, y, s] * Kdag_arr[x, y, s]
#                     KdagK_elem_debug += inc_debug
#                 end
#             end

#             # we add the constraints here
#             Id_elem = Id_arr[x, x1]
#             @constraint(model, KdagK_elem==Id_elem)

#             # debugging
#             @assert isapprox(Id_elem, KdagK_elem_debug, atol=1e-6)

#             # count the number of constraints
#             num_isometry_constraints += 1

#         end
#     end

#     ## OBJECTIVE ##
#     @NLexpression(model, loss, 0)

#     # debugging
#     loss_debug = 0
#     numsquares = 0

#     # We are computing K[x1, y1, s] * ρ[b, x, x1] * Kdag[x, y, s]
#     for y in 1:dY # Y, Y1, and B are the free indices
#         for y1 in 1:dY1
#             for b in 1:dB

#                 # Now sum over the contracted indices
#                 KρKdag_elem = @expression(model, 0)

#                 # for debugging
#                 KρKdag_elem_debug = 0

#                 for x in 1:dX
#                     for x1 in 1:dX1
#                         for s in 1:dS # Kraus index
#                             inc = @expression(model, K[x1, y1, s] * ρ_arr[b, x, x1] * Kdag[x, y, s])
#                             KρKdag_elem = @expression(model, KρKdag_elem + inc)

#                             # debugging
#                             inc_debug = K_arr[x1, y1, s] * ρ_arr[b, x, x1] * Kdag_arr[x, y, s]
#                             KρKdag_elem_debug += inc_debug
#                         end
#                     end
#                 end

#                 # Now take the difference
#                 Δ = KρKdag_elem - ρ̃_arr[b, y, y1]
#                 Δreal = real(Δ)
#                 Δcomp = imag(Δ)
#                 Δsquared = @NLexpression(model, Δreal^2 + Δcomp^2)
#                 loss = @NLexpression(model, loss+Δsquared)

#                 # debugging
#                 Δ_debug = KρKdag_elem_debug - ρ̃_arr[b, y, y1]
#                 Δ_elem = Δ_arr[b, y, y1]
#                 @assert isapprox(Δ_debug, Δ_elem, atol=1e-6)
#                 Δreal_debug = real(Δ_debug)
#                 Δcomp_debug = imag(Δ_debug)
#                 Δsquared_debug = Δreal_debug^2 + Δcomp_debug^2
#                 loss_debug += Δsquared_debug

#                 # count the number of terms we are summing together
#                 numsquares += 1

#             end
#         end
#     end

#     @assert isapprox(loss_debug, initial_loss, atol=1e-6)

#     @NLobjective(model, Min, loss)
#     optimize!(model)

#     println("RESULTS:")
#     @show initial_loss
#     @show true_loss
#     @show objective_value(model)
#     Ksoln = value.(K)

#     @show Ksoln[:,:,1]
#     @show K_arr[:,:,1]
#     @show Ksoln[:,:,2]
#     @show K_arr[:,:,2]
#     @show Ksoln[:,:,3]
#     @show K_arr[:,:,3]
#     @show Ksoln[:,:,4]
#     @show K_arr[:,:,4]

#     println("END RESULTS")
#     println("")
# end

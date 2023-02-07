# using ITensors
using LinearAlgebra
using TSVD
using JuMP
using Ipopt
using Kronecker: ⊗
import Base: Matrix

"""
Reshape MPO into matrix
"""
function Matrix(A::MPO)
    # contract link indices
    Acontracted = *([A[i] for i in eachindex(A)]...)

    # combine primed and unprimed site indices to make matrix
    sinds = hcat(siteinds(A)...)
    unprimed = sinds[1, :]
    primed = sinds[2, :]
    Cunprimed = combiner(unprimed...)
    Cprimed = combiner(primed...)
    Atensor = Cunprimed * Acontracted * Cprimed

    # convert 2D tensor into matrix
    return matrix(Atensor)
end

"""
Approximate a given density matrix with a quantum channel applied to a initial density
matrix. This is done by using non-linear optimization to finding optimal Kraus
operators.

min{Kᵢ} ‖∑ᵢKᵢρKᵢ† - ρ̃‖₂
s.t.    ∑ᵢKᵢ†Kᵢ = I
"""
function approxquantumchannel(ρ, ρ̃; nkraus::Union{Nothing,Int}=nothing)
    @assert size(ρ̃) == size(ρ) "Dimensions of ρ and ρ̃ must match"
    ndim = first(size(ρ))
    @assert ispow2(ndim) "Dimension of density matrix must be a power of 2"
    @assert ndims(ρ) in [2, 3] "Input must have 2 or 3 dimensions"

    # Make all inputs 3D tensors
    if ndims(ρ) == 2
        ρ = reshape(ρ, size(ρ)..., 1)
        ρ̃ = reshape(ρ̃, size(ρ)..., 1)
    end

    # Build Krauss operator variables
    model = Model(Ipopt.Optimizer)
    # complex array variables are not currently supported, so have to reshape
    nqubits = floor(Int64, log(2, ndim))
    maxnkraus = nqubits^2
    nkraus = isnothing(nkraus) ? maxnkraus : nkraus
    Ksdims = (ndim, ndim, nkraus)
    # the optimizer needs help with starting from a feasible point, so we initialize with
    # valid complex-valued Kraus operators.
    # TODO: explore effect of initialization on optimizations
    σy = [0.0 -1.0; 1.0 0.0]
    initK = 1 / sqrt(2) * (I + (1.0im * (σy ⊗ nqubits)))
    Ks = reshape([
            @variable(model, set = ComplexPlane(), start = sqrt(1 / nkraus) * initK[i, j])
            for (i, j, _) in Tuple.(CartesianIndices(Ksdims))
        ], Ksdims)

    # Define Krauss operators contraint: ∑ᵢKᵢ†Kᵢ = I
    @constraint(model, sum(K' * K for K in eachslice(Ks, dims=3)) .== I)

    # # Ensure that the Kraus operators are Hermitian
    # for i in 1:nkraus
    #     K = Ks[:, :, i]
    #     for x in 1:ndim
    #         for y in 1:ndim
    #             if x < y
    #                 @constraint(model, K[x, y] == K'[x, y])
    #             end
    #         end
    #     end
    # end

    # Find the difference between the approximation and tsvd matrix and compute the
    # Frobenius norm: ∑ᵢKᵢρKᵢ† - ρ̃.
    approxs = [
        @expression(model, sum(K * ρi * K' for K in eachslice(Ks, dims=3)))
        for ρi in eachslice(ρ, dims=3)]
    diffs = [@expression(model, approxs[i] - ρ̃[:, :, i]) for i in 1:length(approxs)]

    # Compute the Frobenius norm. This will have quartic terms, so we have to use
    # NLexpression. NLexpression does not yet support complex variables :(
    flatdiffs = Iterators.flatten(diffs)
    diffelems = vcat(real.(flatdiffs), imag.(flatdiffs))
    obj = @NLexpression(model, sum(el^2 for el in diffelems))
    @NLobjective(model, Min, obj)

    # Setup callback to record optimization iteration data
    iterdata = []
    function recorditerdata(data...)
        push!(iterdata, data)
        return true
    end
    MOI.set(model, Ipopt.CallbackFunction(), recorditerdata)

    # Let's optimize!
    optimize!(model)

    # Calculate initial objective value for comparison
    initloss = sum(norm.(ρ - ρ̃) .^ 2)

    optloss = objective_value(model)

    @show initloss
    @show optloss

    return value.(Ks), optloss, initloss, iterdata, model
end




# TODO: functions to: reshape matrix to tensor

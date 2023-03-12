# using ITensors
using LinearAlgebra
using TSVD
using JuMP
using Ipopt
using Kronecker: ⊗
import Base

"""
Convert MPO into matrix
"""
function Base.Matrix(A::MPO)
    # combine primed and unprimed site indices to make matrix
    sinds = hcat(siteinds(A)...)
    unprimed = sinds[1, :]
    primed = sinds[2, :]
    Cunprimed = combiner(unprimed...)
    Cprimed = combiner(primed...)
    Atensor = Cunprimed * *(A...) * Cprimed

    # convert 2D tensor into matrix
    return matrix(Atensor)
end

"""
Convert ITensor into matrix
"""
function toMatrix(A::ITensor, inds1, inds2)
    # get combiners from grouped indices
    C1 = combiner(inds1...)
    C2 = combiner(inds2...)
    Atensor = C1 * A * C2

    # convert 2D tensor into matrix
    return matrix(Atensor)
end

"""
Convert Array with combined dimensions into Tensor
"""
function toITensor(A, inds1, inds2, otherinds...)
    # get combiners from grouped indices
    C1 = combiner(inds1...)
    C2 = combiner(inds2...)

    # Convert matrix to ITensor
    ind1 = combinedind(C1)
    ind2 = combinedind(C2)
    t = ITensor(A, ind1, ind2, otherinds...)

    # Uncombine indices
    return t * C1 * C2
end

"""
Approximate a given density matrix with a quantum channel applied to a initial density
matrix. This is done by using non-linear optimization to finding optimal Kraus
operators.

min{Kᵢ} ‖∑ᵢKᵢρKᵢ† - ρ̃‖₂
s.t.    ∑ᵢKᵢ†Kᵢ = I
"""
function approxquantumchannel(ρ, ρ̃; nkraus::Union{Nothing,Int}=nothing, silent=false)
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
    silent ? set_silent(model) : nothing
    # complex array variables are not currently supported, so have to reshape
    nqubits = floor(Int64, log(2, ndim))
    maxnkraus = (2^nqubits)^2
    nkraus = isnothing(nkraus) ? maxnkraus : nkraus
    Ksdims = (ndim, ndim, nkraus)
    # the optimizer needs help with starting from a feasible point, so we initialize with
    # valid complex-valued Kraus operators.
    # TODO: explore effect of initialization on optimizations
    σy = [0.0 -1.0; 1.0 0.0]
    # initKs = cat(repeat(sqrt(1 / nkraus) * 1 / sqrt(2) * (I + (1.0im * (σy ⊗ nqubits))), outer=[1, 1, nkraus]), dims=3)
    ident = Array(I, ndim, ndim)
    zero = zeros(ndim, ndim)
    initKs = cat(ident, repeat(zero, outer=[1, 1, nkraus - 1]), dims=3)
    Ks = reshape([
            @variable(model, set = ComplexPlane(), start = initKs[i, j, k])   #sqrt(1 / nkraus) * initK[i, j])
            for (i, j, k) in Tuple.(CartesianIndices(Ksdims))
        ], Ksdims)

    # Define Krauss operators contraint: ∑ᵢKᵢ†Kᵢ = I
    @constraint(model, sum(K' * K for K in eachslice(Ks, dims=3)) .== I)

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

    return value.(Ks), optloss, initloss, iterdata, model
end

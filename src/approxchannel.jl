using ITensors
using LinearAlgebra
using TSVD
using JuMP
using Ipopt
using Kronecker

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
function approxquantumchannel(ρ, ρ̃, nkraus=nothing)
    @assert length(size(ρ)) == 2 "ρ must be a 2D matrix"
    @assert size(ρ̃) == size(ρ) "Dimensions of ρ and ρ̃ must match"
    ndim = first(size(ρ))
    @assert ispow2(ndim) "Dimension of density matrix must be a power of 2"

    # 1. Build Krauss operator variables
    model = Model(Ipopt.Optimizer)
    # complex array variables are not currently supported, so have to reshape
    nqubits = floor(Int64, log(2, ndim))
    maxnkraus = nqubits^2
    nkraus = isnothing(nkraus) ? maxnkraus : nkraus
    Ksdims = (ndim, ndim, nkraus)
    # Optimizer needs help with starting from a feasible point, so we initialize with
    # valid complex-valued Kraus operators.
    # TODO: explore effect of initialization on optimizations
    σy = [0.0 -1.0; 1.0 0.0]
    initK = 1 / sqrt(2) * ((I + 1.0im) * (σy ⊗ nqubits))
    Ks = reshape([
            @variable(model, set = ComplexPlane(), start = sqrt(1 / nkraus) * initK[i, j])
            for (i, j, _) in Tuple.(CartesianIndices(Ksdims))
        ], Ksdims)

    # 2. define Krauss operators contraint: ∑ᵢKᵢ†Kᵢ = I
    @constraint(model, sum(K' * K for K in eachslice(Ks, dims=3)) .== I)

    # 3. Find the difference between the approximation and tsvd matrix and compute Frobenius
    #    norm: ∑ᵢKᵢρKᵢ† - ρ̃.
    approx = @expression(model, sum(K * ρ * K' for K in eachslice(Ks, dims=3)))
    diff = @expression(model, approx - ρ̃)

    # 4. Compute the Frobenius norm. This will have quartic terms, so we have to use
    # NLexpression. NLexpression does not yet support complex variables :(
    diffreal = real(diff)
    diffimag = imag(diff)
    fnorm = @NLexpression(model,
        sum(diffreal[i]^2 for i in CartesianIndices(diffreal))
        +
        sum(diffimag[i]^2 for i in CartesianIndices(diffimag))
    )
    @NLobjective(model, Min, fnorm)

    # 5. setup callback to record optimization iteration data
    iterdata = []
    function recorditerdata(data...)
        push!(iterdata, data)
        return true
    end
    MOI.set(model, Ipopt.CallbackFunction(), recorditerdata)

    # 6. Let's optimize!
    optimize!(model)

    return value.(Ks), iterdata, model
end

# TODO: functions to: reshape matrix to tensor

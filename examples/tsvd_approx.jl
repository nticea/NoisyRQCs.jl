using LinearAlgebra
using TSVD
using JuMP
using Ipopt

# parameters
ndims_out = 16
npurestates = 3
nkraus = 4
truncdims = 2

# 1. generate random density matrix
# TODO: better sample distribution for densities
purestates = mapslices(x -> x / norm(x), rand(Complex{Float64}, ndims_out, npurestates), dims=1)
puredensities = Array([r * r' for r in eachslice(purestates, dims=2)])
weights = rand(Float64, npurestates)
weights = weights ./ sum(weights)
ρ = sum(puredensities .* weights)

# 2. Compute exact truncated SVD
# TODO: do actual bond dimension reduction
(U, s, V) = tsvd(ρ, truncdims)
ρ̃ = U * diagm(s) * V'

# 3. Approximate truncated density matrix with quantum operation by finding optimal Kraus
#    operators using non-convex optimization (quartic objective with quadratic constraint)
#
#                                    min{Kᵢ} ‖∑ᵢKᵢρKᵢ† - ρ̃‖₂
#                                    s.t.    ∑ᵢKᵢ†Kᵢ = I
model = Model(Ipopt.Optimizer)

# a. Build Krauss operator variables
# complex array variables are not currently supported, so have to reshape
Ksdims = (ndims_out, ndims_out, nkraus)
# Optimizer needs help with starting from a feasible point, using Kᵢ = I
Ks = reshape([
        @variable(model, set = ComplexPlane(), start = I[i, j])
        for (i, j, _) in Tuple.(CartesianIndices(Ksdims))
    ], Ksdims)

# b. define Krauss operators contraint: ∑ᵢKᵢ†Kᵢ = I
@constraint(model, sum(K' * K for K in eachslice(Ks, dims=3)) .== I)

# c. Find the difference between the approximation and tsvd matrix and compute Frobenius norm
#                                    ∑ᵢKᵢρKᵢ† - ρ̃.
approx = @expression(model, sum(K * ρ * K' for K in eachslice(Ks, dims=3)))
diff = @expression(model, approx - ρ̃)

# d. Compute the Frobenius norm. This will have quartic terms, so we have to use NLexpression
# NLexpression does not support complex variables :(
diffreal = real(diff)
diffimag = imag(diff)
fnorm = @NLexpression(model,
    sum(diffreal[i]^2 for i in CartesianIndices(diffreal))
    +
    sum(diffimag[i]^2 for i in CartesianIndices(diffimag))
)
@NLobjective(model, Min, fnorm)

# e. Let's optimize!
optimize!(model)

# 4. Process results
@show objective_value(model)
@show sum((ρ-ρ̃) .^ 2)

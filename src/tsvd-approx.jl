
using LinearAlgebra
using MadNLP
using TSVD

# generate random density matrix
ndims = 2
npurestates = 10
purestates = mapslices(x -> x / norm(x), rand(Complex{Float64}, ndims, npurestates), dims=1)
puredensities = Array([r * r' for r in eachslice(purestates, dims=2)])
weights = rand(Float64, numpurestates)
weights = weights ./ sum(weights)
r = sum(puredensities .* weights)

# TSVD
tdim = 1
(U, s, V) = tsvd(r, tdim)
rprime = U * diagm(s) * V'

# optimization
nkrauss = 100

model = Model(Ipopt.Optimizer)

# Krauss operator variables
# complex array variables are not currently  supported, so have to reshape
Ksdims = (ndims, ndims, nkrauss)
Kindices = CartesianIndices(Ksdims)
Kelems = [@variable(model, set = ComplexPlane(), start = I[i, j]) for (i, j, k) in Tuple.(Kindices)]
Ks = reshape(Kelems, Ksdims)

# Krauss operator contraints: Kᵢ†Kᵢ = 1
[@constraint(model, K' * K .== I) for K in eachslice(Ks, dims=3)]

# Find the difference between the approximation and tsvd matrix and compute Frobenius norm
#                                  ∑ᵢKᵢρKᵢ† - ρ̃.
approx = @expression(model, sum(K * r * K' for K in eachslice(Ks, dims=3)))
diff = @expression(model, approx - rprime)

# Compute the Frobenius norm. This will have quartic terms, so we have to use NLexpression
diffreal = real(diff)
diffimag = imag(diff)
fnorm = @NLexpression(model,
    sum(diffreal[i]^2 for i in CartesianIndices(diffreal))
    +
    sum(diffimag[i]^2 for i in CartesianIndices(diffimag))
)
@NLobjective(model, Min, fnorm)

optimize!(model)

@show objective_value(model)

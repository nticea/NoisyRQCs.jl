
using LinearAlgebra
using TSVD
using JuMP
using Ipopt

# 1. generate random density matrix
nsites = 4
bonddim = 10
sites = siteinds("S=1/2", nsites)
psi = randomMPS(sites, bonddim)
rho = density_matrix(psi)

# 2. Make truncated density matrix
truncatedbonddim = 2
trho = copy(rho)
NDTensors.truncate!(trho, maxdim=truncatedbonddim)

# Reshape tensor into matrix
# TODO: This is not memory efficient. Can we directly optimize the MPO?
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

ρ = Matrix(rho)
ρ̃ = Matrix(trho)

# TODO:
# Reshape matrix to tensor
# Make MPO from tensor

# 3. Approximate truncated density matrix with quantum operation by finding optimal Kraus
#    operators using non-convex optimization (quartic objective with quadratic constraint)
#
#                                 min{Kᵢ} ‖∑ᵢKᵢρKᵢ† - ρ̃‖₂
#                                 s.t.    ∑ᵢKᵢ†Kᵢ = I
model = Model(Ipopt.Optimizer)

# a. Build Krauss operator variables
# complex array variables are not currently supported, so have to reshape
nkraus = 4
Ksdims = (size(ρ)..., nkraus)
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
@show ρ
@show ρ̃
@show value.(diff)

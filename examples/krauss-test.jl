
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")

"""
Tests the channel approximation function.
"""

# 1. generate random density matrices
nsites = 2
bonddim = 100
sites = siteinds("Qubit", nsites)
psi = randomMPS(ComplexF64, sites, linkdims=bonddim)
rho = density_matrix(psi)

# 2. Make truncated density matrix
nbondstrunc = 1
centersite = nsites ÷ 2
startsite = centersite - (nbondstrunc ÷ 2)
siterange = startsite:(startsite+nbondstrunc)
truncatedbonddim = 1
trho = copy(rho)
NDTensors.truncate!(trho, maxdim=truncatedbonddim, site_range=siterange)

# 3. Find approximate quantum channel
nkraus = 6
ρ = Matrix(rho)
ρ̃ = Matrix(trho)
Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃, nkraus=nkraus)

@show (initloss - optloss) / initloss

# Check completeness
compl = +([Ki' * Ki for Ki in eachslice(Ks, dims=3)]...)
@assert compl ≈ I "Kraus array not complete!"

# Transform Kraus operator into tensor
krausidx = Index(last(size(Ks)), "Kraus")
Kraw = toITensor(Ks, prime.(sites), sites, krausidx)

# Build Kdag TODO: there must be a better way
Kdaglist = [Ki' for Ki in eachslice(Ks, dims=3)]
Kdagarr = reshape(reduce(hcat, Kdaglist), size(Ks)...)
Kdagraw = toITensor(Kdagarr, prime.(sites), sites, krausidx)

# Check completeness with matrices
res = prime(Kraw) * Kdagraw * δ(krausidx, prime(krausidx))

# Put Kraus tensor into canonical form
K = getcanonicalkraus(Kraw, krausidx)
Kdag = getcanonicalkraus(Kdagraw, krausidx)

# Check completeness with tensors
complete = apply(Kdag, *([δ(ind, prime(ind)) for ind in sites]...), apply_dag=true)
array(complete)

# Test channel, make sure loss it the same
approx = apply(K, rho, apply_dag=true)
@assert sum(norm.(array(*(approx...) - *(trho...))) .^ 2) ≈ optloss "Channel loss does not match optimization value!"

print("Completeness and channel tests passed!")

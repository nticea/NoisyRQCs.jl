
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/kraus.jl")

"""
Tests the channel approximation function.
"""
function checkcompleteness(K, sites)
    # Put Kraus tensor into canonical form
    # K = getcanonicalkraus(K)
    krausidx = getkrausind(K)

    # Check completeness with tensors
    Kdag = swapprime(dag(K), 0 => 1) * δ(krausidx, krausidx')
    complete = apply(Kdag, K)
    delt = *([δ(ind, ind') for ind in sites]...)
    @assert complete ≈ delt "Kraus tensor fails completeness condition"
end

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
krausidx = Index(last(size(Ks)), KRAUS_TAG)
K = toITensor(Ks, prime.(sites), sites, krausidx)

# Make sure loss is the same
approx = apply(K, rho, apply_dag=true)
@assert sum(norm.(array(*(approx...) - *(trho...))) .^ 2) ≈ optloss "Channel loss does not match optimization value!"

checkcompleteness(K, sites)
println("Truncation-approximating noise channel passed tests!")

# Test generated channels
sites = siteinds("Qubit", 3)

println("Testing depolarizing noise...")
K = depolarizing_noise(sites, 0.5)
checkcompleteness(K, sites)
println("Depolarizing noise channel passed tests!")

println("Testing dephasing noise...")
K = dephasing_noise(sites, 0.5)
checkcompleteness(K, sites)
println("Dephasing noise channel passed tests!")

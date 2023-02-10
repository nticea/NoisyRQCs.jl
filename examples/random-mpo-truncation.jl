
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ITensors
using Plots

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")

"""
Tests the channel approximation function.
"""

# generate random density matrices
nsites = 2
bonddim = 1000
sites = siteinds("Qubit", nsites)
psi = randomMPS(ComplexF64, sites, linkdims=bonddim)
rho = density_matrix(psi)

# Make truncated density matrix
nbondstrunc = 1
centersite = nsites ÷ 2
startsite = centersite - (nbondstrunc ÷ 2)
siterange = startsite:(startsite+nbondstrunc)
truncatedbonddim = 1
trho = copy(rho)
NDTensors.truncate!(trho, maxdim=truncatedbonddim, site_range=siterange)

# Find approximate quantum channel
nkraus = 6
ρ = Matrix(rho)
ρ̃ = Matrix(trho)
Ks, optloss, initloss, iterdata, model = approxquantumchannel(ρ, ρ̃, nkraus=nkraus)

@show (initloss - optloss) / initloss

# Transform Kraus operator into tensor
krausidx = Index(last(size(Ks)), "Kraus")
Kraw = toITensor(Ks, prime.(sites), sites, krausidx)
K = getcanonicalkraus(Kraw, krausidx)

# Plot norm distribution. The U should be a diagonal matrix with entries with norm 1.
U, S, V = svd(K, krausidx)
mags = array(diag(S))
relmags = mags / sum(mags)
p = bar(
    relmags,
    ylim=[0, 1],
    ylabel="Relative magnitude",
    xlabel="Kraus operators",
    title="Relative magnitudes of Kraus operators",
    legend=:none,
)

# Plot pauli decompositions

"""
Ideas:

Average over many MPDOs

Look at:
- distribution of norms
- clustering of pauli decompositions vs random noise

optimal loss with 2 vs 4 site Kraus operators

difference in entanglement entropy between truncated and approximation

different truncation dimensions
"""

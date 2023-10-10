
using ITensors

include("../src/circuit.jl")
include("../src/utilities.jl")
include("../src/approxchannel.jl")
include("../src/kraus.jl")

noises = 0:0.1:1

losses = []
approxes = []
# for ϵ in noises
# make density
nsites = 2
sites = siteinds("Qubit", nsites)
psi = randomMPS(ComplexF64, sites, linkdims=100)
init = density_matrix(psi)

# make dephasing channel
noise = 0.3
K = dephasing_noise(sites, noise)

# get density after channel
final = apply(K, init, apply_dag=true)

# approximate channel
nkraus = 4
Kapprox, optloss, initloss, iterdata, model = approxquantumchannel(init, final; nkraus, silent=false)
#     push!(approxes, Kapprox)
#     push!(losses, (optloss, initloss))
# end

function plotkraus(K, sites)
    # Get norm distribution
    pdecomp, relnorms = analyzekraus(K, sites)

    # Kraus operator norms
    barplot = bar(
        relnorms,
        ylabel="Relative Frobenius norm",
        xlabel="Kraus operators",
        title="Relative Frobenius norms of Kraus operators",
        titlefont=font(11),
        legend=:none,
    )

    # Kraus operator Pauli decomposition
    pdnorms = norm.(pdecomp)
    pdplot = plot_paulidecomp(pdnorms)

    return barplot, pdplot
end

# i = 10
# noise = noises[i]
# K = approxes[i]
# loss = losses[i]
K = Kapprox

barplot, pdplot = plotkraus(Kapprox, sites)
plot(barplot)
plot(pdplot)

nsites = 2
sites = siteinds("Qubit", nsites)
K = dephasing_noise(sites, noise)
barplot, pdplot = plotkraus(K, sites)
plot(barplot)
plot(pdplot)


## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include("../src/circuit.jl")
include("../src/utilities.jl")

ITensors.set_warn_order(50)

## LOADING IN RESULTS ##
dir = "/Users/nicole/Dropbox/Grad school/Vedika/noisy_RQCs/NoisyRQCs.jl/outputs/benchmark/"
fnames = filter(contains(r".h5$"), readdir(dir; join=true))
r1 = load_results(fnames[1])
L = r1.L
T = r1.T

bitdist = zeros(length(fnames), 4^L)
entropy = zeros(length(fnames), T)
for (n,fn) in enumerate(fnames)
    results = load_results(fn)
    bitdist[n,:] = results.bitdist
    entropy[n,:] = results.entropy
end

_porter_thomas_fit(vec(bitdist), 4^L, true)
entropy_avg = vec(mean(entropy, dims=1))
plot_entropy(entropy_avg)
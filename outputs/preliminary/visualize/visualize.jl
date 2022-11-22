
## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,"../../.."))
include("../../../src/circuit.jl")
include("../../../src/utilities.jl")

ITensors.set_warn_order(50)

## LOADING IN RESULTS ##
dir = "/Users/nicole/sherlock/code/NoisyRQCs.jl/outputs/preliminary/results"
fnames = filter(contains(r".h5$"), readdir(dir; join=true))

# Sort the files by conditions 



# bitdist = zeros(length(fnames), 4^L)
# entropy = zeros(length(fnames), T)
# for (n,fn) in enumerate(fnames)
#     results = load_results(fn)
#     bitdist[n,:] = results.bitdist
#     entropy[n,:] = results.entropy
# end

# _porter_thomas_fit(vec(bitdist), 2^L, true)
# entropy_avg = vec(mean(entropy, dims=1))
# plot_entropy(entropy_avg, L)
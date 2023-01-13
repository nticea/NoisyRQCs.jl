
## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__,"../../.."))
include("../../../src/circuit.jl")
include("../../../src/utilities.jl")
using Plots 

ITensors.set_warn_order(50)

## LOADING IN RESULTS ##
dir = "/Users/nicole/Dropbox/Grad school/Vedika/noisy_RQCs/NoisyRQCs.jl/outputs/preliminary/results"
fnames = filter(contains(r".h5$"), readdir(dir; join=true))

# Sort the files by conditions 
εs = [0, 0.1, 0.01]
maxdims = [500, 1000, nothing]

# Some formatting for the plots
ls = [:solid, :dash, :dot]
color = ["blue", "green", "orange"]

# Initialize the plots
global p1 = plot() # midpoint operator entropy
global p2 = plot() # trace 
global p3 = plot() # entanglement entropy as a function of position in the chain 

global L 
global T 

for (i,ε) in enumerate(εs)
    if ε==0 # just some dumb accounting to make sure I can read the fname in properly 
        ε = Int(ε)
    end
    for (j,maxdim) in enumerate(maxdims)
        # get the filenames coresponding to these conditions 
        fns = filter(contains("$(ε)ε_$(maxdim)maxdim"), fnames)

        # extract some parameters 
        r1 = load_results(fns[1])
        global L = r1.L
        global T = r1.T

        # get average values for every observable 
        nsamples = length(fns)
        @show ε, maxdim, nsamples 
        bitdist = zeros(nsamples, 4^L)
        state_entropy = zeros(nsamples, T)
        operator_entanglement = zeros(nsamples, T, L-3)
        trace = zeros(nsamples, T)
        for (n,fn) in enumerate(fns) 
            r = load_results(fn)
            bitdist[n,:] = r.bitdist
            state_entropy[n,:] = r.state_entropy
            operator_entanglement[n,:,:] = r.operator_entanglement
            trace[n,:] = r.trace 
        end
        # take an average over the appropriate dimensions 
        operator_entanglement_avg = mean(operator_entanglement, dims=1)[1,:,:]
        state_entropy_avg = vec(mean(state_entropy, dims=1))
        midpoint_op_entanglement = operator_entanglement_avg[:, floor(Int, (L-3)/2)]
        evolved_op_entanglement = operator_entanglement_avg[end,:]
        trace_avg = vec(mean(trace, dims=1))
        
        # plot everything out 
        global p1 = plot!(p1, midpoint_op_entanglement, linestyle=ls[i], color=color[j],
                            label="$(ε) ε, $(maxdim) maxdim")
        global p2 = plot!(p2, evolved_op_entanglement, linestyle=ls[i], color=color[j],
                            label="$(ε) ε, $(maxdim) maxdim")
        global p3 = plot!(p3, trace_avg, linestyle=ls[i], color=color[j],
                            label="$(ε) ε, $(maxdim) maxdim")
    end
end

title!(p1, "Midpoint operator entanglement")
xlabel!(p1, "Time")
title!(p2, "Operator entanglement at T=$(T)")
xlabel!(p2, "Position in chain")
title!(p3, "Trace of ρ")
xlabel!(p3, "Time")
p = plot(p1, p2, p3, layout=Plots.grid(1,3, widths=(1/3,1/3,1/3)), size=(2500,750))
#savefig(p, "results.pdf")
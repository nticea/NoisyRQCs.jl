
## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
include("../../src/circuit.jl")
include("../../src/utilities.jl")
include("../../src/results.jl")
using StatsBase
ITensors.set_warn_order(50)

# get all loadpaths with system size L 
L = 9
εs, max_inner_dims, results = [], [], []

files = readdir(joinpath(@__DIR__, "data"))
for f in files
    if startswith(f, "results_$(L)Nx")
        res = load_results(joinpath(@__DIR__, "data", f))
        L̃, T, ε, maxdim, max_inner_dim, st_ent, op_ent, trace, logneg, mutual_info = splat_struct(res)

        @assert L̃ == L
        map(push!, [εs, max_inner_dims, results], [ε, max_inner_dim, res])
    end
end

εs = unique(εs)
max_inner_dims = unique(max_inner_dims)

## PLOTTING ## 

global p1 = plot()
global p2 = plot()
global p3 = plot()
global p4 = plot()
global hmaps = []

ε_cmap = cgrad(:Set1_4, 4, categorical=true)
innerdim_ls = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dashdot, :solid, :dash]

for (i, ε) in enumerate(εs)
    for (j, max_inner_dim) in enumerate(max_inner_dims)

        # find the results that match these parameters 
        # average over them 
        st_ents = []
        op_ents = []
        traces = []
        lognegs = []
        mutual_infos = []

        for r in results
            L̃, T, ε̃, maxdim, max_inner_dim2, st_ent, op_ent, trace, logneg, mutual_info = splat_struct(r)
            if ε == ε̃ && max_inner_dim2 == max_inner_dim
                map(push!, [st_ents, op_ents, traces, lognegs, mutual_infos], [st_ent, op_ent, trace, logneg, mutual_info])
            end
        end

        if length(st_ents) > 0
            # state entropy
            toplot = average_elems(st_ents)
            global p1 = plot!(p1, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(max_inner_dim)",
                c=ε_cmap[i], ls=innerdim_ls[j], title="Second Rényi Entropy of State at L/2", legend=:bottomright)

            # operator entropy 
            S = average_elems(op_ents)
            mid = floor(Int, L / 2)
            if size(S)[1] != L
                S = transpose(S)
            end
            toplot = S[mid, :]
            global p2 = plot!(p2, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(max_inner_dim)",
                c=ε_cmap[i], ls=innerdim_ls[j], title="Operator Entanglement Entropy at L/2", legend=:bottomright)

            # trace
            toplot = average_elems(traces)
            global p3 = plot!(p3, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(max_inner_dim)",
                c=ε_cmap[i], ls=innerdim_ls[j], title="Trace", legend=:bottomright, ylims=(0.7, 1.3))

            # logarithmic negativity
            toplot = average_elems(lognegs)
            global p4 = plot!(p4, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(max_inner_dim)",
                c=ε_cmap[i], ls=innerdim_ls[j], title="Logarithmic negativity", legend=:bottomright)

            # mutual information
            mutual_info = average_elems(mutual_infos)
            toplot = mutual_info[:, 2:end]
            push!(hmaps, heatmap(toplot, xticks=collect(1:L), xlabel="Distance (sites)", ylabel="Time", title="ε=$ε, max inner dim=$max_inner_dim", clims=(0, 1)))
        end

    end
end
#p1 = hline!(p1, [saturation_value(L)], label="saturation value")

# make superplots
bigplot1 = plot(p1, p2, p3, p4, layout=Plots.grid(2, 2, widths=[1 / 2, 1 / 2]), size=(1250, 1000))
bigplot2 = plot(hmaps..., layout=Plots.grid(3, 3, widths=[1 / 3, 1 / 3, 1 / 3]), size=(2000, 1500))

# # save 
# savefig(bigplot1, "entanglement_mpdo_L$L.png")
# savefig(bigplot2, "mutual_info_mpdo_L$L.png")

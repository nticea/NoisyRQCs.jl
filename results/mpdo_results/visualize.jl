## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
include("../../src/circuit.jl")
include("../../src/utilities.jl")
include("../../src/results.jl")
using StatsBase, StatsPlots, DataFrames
ITensors.set_warn_order(50)

## THIS IS THE ONLY INPUT PARAMETER ## 
L = 9

## SCRIPT ## 
datapath = joinpath(@__DIR__, "data")
εs, max_inner_dims, maxdims, results = [], [], [], []
files = readdir(datapath)
for f in files
    if startswith(f, "results_$(L)L")
        res = load_results(joinpath(datapath, f))
        _, L̃, T, ε, maxdim, max_inner_dim, st_ent, op_ent, trace, logneg, mutual_info = splat_struct(res)
        @assert L̃ == L
        map(push!, [εs, max_inner_dims, maxdims, results], [ε, max_inner_dim, maxdim, res])
    end
end

εs = unique(εs)
max_inner_dims = unique(max_inner_dims)
maxdims = unique(maxdims)

## PLOTTING ## 

global p1 = plot()
global p2 = plot()
global p3 = plot()
global p4 = plot()
global hmaps = []

ε_cmap = cgrad(:Set1_9, 9, categorical=true)
innerdim_ls = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dashdot, :solid, :dash]

for (i, ε) in enumerate(εs)
    for (j, max_inner_dim) in enumerate(max_inner_dims)
        for (k, maxdim) in enumerate(maxdims)

            # find the results that match these parameters 
            # average over them 
            st_ents = []
            op_ents = []
            traces = []
            lognegs = []
            mutual_infos = []

            for r in results
                _, L̃, T, ε̃, maxdim2, max_inner_dim2, st_ent, op_ent, trace, logneg, mutual_info = splat_struct(r)
                if ε == ε̃ && max_inner_dim2 == max_inner_dim && maxdim2 == maxdim
                    map(push!, [st_ents, op_ents, traces, lognegs, mutual_infos], [st_ent, op_ent, trace, logneg, mutual_info])
                end
            end

            if max_inner_dim == 0
                label = "MPO: ε=$(ε), χ=$maxdim"
            else
                label = "MPDO: ε=$(ε), κ=$(max_inner_dim), χ=$maxdim"
            end


            if length(st_ents) > 0
                # state entropy
                toplot = average_elems(st_ents)
                global p1 = plot!(p1, 1:length(toplot), toplot, label=label,
                    c=ε_cmap[k], ls=innerdim_ls[j], title="Second Rényi Entropy of State at L/2 (L=$L)", legend=:bottomright)

                # operator entropy 
                S = average_elems(op_ents)
                mid = floor(Int, L / 2)
                if size(S)[1] != L
                    S = transpose(S)
                end
                toplot = S[mid, :]
                global p2 = plot!(p2, 1:length(toplot), toplot, label=label,
                    c=ε_cmap[k], ls=innerdim_ls[j], title="Operator Entanglement Entropy at L/2 (L=$L)", legend=:bottomright)

                # trace
                toplot = average_elems(traces)
                global p3 = plot!(p3, 1:length(toplot), toplot, label=label,
                    c=ε_cmap[k], ls=innerdim_ls[j], title="Trace (L=$L)", legend=:bottomright)

                # logarithmic negativity
                toplot = average_elems(lognegs)
                global p4 = plot!(p4, 1:length(toplot), toplot, label=label,
                    c=ε_cmap[k], ls=innerdim_ls[j], title="Logarithmic negativity (L=$L)", legend=:bottomright)

                # mutual information
                mutual_info = average_elems(mutual_infos)
                toplot = mutual_info[:, 2:end]
                push!(hmaps, heatmap(toplot, xticks=collect(1:L), xlabel="Distance (sites)", ylabel="Time", title="Mutual Information: " * label, clims=(0, 1)))
            end
        end
    end
end
p1 = hline!(p1, [saturation_value(L)], label="saturation value", color="red")

# make superplots
bigplot1 = plot(p1, p2, p3, p4, layout=Plots.grid(2, 2, widths=[1 / 2, 1 / 2]), size=(1250, 1000))
bigplot2 = plot(hmaps..., layout=Plots.grid(4, 3, widths=[1 / 3, 1 / 3, 1 / 3]), size=(2000, 1500))

# Performance
benchmarkpath = joinpath(datapath, "performance_L$L.csv")
df_performance = load_performance_dataframe(benchmarkpath)

perf1 = @df df_performance plot(
    :max_outer_dim,
    :bytes,
    group=:max_inner_dim,
    m=(0.75, [:+ :h :star7 :circle], 5),
)
title!("Memory usage")
xlabel!("Max outer dimension")
ylabel!("Bytes")

perf2 = @df df_performance plot(
    :max_outer_dim,
    :time,
    group=:max_inner_dim,
    m=(0.75, [:+ :h :star7 :circle], 5),
)
title!("Runtime")
xlabel!("Max outer dimension")
ylabel!("Time")

bigplot3 = plot(perf1, perf2, layout=Plots.grid(1, 2, widths=[1 / 2, 1 / 2]), size=(1250, 1000))

## SAVING INFO ## 
# savefig(bigplot1, "entanglement_$(L)L.png")
# savefig(bigplot2, "mutual_info_$(L)L.png")
# savefig(bigplot3, "performance_$(L)L.png")

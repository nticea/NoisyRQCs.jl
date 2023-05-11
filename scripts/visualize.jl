
## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/circuit.jl")
include("../src/utilities.jl")

ITensors.set_warn_order(50)

# get all loadpaths with system size L 
L = 9
files = readdir(@__DIR__)
for f in files
    if startswith(f, "$(L)Nx_results.h5")
        results = load_results(joinpath(@__DIR__, f))
        L̃, T, ε, maxdim, max_inner_dim, st_ent, op_ent, trace, logneg, mutual_info = splat_struct(results)
        @assert L̃ == L
    end
    map(push!, [εs, max_inner_dims], [ε, max_inner_dim])
end

global p1 = plot()
global p2 = plot()
global p3 = plot()
global p4 = plot()
global p5 = plot()

ε_cmap = cgrad(:Set1_4, 4, categorical=true)
innerdim_ls = [:solid, :dash, :dot]

for (i, ε) in enumerate(εs)
    for (j, maxdim) in enumerate(max_inner_dims)
        # state entropy
        toplot = st_ents[i][j]
        global p1 = plot!(p1, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Second Rényi Entropy of State at L/2", legend=:bottomright)

        # operator entropy 
        S = op_ents[i][j]
        mid = floor(Int, L / 2)
        if size(S)[1] != L
            S = transpose(S)
        end
        toplot = S[mid, :]
        global p2 = plot!(p2, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Operator Entanglement Entropy at L/2", legend=:bottomright)

        # trace 
        toplot = traces[i][j]
        global p3 = plot!(p3, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Trace", legend=:bottomright)

        # logarithmic negativity
        toplot = lognegs[i][j]
        global p4 = plot!(p4, 1:length(toplot), toplot, label="ε=$(ε), max inner dim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Logarithmic negativity", legend=:bottomright)

        # mutual information
        toplot = mutual_infos[i][j]
        global p5 = heatmap(toplot, xlabel="Distance (sites)", ylabel="Time", title="Mutual Information")

    end
end

p1 = hline!(p1, [saturation_value(L)])
plot(p1, p2, p3, layout=Plots.grid(1, 3, widths=[1 / 3, 1 / 3, 1 / 3]), size=(2000, 500))
plot(p4, p5, layout=Plots.grid(1, 2, widths=[1 / 2, 1 / 2]), size=(1400, 500))
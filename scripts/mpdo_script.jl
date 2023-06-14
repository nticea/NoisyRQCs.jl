## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")
include("../src/results.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 20
εs = [0, 1e-2, 1e-3]
maxdim = 14
max_inner_dims = [4, 8, 12, 16]
normalize_ρ = true

function mpdo_circuit_run(; L, T, maxdim, εs, max_inner_dims, normalize_ρ::Bool=true)
    # Initialize the wavefunction to product state (all 0)
    ψ0 = initialize_wavefunction(L=L)

    st_ents = []
    op_ents = []
    traces = []
    lognegs = []
    mutual_infos = []

    for ε in εs
        stents = []
        opents = []
        ts = []
        lns = []
        MIs = []
        for max_inner_dim in max_inner_dims
            @show ε, max_inner_dim
            # Apply the MPDO circuit
            ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim,
                max_inner_dim=max_inner_dim, benchmark=true, normalize_ρ=normalize_ρ)
            push!(stents, state_entanglement)
            push!(opents, operator_entanglement)
            push!(ts, trace)
            push!(lns, logneg)
            push!(MIs, MI)
        end
        push!(st_ents, stents)
        push!(op_ents, opents)
        push!(traces, ts)
        push!(lognegs, lns)
        push!(mutual_infos, MIs)
    end

    return st_ents, op_ents, traces, lognegs, mutual_infos
end

# actually run the script 
st_ents, op_ents, traces, lognegs, mutual_infos = mpdo_circuit_run(L=L, T=T, maxdim=maxdim, εs=εs, max_inner_dims=max_inner_dims, normalize_ρ=normalize_ρ)

## PLOTTING ## 

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

p1 = hline!(p1, [saturation_value(L)], label=["saturation value"])
plot(p1, p2, p3, p4, layout=Plots.grid(2, 2, widths=[1 / 2, 1 / 2]), size=(1500, 1500))
#plot(p4, p5, layout=Plots.grid(1, 2, widths=[1 / 2, 1 / 2]), size=(1400, 500))

# plot_entropy(state_entanglement, L, title="MPDO Second Renyi entropy, ε=$(ε)")
# plot_operator_entanglement(op_entanglement, L, title="MPDO operator entropy, ε=$(ε)")
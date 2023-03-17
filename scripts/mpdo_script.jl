## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")
include("../src/results.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 100
εs = [0]
maxdim = 16
max_inner_dims = [16]
normalize_ρ = true

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

global st_ents = []
global op_ents = []
global traces = []
global lognegs = []
global mutual_infos = []
for ε in εs
    stents = []
    opents = []
    ts = []
    lns = []
    MIs = []
    for max_inner_dim in max_inner_dims
        @show ε, max_inner_dim
        # Apply the MPDO circuit
        ψ, state_entanglement, operator_entanglement, logneg, MI, trace = apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim,
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

## PLOTTING ## 

global p1 = plot()
global p2 = plot()
global p3 = plot()

ε_cmap = cgrad(:Set1_4, 4, categorical=true)
innerdim_ls = [:solid, :dash, :dot]

for (i, ε) in enumerate(εs)
    for (j, max_inner_dim) in enumerate(max_inner_dims)
        # state entropy
        toplot = st_ents[i][j]
        global p1 = plot!(p1, 1:length(toplot), toplot, label="ε=$(ε), innerdim=$(max_inner_dim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Second Rényi Entropy of State at L/2", legend=:bottomright)

        # operator entropy 
        S = op_ents[i][j]
        mid = floor(Int, L / 2)
        if size(S)[1] != L
            S = transpose(S)
        end
        toplot = S[mid, :]
        global p2 = plot!(p2, 1:length(toplot), toplot, label="ε=$(ε), innerdim=$(max_inner_dim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Operator Entanglement Entropy at L/2", legend=:bottomright)

        # trace 
        toplot = traces[i][j]
        global p3 = plot!(p3, 1:length(toplot), toplot, label="ε=$(ε), innerdim=$(max_inner_dim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Trace", legend=:bottomright)

        # logarthmic negativity 
        toplot = lognegs[i][j][:, 2:end]
        global p4 = heatmap(toplot)

        # logarthmic negativity 
        toplot = mutual_infos[i][j][1:20, 2:end]
        global p4 = heatmap(toplot)

    end
end

plot(p1, p2, p3, layout=Plots.grid(1, 3, widths=[1 / 3, 1 / 3, 1 / 3]), size=(2000, 500))

# plot_entropy(state_entanglement, L, title="MPDO Second Renyi entropy, ε=$(ε)")
# plot_operator_entanglement(op_entanglement, L, title="MPDO operator entropy, ε=$(ε)")
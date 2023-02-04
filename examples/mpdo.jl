## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")
include("../src/results.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 100
εs = [0, 1e-4, 1e-2]
maxdim = 16
max_inner_dims = [1, 2]

# Initialize the wavefunction to product state (all 0)
ψ0 = initialize_wavefunction(L=L)

global st_ents = []
global op_ents = []
global traces = []
for ε in εs
    stents = []
    opents = []
    ts = []
    for max_inner_dim in max_inner_dims
        # Apply the MPDO circuit
        ψ, state_entanglement, op_entanglement, trace = apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim, max_inner_dim=max_inner_dim, benchmark=true)
        push!(stents, state_entanglement)
        push!(opents, op_entanglement)
        push!(ts, trace)
    end
    push!(st_ents, stents)
    push!(op_ents, opents)
    push!(traces, ts)
end

p1 = plot()
p2 = plot()
ε_cmap = cgrad(:Set1_4, length(εs), categorical=true)
innerdim_ls = [:solid, :dash, :dot]

for (i, ε) in enumerate(εs)
    for (j, max_inner_dim) in enumerate(max_inner_dims)
        @show ε, max_inner_dim
        # state entropy
        toplot = st_ents[i][j]
        p1 = plot!(p1, 1:length(toplot), toplot, label="ε=$(ε), innerdim=$(max_inner_dim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Second Rényi Entropy of State at L/2", legend=:bottomright)

        # operator entropy 
        S = op_ents[i][j]
        mid = floor(Int, L / 2)
        if size(S)[1] != L
            S = transpose(S)
        end
        toplot = S[mid, :]
        p2 = plot!(p2, 1:length(toplot), toplot, label="ε=$(ε), innerdim=$(max_inner_dim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Operator Entanglement Entropy at L/2", legend=:bottomright)
    end
end

plot(p1, p2, layout=Plots.grid(1, 2, widths=[1 / 2, 1 / 2]), size=(1500, 500))

# plot_entropy(state_entanglement, L, title="MPDO Second Renyi entropy, ε=$(ε)")
# plot_operator_entanglement(op_entanglement, L, title="MPDO operator entropy, ε=$(ε)")
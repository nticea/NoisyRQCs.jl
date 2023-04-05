## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/circuit.jl")
include("../src/results.jl")
using Random

ITensors.set_warn_order(50)
# Random.seed!(123456)

## PARAMETERS ## 
L = 9
T = 100
εs = [0, 1e-2, 1e-3]
maxdims = [nothing, 250, 200]

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
    for maxdim in maxdims
        @show ε, maxdim
        # Apply the MPDO circuit
        ρ, state_entanglement, operator_entanglement, logneg, MI, trace = apply_circuit(ψ0, T; random_type="Haar", ε=ε, benchmark=true, maxdim=maxdim)
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
global p4 = plot()
global p5 = plot()

ε_cmap = cgrad(:Set1_4, 4, categorical=true)
innerdim_ls = [:solid, :dash, :dot]

for (i, ε) in enumerate(εs)
    for (j, maxdim) in enumerate(maxdims)
        # state entropy
        toplot = st_ents[i][j]
        global p1 = plot!(p1, 1:length(toplot), toplot, label="ε=$(ε), maxdim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Second Rényi Entropy of State at L/2", legend=:bottomright)

        # operator entropy 
        S = op_ents[i][j]
        mid = floor(Int, L / 2)
        if size(S)[1] != L
            S = transpose(S)
        end
        toplot = S[mid, :]
        global p2 = plot!(p2, 1:length(toplot), toplot, label="ε=$(ε), maxdim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Operator Entanglement Entropy at L/2", legend=:bottomright)

        # trace 
        toplot = traces[i][j]
        global p3 = plot!(p3, 1:length(toplot), toplot, label="ε=$(ε), maxdim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Trace", legend=:bottomright)

        # logarithmic negativity
        toplot = lognegs[i][j]
        global p4 = plot!(p4, 1:length(toplot), toplot, label="ε=$(ε), maxdim=$(maxdim)",
            c=ε_cmap[i], ls=innerdim_ls[j], title="Logarithmic negativity", legend=:bottomright)

        # mutual information
        toplot = mutual_infos[i][j]
        global p5 = heatmap(toplot, xlabel="Distance (sites)", ylabel="Time", title="Mutual Information")

    end
end
p1 = hline!(p1, [saturation_value(L)])
plot(p1, p2, p3, layout=Plots.grid(1, 3, widths=[1 / 3, 1 / 3, 1 / 3]), size=(2000, 500))
plot(p4, p5, layout=Plots.grid(1, 2, widths=[1 / 2, 1 / 2]), size=(1400, 500))

# plot_entropy(state_entanglement, L, title="MPDO Second Renyi entropy, ε=$(ε)")
# plot_operator_entanglement(op_entanglement, L, title="MPDO operator entropy, ε=$(ε)")

# averaging for MIs
# MI = [mutual_infos[i][1] for i in 1:size(mutual_infos)[1]]
# MInew = zeros(length(MI), size(MI[1])...)
# for i in 1:length(MI)
#     MInew[i, :, :] = MI[i]
# end
# MIavg = mean(MInew, dims=1)
# MIavg = MIavg[1, :, :]
# heatmap(MIavg)
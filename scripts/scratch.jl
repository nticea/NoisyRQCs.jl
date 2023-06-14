## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../src/MPDO.jl")
include("../src/circuit.jl")
include("../src/results.jl")

ITensors.set_warn_order(50)

## PARAMETERS ## 
L = 9
T = 20
ε = 1e-3
maxdim_mpdo = 4
max_inner_dim = 4
normalize_ρ = true
benchmark = true

niter = 3

# Initialize the state 
ψ0 = initialize_wavefunction(L=L)

state_entanglement_mpdo = zeros(niter, T)
operator_entanglement_mpdo = zeros(niter, T)
logneg_mpdo = zeros(niter, T)
MI_mpdo = zeros(niter, T)
trace_mpdo = zeros(niter, T)

state_entanglement_mpo = zeros(niter, T)
operator_entanglement_mpo = zeros(niter, T)
logneg_mpo = zeros(niter, T)
MI_mpo = zeros(niter, T)
trace_mpo = zeros(niter, T)

state_entanglement_true = zeros(niter, T)
operator_entanglement_true = zeros(niter, T)
logneg_true = zeros(niter, T)
MI_true = zeros(niter, T)
trace_true = zeros(niter, T)

for i in 1:niter

    # Apply the MPDO circuit
    ψ, st_ent, op_ent, logneg, mutual_info, trace = @time apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=maxdim_mpdo,
        max_inner_dim=max_inner_dim, benchmark=benchmark, normalize_ρ=normalize_ρ)

    state_entanglement_mpdo[i, :] = st_ent
    operator_entanglement_mpdo[i, :] = op_ent[:, floor(Int, L / 2)]
    logneg_mpdo[i, :] = logneg
    #MI_mpdo[i, :] = mutual_info
    trace_mpdo[i, :] = trace

    #Also do the MPO circuit 
    ρ, st_ent, op_ent, logneg, mutual_info, trace = @time apply_circuit(ψ0, T; random_type="Haar", ε=ε, benchmark=benchmark, maxdim=maxdim_mpdo^2)

    state_entanglement_mpo[i, :] = st_ent
    operator_entanglement_mpo[i, :] = op_ent[:, floor(Int, L / 2)]
    logneg_mpo[i, :] = logneg
    #MI_mpo[i, :] = mutual_info
    trace_mpo[i, :] = trace

    ρ, st_ent, op_ent, logneg, mutual_info, trace = @time apply_circuit(ψ0, T; random_type="Haar", ε=ε, benchmark=benchmark, maxdim=nothing)

    state_entanglement_true[i, :] = st_ent
    operator_entanglement_true[i, :] = op_ent[:, floor(Int, L / 2)]
    logneg_true[i, :] = logneg
    #MI_true[i, :] = mutual_info
    trace_true[i, :] = trace
end

# do all averages 
state_entanglement_mpdo = mean(state_entanglement_mpdo, dims=1)'
operator_entanglement_mpdo = mean(operator_entanglement_mpdo, dims=1)'
logneg_mpdo = mean(logneg_mpdo, dims=1)'
MI_mpdo = mean(MI_mpdo, dims=1)'
trace_mpdo = mean(trace_mpdo, dims=1)'

state_entanglement_mpo = mean(state_entanglement_mpo, dims=1)'
operator_entanglement_mpo = mean(operator_entanglement_mpo, dims=1)'
logneg_mpo = mean(logneg_mpo, dims=1)'
MI_mpo = mean(MI_mpo, dims=1)'
trace_mpo = mean(trace_mpo, dims=1)'

state_entanglement_true = mean(state_entanglement_true, dims=1)'
operator_entanglement_true = mean(operator_entanglement_true, dims=1)'
logneg_true = mean(logneg_true, dims=1)'
MI_true = mean(MI_true, dims=1)'
trace_true = mean(trace_true, dims=1)'

# Plot things to compare 
p1 = plot(1:T, state_entanglement_mpdo, title="State entanglement", label="mpdo")
p1 = plot!(p1, 1:T, state_entanglement_mpo, title="State entanglement", label="mpo (truncated)")
p1 = plot!(p1, 1:T, state_entanglement_true, title="State entanglement", label="mpo (true)")

p2 = plot(1:T, operator_entanglement_mpdo, title="Operator entanglement", label="mpdo")
p2 = plot!(p2, 1:T, operator_entanglement_mpo, title="Operator entanglement", label="mpo (truncated)")
p2 = plot!(p2, 1:T, operator_entanglement_true, title="Operator entanglement", label="mpo (true)")

p3 = plot(1:T, trace_mpdo, title="Trace", label="mpdo")
p3 = plot!(p3, 1:T, trace_mpo, title="Trace", label="mpo (truncated)")
p3 = plot!(p3, 1:T, trace_true, title="Trace", label="mpo (true)")

p4 = plot(1:T, logneg_mpdo, title="Logarithmic negativity", label="mpdo")
p4 = plot!(p4, 1:T, logneg_mpo, title="Logarithmic negativity", label="mpo (truncated)")
p4 = plot!(p4, 1:T, logneg_true, title="Logarithmic negativity", label="mpo (true)")

p = plot(p1, p2, p3, p4, layout=Plots.grid(2, 2, widths=[1 / 2, 1 / 2]), size=(1250, 1000))
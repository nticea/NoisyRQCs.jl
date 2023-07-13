## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/circuit.jl")
include("../../src/results.jl")
using Dates
using StatsPlots

ITensors.set_warn_order(50)

## PARAMETERS ## 
α = nothing #0.9
L = 11
T = 20
ε = 1e-4
normalize_ρ = true
multithread = false

savedir = joinpath(@__DIR__, "data")
perf_savepath = joinpath(savedir, "performance_L$L.csv")
df_performance = load_performance_dataframe(perf_savepath)

function mpdo_circuit_run(; ε, α, L, T, max_outer_dim, max_inner_dim, normalize_ρ::Bool, multithread::Bool)

    # Initialize the wavefunction to product state (all 0)
    ψ0 = initialize_wavefunction(L=L)

    @show ε, max_inner_dim, max_outer_dim
    # Apply the MPDO circuit
    all_results = @timed apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=max_outer_dim,
        max_inner_dim=max_inner_dim, benchmark=true, normalize_ρ=normalize_ρ)

    # extract the results and update performance stats 
    ψ, state_entanglement, operator_entanglement, logneg, MI, trace = all_results.value
    update_performance!(df_performance, L=L, ε=ε,
        max_outer_dim=max_outer_dim, max_inner_dim=max_inner_dim, results=all_results)

    # make a results struct and save it 
    results = Results(0, L, T, ε, max_outer_dim, max_inner_dim, state_entanglement, operator_entanglement, trace, logneg, MI)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
    savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_outer_dim)max_outer_dim_$(max_inner_dim)max_inner_dim" * timestamp * ".h5")
    save_structs(results, savepath)
    CSV.write(perf_savepath, df_performance)
end

function mpo_circuit_run(; ε, α, L, T, max_outer_dim, normalize_ρ::Bool=true)

    # Initialize the wavefunction to product state (all 0)
    ψ0 = initialize_wavefunction(L=L)

    # Apply the MPO circuit
    all_results = @timed apply_circuit(ψ0, T, ε=ε, maxdim=max_outer_dim^2,
        benchmark=true, normalize_ρ=normalize_ρ)

    # extract the results and update performance stats 
    ψ, state_entanglement, operator_entanglement, logneg, MI, trace = all_results.value
    update_performance!(df_performance, L=L, ε=ε,
        max_outer_dim=max_outer_dim, max_inner_dim=nothing, results=all_results)

    # make a results struct and save it 
    results = Results(0, L, T, ε, maxlinkdim(ψ), 0, state_entanglement, operator_entanglement, trace, logneg, MI)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
    savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_outer_dim)max_outer_dim_" * timestamp * ".h5")
    save_structs(results, savepath)
    CSV.write(perf_savepath, df_performance)
end

# actually run the script 

# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=15, normalize_ρ=normalize_ρ)
# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=25, normalize_ρ=normalize_ρ)
# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=30, normalize_ρ=normalize_ρ)
# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=32, normalize_ρ=normalize_ρ)

mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=15, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=25, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=30, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=32, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)

# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=14, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)
# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=14, max_inner_dim=4, normalize_ρ=normalize_ρ, multithread=multithread)
# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=14, max_inner_dim=8, normalize_ρ=normalize_ρ, multithread=multithread)
# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=14, normalize_ρ=normalize_ρ)

# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=15, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)
# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=15, max_inner_dim=4, normalize_ρ=normalize_ρ, multithread=multithread)
# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=15, max_inner_dim=8, normalize_ρ=normalize_ρ, multithread=multithread)
# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=15, normalize_ρ=normalize_ρ)

# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=16, max_inner_dim=1, normalize_ρ=normalize_ρ, multithread=multithread)
# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=16, max_inner_dim=4, normalize_ρ=normalize_ρ, multithread=multithread)
# mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=16, max_inner_dim=8, normalize_ρ=normalize_ρ, multithread=multithread)
# mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=16, normalize_ρ=normalize_ρ)



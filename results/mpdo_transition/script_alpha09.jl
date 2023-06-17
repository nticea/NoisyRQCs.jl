## IMPORTS ##
using Pkg
Pkg.activate(joinpath(@__DIR__, "../../"))
include("../../src/MPDO.jl")
include("../../src/circuit.jl")
include("../../src/results.jl")
using Dates

ITensors.set_warn_order(50)

## PARAMETERS ## 
α = nothing #0.9
L = 11
T = 20
ε = 1e-3
normalize_ρ = true
multithread = false

savedir = joinpath(@__DIR__, "data")

function mpdo_circuit_run(; ε, α, L, T, max_outer_dim, max_inner_dim, normalize_ρ::Bool, multithread::Bool)

    # Initialize the wavefunction to product state (all 0)
    ψ0 = initialize_wavefunction(L=L)
    GC.gc()

    @show ε, max_inner_dim, max_outer_dim
    # Apply the MPDO circuit
    ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit_mpdo(ψ0, T, ε=ε, maxdim=max_outer_dim,
        max_inner_dim=max_inner_dim, benchmark=true, normalize_ρ=normalize_ρ, multithread=multithread)

    # make a results struct and save it 
    results = Results(L, T, ε, max_outer_dim, max_inner_dim, state_entanglement, operator_entanglement, trace, logneg, MI)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
    savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_outer_dim)max_outer_dim_$(max_inner_dim)max_inner_dim" * timestamp * ".h5")
    save_structs(results, savepath)

end

function mpo_circuit_run(; ε, α, L, T, max_outer_dim, normalize_ρ::Bool=true)

    # Initialize the wavefunction to product state (all 0)
    ψ0 = initialize_wavefunction(L=L)
    GC.gc()

    # Apply the MPO circuit
    ψ, state_entanglement, operator_entanglement, logneg, MI, trace = @time apply_circuit(ψ0, T, ε=ε, maxdim=max_outer_dim^2,
        benchmark=true, normalize_ρ=normalize_ρ)

    # make a results struct and save it 
    results = Results(L, T, ε, maxlinkdim(ψ), 0, state_entanglement, operator_entanglement, trace, logneg, MI)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH:MM:SS")
    savepath = joinpath(savedir, "results_$(L)Nx_$(α)α_$(max_outer_dim)max_outer_dim_" * timestamp * ".h5")
    save_structs(results, savepath)
end

# actually run the script 
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=30, max_inner_dim=50, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=30, max_inner_dim=58, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=30, max_inner_dim=64, normalize_ρ=normalize_ρ, multithread=multithread)
mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=30, normalize_ρ=normalize_ρ)

mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=31, max_inner_dim=50, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=31, max_inner_dim=58, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=31, max_inner_dim=64, normalize_ρ=normalize_ρ, multithread=multithread)
mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=31, normalize_ρ=normalize_ρ)

mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=32, max_inner_dim=50, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=32, max_inner_dim=58, normalize_ρ=normalize_ρ, multithread=multithread)
mpdo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=32, max_inner_dim=64, normalize_ρ=normalize_ρ, multithread=multithread)
mpo_circuit_run(α=α, ε=ε, L=L, T=T, max_outer_dim=32, normalize_ρ=normalize_ρ)



